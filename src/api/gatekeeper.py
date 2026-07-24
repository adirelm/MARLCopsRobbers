"""ApiGatekeeper — the outbound-egress governor (§5 REQUIRED, T5.9).

The governed-egress seam. ``execute(channel, call)`` runs a per-channel token bucket
(rate + burst from the versioned ``config/rate_limits.json``) — admits up to the burst,
then a FIFO overflow queue absorbs the excess (NO crash) and DRAINS as tokens refill; a
full queue (``max_queue``) rejects with an explicit error and logs it. Every call is
logged (``log_all_calls``); the clock is injectable for deterministic tests;
``get_queue_status`` reports per-channel queue depth. At runtime the graded **Gmail
report** send routes through it (``reporting/send.py``); ``bearer_get``/``bearer_post`` are
the governed HTTP-egress path (peer-MCP traffic itself uses the FastMCP client transport).

Input: the versioned rate-limit JSON (``_validate_config`` -> ValueError) + an injectable
clock; per call, ``(channel, thunk)`` (``_validate_input`` -> TypeError).
Output: the thunk's result when admitted, the ``DEFERRED`` sentinel when queued, plus the
per-channel queue depths from ``get_queue_status``.
Setup: ``ApiGatekeeper()`` reads ``config/rate_limits.json``; pass ``path``/``clock`` to
inject a test spec + a deterministic clock (no network, no global state).

Concurrency (T5.9 wave-3): FastMCP dispatches on worker threads, so every bucket/queue
TRANSACTION is one-lock guarded (burst can't be over-admitted; ``get_queue_status`` is a
consistent snapshot), yet admitted thunks RUN OUTSIDE the lock — a thunk may re-enter
``execute`` and re-taking the lock would self-deadlock. The overflow queue is BEST-EFFORT,
NOT a background worker: it flushes only when a LATER ``execute`` frees tokens, so every
caller treats ``DEFERRED`` as terminal (send returns ``reason=deferred``; the peer RAISES).
A drained call is a PRIOR caller's, so its failure is caught + logged (``_run_isolated``),
never raised into the unrelated caller that triggered the drain.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "rate_limits.json"
_log = logging.getLogger("marl.api.gatekeeper")

# Returned (instead of a silent None) when a call is deferred to the FIFO overflow
# queue: the caller MUST treat it as "not yet run" (the call executes on a later
# drain). Egress through the gatekeeper is fire-and-forget; a caller needing the
# response synchronously should check get_queue_status() first or run within burst.
DEFERRED = object()


def _validate_config(spec: dict) -> dict:
    """Return the rate-limit ``spec`` after checking every knob the governor reads.

    Raises:
        ValueError: If ``limits`` / ``max_queue`` are absent, ``max_queue`` is < 1, or a
            channel omits (or non-positively sets) ``per_minute`` / ``burst``.
    """
    for key in ("limits", "max_queue"):
        if key not in spec:
            raise ValueError(f"rate-limit config must define {key!r}")
    if int(spec["max_queue"]) < 1:
        raise ValueError(f"rate-limit config max_queue must be >= 1, got {spec['max_queue']!r}")
    for channel, limit in spec["limits"].items():
        for key in ("per_minute", "burst"):
            if key not in limit:
                raise ValueError(f"egress channel {channel!r} must define {key!r}")
        if float(limit["per_minute"]) <= 0:
            raise ValueError(f"egress channel {channel!r} per_minute must be > 0")
        if int(limit["burst"]) < 1:
            raise ValueError(f"egress channel {channel!r} burst must be >= 1")
    return spec


def _validate_input(channel: object, call: object) -> None:
    """Type-check the public ``execute`` arguments.

    Raises:
        TypeError: If ``channel`` is not a str or ``call`` is not callable.
    """
    if not isinstance(channel, str):
        raise TypeError(f"channel must be a str, got {type(channel).__name__}")
    if not callable(call):
        raise TypeError(f"call must be a zero-arg callable, got {type(call).__name__}")


class _TokenBucket:
    """A refilling token bucket: ``per_minute`` tokens/min, capacity ``burst``."""

    def __init__(self, per_minute: float, burst: int, clock: Callable[[], float]) -> None:
        """Start full (``burst`` tokens) and refill at ``per_minute/60`` tokens/sec."""
        self._rate = float(per_minute) / 60.0
        self._capacity = float(burst)
        self._tokens = float(burst)
        self._clock = clock
        self._last = clock()

    def try_consume(self) -> bool:
        """Refill by elapsed time, then consume one token if available."""
        now = self._clock()
        self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._rate)
        self._last = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


class ApiGatekeeper:
    """The single egress governor: token-bucket admission + FIFO overflow queue."""

    def __init__(self, path: str | Path | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        """Load the per-channel limits + ``max_queue`` from the versioned config."""
        spec = _validate_config(json.loads(Path(path or _DEFAULT_PATH).read_text(encoding="utf-8")))
        self._buckets = {
            ch: _TokenBucket(c["per_minute"], c["burst"], clock) for ch, c in spec["limits"].items()
        }
        self._queues: dict[str, deque] = {ch: deque() for ch in self._buckets}
        self._max_queue = int(spec["max_queue"])
        self._lock = threading.Lock()  # guards each bucket/queue transaction (thunks run OUTSIDE it)

    def execute(self, channel: str, call: Callable[[], object]) -> object:
        """Run ``call`` now if a token is free (FIFO-fair), else enqueue (or reject if full).

        Args:
            channel: One of the configured egress channels (``peer_mcp`` / ``gmail``
                ).
            call: A zero-arg thunk performing the outbound side effect.

        Returns:
            The call's result when run immediately, or the :data:`DEFERRED` sentinel
            when the call is enqueued for a later drain (NOT a result — the caller
            must check for ``is DEFERRED``).

        Raises:
            TypeError: On a wrong-typed ``channel`` / ``call`` (§16 input guard).
            KeyError: On an unknown channel.
            RuntimeError: When the channel's FIFO queue is full (``max_queue``).
        """
        _validate_input(channel, call)
        if channel not in self._buckets:
            raise KeyError(f"unknown egress channel {channel!r}")
        # DECIDE under the lock (atomic token/queue transaction); EXECUTE thunks after
        # releasing it — a thunk may itself call execute() and must not re-enter the lock.
        with self._lock:
            drained = self._drain_collect(channel)
            if not self._queues[channel] and self._buckets[channel].try_consume():
                outcome = "run"
            elif len(self._queues[channel]) >= self._max_queue:
                outcome = "reject"
            else:
                self._queues[channel].append(call)
                _log.info("egress channel=%s status=queued depth=%d", channel, len(self._queues[channel]))
                outcome = "defer"
        for queued in drained:  # FIFO: prior deferrals run before this call; failures isolated
            self._run_isolated(channel, queued)
        if outcome == "reject":
            _log.error("egress channel=%s status=rejected reason=queue_full", channel)
            raise RuntimeError(f"egress channel {channel!r} queue full (max_queue={self._max_queue})")
        if outcome == "run":
            return self._run(channel, call)
        return DEFERRED

    def _drain_collect(self, channel: str) -> list[Callable[[], object]]:
        """Pop queued calls FIFO while tokens are available (charges tokens; does NOT run them)."""
        queue, bucket = self._queues[channel], self._buckets[channel]
        drained: list[Callable[[], object]] = []
        while queue and bucket.try_consume():
            drained.append(queue.popleft())
        return drained

    def _run(self, channel: str, call: Callable[[], object]) -> object:
        """Execute one admitted call (``log_all_calls``)."""
        _log.info("egress channel=%s status=executed", channel)
        return call()

    def _run_isolated(self, channel: str, call: Callable[[], object]) -> None:
        """Run a DRAINED (prior-deferred) call, absorbing its failure so no later caller is corrupted."""
        try:
            self._run(channel, call)
        except Exception:
            _log.exception("egress channel=%s status=drained_call_failed", channel)

    def get_queue_status(self) -> dict[str, int]:
        """Return the per-channel FIFO overflow-queue depth (a consistent locked snapshot)."""
        with self._lock:
            return {channel: len(queue) for channel, queue in self._queues.items()}
