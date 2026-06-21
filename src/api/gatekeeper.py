"""ApiGatekeeper — the SINGLE outbound-egress governor (§5 REQUIRED, T5.9).

EVERY outbound API call (peer-MCP HTTP, Gmail, Prefect deploy) routes through
``ApiGatekeeper.execute(channel, call)``: a per-channel token bucket (rate + burst
from the versioned ``config/rate_limits.json``) admits calls up to the burst, then a
FIFO overflow queue absorbs the excess (NO crash) and DRAINS as tokens refill; a
full queue (``max_queue``) rejects the new call with an explicit error and logs it.
Every call is logged (``log_all_calls``). The clock is injectable for deterministic
tests. ``get_queue_status`` reports per-channel queue depth.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "rate_limits.json"
_log = logging.getLogger("marl.api.gatekeeper")


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
        spec = json.loads(Path(path or _DEFAULT_PATH).read_text(encoding="utf-8"))
        self._buckets = {
            ch: _TokenBucket(c["per_minute"], c["burst"], clock) for ch, c in spec["limits"].items()
        }
        self._queues: dict[str, deque] = {ch: deque() for ch in self._buckets}
        self._max_queue = int(spec["max_queue"])

    def execute(self, channel: str, call: Callable[[], object]) -> object:
        """Run ``call`` now if a token is free (FIFO-fair), else enqueue (or reject if full).

        Args:
            channel: One of the configured egress channels (``peer_mcp`` / ``gmail``
                / ``prefect_deploy``).
            call: A zero-arg thunk performing the outbound side effect.

        Returns:
            The call's result when run immediately, or ``None`` when deferred.

        Raises:
            KeyError: On an unknown channel.
            RuntimeError: When the channel's FIFO queue is full (``max_queue``).
        """
        if channel not in self._buckets:
            raise KeyError(f"unknown egress channel {channel!r}")
        self._drain(channel)
        if not self._queues[channel] and self._buckets[channel].try_consume():
            return self._run(channel, call)
        if len(self._queues[channel]) >= self._max_queue:
            _log.error("egress channel=%s status=rejected reason=queue_full", channel)
            raise RuntimeError(f"egress channel {channel!r} queue full (max_queue={self._max_queue})")
        self._queues[channel].append(call)
        _log.info("egress channel=%s status=queued depth=%d", channel, len(self._queues[channel]))
        return None

    def _drain(self, channel: str) -> None:
        """Flush queued calls FIFO while tokens are available."""
        queue, bucket = self._queues[channel], self._buckets[channel]
        while queue and bucket.try_consume():
            self._run(channel, queue.popleft())

    def _run(self, channel: str, call: Callable[[], object]) -> object:
        """Execute one admitted call (``log_all_calls``)."""
        _log.info("egress channel=%s status=executed", channel)
        return call()

    def get_queue_status(self) -> dict[str, int]:
        """Return the per-channel FIFO overflow-queue depth."""
        return {channel: len(queue) for channel, queue in self._queues.items()}
