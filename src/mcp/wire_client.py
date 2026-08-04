"""§9 wire client — bearer JSON-over-HTTP to ONE partner endpoint (brief §2 + P8).

The transport half of ``docs/interfaces/partner_agent_brief.md``. Raw egress goes through
``src.api.http_client`` — the single sanctioned httpx wrapper (V3 §5 egress boundary; this
module never imports httpx). The token-bucket gatekeeper is deliberately NOT in this path:
its DEFERRED sentinel cannot honor the brief's synchronous 10 s per-move window, and the
match's ~2 req/s is already the configured ``peer_mcp`` sustained rate. Each egress call
(POST *and* the ``health`` GET) runs on a daemon worker joined at the WALL-CLOCK budget —
httpx's scalar timeout is only per-phase, so a peer dribbling bytes inside every read
window could otherwise hang the referee unboundedly; an expired worker is abandoned and
its boxed result discarded. Abandoned orphans are BOUNDED: a per-client in-flight cap
fails fast once too many workers are still winding down, so repeated dribbling voids can
never accumulate threads/sockets without limit. A timeout, network fault, non-2xx status,
or non-JSON reply triggers exactly ONE re-POST of the IDENTICAL body (safe by the brief's
``(session_id, tick)`` idempotency rule), then :class:`VoidSubGame` — the §3.7 technical
void the referee's :class:`~src.mcp.wire_schedule.SeedSchedule` (re-exported here)
consumes. Every request/response/error is emitted to the ``on_event`` hook so the referee
can write the per-request JSONL log — the match's shareable fairness artifact.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from src.api.http_client import bearer_get, bearer_post
from src.mcp.wire_schedule import SeedSchedule  # noqa: F401 — public re-export (import surface kept)

# Safety cap on concurrently-alive egress workers per client. A live call joins within the
# budget (inflight ~1); only ABANDONED orphans keep it elevated, so this bounds the orphans
# that repeated dribbling voids can leave behind. Injectable for deterministic tests.


class VoidSubGame(Exception):  # noqa: N818 — named for the §3.7 OUTCOME (a void), not an error kind
    """The current sub-game is technically void (P8: per-move budget + one retry exhausted)."""


def _ms(started: float) -> float:
    """Return elapsed milliseconds since ``started`` (a ``time.monotonic`` stamp)."""
    return round((time.monotonic() - started) * 1000.0, 3)


class WireClient:
    """Bearer-authed HTTP client for one partner endpoint base URL (cop or thief)."""

    def __init__(  # noqa: PLR0913 — the transport knobs are all distinct config inputs
        self,
        base_url: str,
        token: str,
        timeout_s: float,
        retries: int,
        max_inflight: int,
        label: str = "",
        on_event: Callable[[dict], None] | None = None,
        post_fn: Callable = bearer_post,
        get_fn: Callable = bearer_get,
    ) -> None:
        """Bind the endpoint, bearer token, and P8 timeout/retry budget.

        Args:
            base_url: Partner endpoint base URL (paths are appended to it).
            token: Bearer token VALUE (resolved from the env var config NAMES).
            timeout_s: Per-attempt WALL-CLOCK budget in seconds (``wire_match.timeout_s``);
                httpx's scalar timeout is only per-phase, so a late-landing reply still faults.
            retries: Re-POSTs after a fault (``wire_match.retries``; the brief fixes 1).
            label: Log label for the JSONL url-role field, e.g. ``group_2-cop``.
            on_event: Optional hook receiving request/response/error event dicts.
            post_fn: POST egress callable ``(url, token, payload, timeout) -> response``;
                defaults to the sanctioned ``bearer_post`` (tests inject fakes).
            get_fn: GET egress callable ``(url, token, timeout) -> response``.
            max_inflight: Per-client cap on concurrently-alive egress workers (fail-fast bound).
        """
        self._base, self._token = base_url.rstrip("/"), token
        self._timeout_s = float(timeout_s)
        self._retries = int(retries)
        self._label = label
        self._on_event = on_event
        self._post_fn = post_fn
        self._get_fn = get_fn
        self._max_inflight = int(max_inflight)
        self._inflight = 0
        self._inflight_lock = threading.Lock()

    def health(self) -> bool:
        """GET ``/health`` under the SAME wall-clock budget; True iff ``status: ok`` in time.

        Routed through :meth:`_bounded_call` so a dribbling/hung partner can never make the
        liveness probe outlast the P8 budget — an expiry just reads as "not healthy".
        """
        try:
            resp = self._bounded_call(
                lambda: self._get_fn(self._base + "/health", self._token, self._timeout_s)
            )
            return bool(resp.is_success) and resp.json().get("status") == "ok"
        except Exception:  # any transport/parse/timeout failure just means "not healthy"
            return False

    def new_sub_game(self, payload: dict) -> dict:
        """POST ``/new_sub_game`` (sub-game start / void-replay reset); return the JSON reply."""
        return self._post("/new_sub_game", payload)

    def request_move(self, payload: dict) -> dict:
        """POST ``/request_move`` for one tick; return the JSON reply (``{"action": ...}``)."""
        return self._post("/request_move", payload)

    def _emit(self, event: dict) -> None:
        """Send one log event to the hook (no-op when unhooked)."""
        if self._on_event is not None:
            self._on_event(event)

    def _bounded_call(self, thunk: Callable[[], object]) -> object:
        """Run ``thunk`` on a daemon worker; enforce the wall clock via ``join``; cap orphans.

        On expiry the worker is abandoned (daemon — it may finish later, but its result
        lands only in the discarded local ``box``, so it can corrupt nothing) and
        ``TimeoutError`` is raised IMMEDIATELY, keeping P8's budget even against a dribbling
        peer that feeds bytes inside every per-phase httpx read window. ``_inflight`` counts
        currently-alive workers (each decrements in its ``finally``): once it reaches
        ``_max_inflight`` a new call fails fast INSTEAD of spawning another orphan, so
        repeated voids cannot grow threads/sockets without bound.
        """
        with self._inflight_lock:
            if self._inflight >= self._max_inflight:
                raise TimeoutError(f"in-flight egress workers at cap {self._max_inflight}; failing fast")
            self._inflight += 1
        box: dict[str, object] = {}

        def _run() -> None:
            try:
                box["out"] = thunk()
            except Exception as exc:  # boxed: re-raised below on the caller thread
                box["exc"] = exc
            finally:
                with self._inflight_lock:
                    self._inflight -= 1

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        worker.join(self._timeout_s)
        if worker.is_alive():
            raise TimeoutError(f"wall clock budget {self._timeout_s}s expired mid-attempt")
        if "exc" in box:
            raise box["exc"]  # type: ignore[misc] — always an Exception when present
        return box["out"]

    def _bounded_post(self, url: str, payload: dict) -> object:
        """Run one blocking POST under the shared wall-clock + orphan-cap guard."""
        return self._bounded_call(lambda: self._post_fn(url, self._token, payload, self._timeout_s))

    def _post(self, path: str, payload: dict) -> dict:
        """POST ``payload``; ONE idempotent re-POST of the SAME body on a fault, then void.

        Raises:
            VoidSubGame: When the initial attempt and every allowed retry faulted (P8).
        """
        url = self._base + path
        last: Exception | None = None
        for attempt in range(self._retries + 1):
            base = {"label": self._label, "url": url, "attempt": attempt}
            self._emit({"direction": "request", **base, "payload": payload})
            started = time.monotonic()
            try:
                resp = self._bounded_post(url, payload)
                if (elapsed := time.monotonic() - started) > self._timeout_s:  # second line of defence
                    raise TimeoutError(f"wall clock {elapsed:.3f}s > {self._timeout_s}s budget")
                if not resp.is_success:
                    raise RuntimeError(f"HTTP {resp.status_code}")
                body = resp.json()
            except Exception as exc:  # P8: ANY failure to get valid JSON within budget is a fault
                last = exc
                self._emit({"direction": "error", **base, "error": repr(exc), "latency_ms": _ms(started)})
                continue
            self._emit({"direction": "response", **base, "response": body, "latency_ms": _ms(started)})
            return body
        raise VoidSubGame(f"{self._label} {path}: fault after {self._retries + 1} attempt(s): {last!r}")
