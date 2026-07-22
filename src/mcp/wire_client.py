"""§9 wire client — bearer JSON-over-HTTP to ONE partner endpoint (brief §2 + P8).

The transport half of ``docs/interfaces/partner_agent_brief.md``. Raw egress goes through
``src.api.http_client`` — the single sanctioned httpx wrapper (V3 §5 egress boundary; this
module never imports httpx). The token-bucket gatekeeper is deliberately NOT in this path:
its DEFERRED sentinel cannot honor the brief's synchronous 10 s per-move window, and the
match's ~2 req/s is already the configured ``peer_mcp`` sustained rate. A timeout, network
fault, non-2xx status, or non-JSON reply triggers exactly ONE re-POST of the IDENTICAL
body (safe by the brief's ``(session_id, tick)`` idempotency rule), then
:class:`VoidSubGame` — the §3.7 technical void the referee's :class:`SeedSchedule`
consumes. ``GET /health`` needs no auth on the partner side; sending the header is
harmless. Every request/response/error is emitted to the ``on_event`` hook so the referee
can write the per-request JSONL log — the match's shareable fairness artifact.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from src.api.http_client import bearer_get, bearer_post


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
        label: str = "",
        on_event: Callable[[dict], None] | None = None,
        post_fn: Callable = bearer_post,
        get_fn: Callable = bearer_get,
    ) -> None:
        """Bind the endpoint, bearer token, and P8 timeout/retry budget.

        Args:
            base_url: Partner endpoint base URL (paths are appended to it).
            token: Bearer token VALUE (resolved from the env var config NAMES).
            timeout_s: Per-request budget in seconds (``wire_match.timeout_s``).
            retries: Re-POSTs after a fault (``wire_match.retries``; the brief fixes 1).
            label: Log label for the JSONL url-role field, e.g. ``group_2-cop``.
            on_event: Optional hook receiving request/response/error event dicts.
            post_fn: POST egress callable ``(url, token, payload, timeout) -> response``;
                defaults to the sanctioned ``bearer_post`` (tests inject fakes).
            get_fn: GET egress callable ``(url, token, timeout) -> response``.
        """
        self._base = base_url.rstrip("/")
        self._token = token
        self._timeout_s = float(timeout_s)
        self._retries = int(retries)
        self._label = label
        self._on_event = on_event
        self._post_fn = post_fn
        self._get_fn = get_fn

    def health(self) -> bool:
        """GET ``/health``; True iff the endpoint answers ``status: ok`` (never raises)."""
        try:
            resp = self._get_fn(self._base + "/health", self._token, self._timeout_s)
            return bool(resp.is_success) and resp.json().get("status") == "ok"
        except Exception:  # any transport/parse failure just means "not healthy"
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
                resp = self._post_fn(url, self._token, payload, self._timeout_s)
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


class SeedSchedule:
    """P7 seed schedule + the agreed void amendment — the consumer of :class:`VoidSubGame`.

    Kept beside the client so the P8 fault pair lives together: the client RAISES the
    void, this schedule decides what it means for the match. Amendment (agreed before
    implementation): a void replays the SAME sub-game with the SAME seed; only after
    ``max_void_replays`` CONSECUTIVE voids of one sub-game does the next unused spare
    seed replace s_k for the whole pair k/k+3, replaying the base game if already played.
    """

    def __init__(self, seeds: list, num_games: int, max_void_replays: int) -> None:
        """Freeze the agreed ORDERED list: first ``num_games/2`` = pair seeds, rest = spares."""
        if len(seeds) < num_games:
            raise ValueError(f"P7 requires >= {num_games} jointly agreed seeds, got {len(seeds)}")
        self._pairs = num_games // 2  # §9.1: ids 1..3 mirror 4..6 -> one seed per pair
        self._seeds = [int(s) for s in seeds]
        self._pair_seed = {k: self._seeds[k] for k in range(self._pairs)}
        self._spare = self._pairs
        self._pending = list(range(1, num_games + 1))
        self._done: set[int] = set()
        self._voids = 0
        self._max_voids = int(max_void_replays)

    def next_game(self) -> tuple[int, int] | None:
        """Return ``(game_id, seed)`` for the next sub-game, or None when the match is done."""
        if not self._pending:
            return None
        return self._pending[0], self._pair_seed[(self._pending[0] - 1) % self._pairs]

    def record_result(self, game_id: int) -> None:
        """Mark ``game_id`` validly completed; the consecutive-void counter resets."""
        self._pending.remove(game_id)
        self._done.add(game_id)
        self._voids = 0

    def record_void(self, game_id: int) -> list[int]:
        """Register one technical void; return the ids whose completed records became stale.

        Below the threshold: replay the SAME sub-game, SAME seed (empty list). At the
        threshold: the next spare seed replaces s_k for the pair k/k+3, and an
        already-played base game is re-queued FIRST (and returned as stale).

        Raises:
            RuntimeError: When escalation is required but every spare seed is used up.
        """
        self._voids += 1
        if self._voids < self._max_voids:
            return []
        if self._spare >= len(self._seeds):
            raise RuntimeError(f"P7 spare seeds exhausted while voiding sub-game {game_id}")
        pair = (game_id - 1) % self._pairs
        self._pair_seed[pair] = self._seeds[self._spare]
        self._spare += 1
        self._voids = 0
        base = pair + 1
        if base != game_id and base in self._done:
            self._done.discard(base)
            self._pending.insert(0, base)
            return [base]
        return []
