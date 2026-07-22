"""Shared stubs for the §9 wire-referee tests — no sockets, PLACEHOLDER identity only.

``SEEDS`` were chosen so the cop and thief spawn in DIFFERENT columns on every seed:
with the default stub policies (cop always ``up``, thief always ``down``) neither agent
ever changes column, so capture is impossible and every clean sub-game deterministically
ends as a 25-move thief timeout. Seed 5 (the first spare) also spawns differently from
seed 2, which the escalation test relies on to detect the replacement layout.
"""

from __future__ import annotations

from src.mcp.wire_client import VoidSubGame
from src.utils.config_loader import load_config

SEEDS = [1, 2, 3, 5, 6, 7, 8]  # first 3 = pair seeds for (1,4)/(2,5)/(3,6); rest = spares
STUDENTS = [{"role": "A", "full_name": "Placeholder One", "id": "12345"}]
REPOS = ("https://github.com/example/ours", "https://github.com/example/theirs")


def wire_cfg(seeds: list | None = None) -> dict:
    """Return a fresh config copy with the P7 seed list filled for tests."""
    cfg = load_config()
    cfg["wire_match"]["seeds"] = list(SEEDS if seeds is None else seeds)
    return cfg


class StubAgent:
    """Scripted partner endpoint covering BOTH roles of one group (no HTTP).

    ``actions`` maps role -> action string OR callable(payload) -> action string.
    ``fail_first`` maps session_id -> number of tick-0 calls that raise VoidSubGame,
    simulating the wire client's exhausted-retry fault path (P8).
    """

    def __init__(self, actions: dict | None = None, fail_first: dict | None = None) -> None:
        """Bind the scripted per-role policy and the scripted fault plan."""
        self.actions = actions or {"cop": "up", "thief": "down"}
        self.fail_first = dict(fail_first or {})
        self.roles: dict[str, str] = {}
        self.new_calls: list[dict] = []
        self.move_calls: list[dict] = []

    def new_sub_game(self, payload: dict) -> dict:
        """Record the reset call, remember the session's role, ack ok."""
        self.roles[payload["session_id"]] = payload["your_role"]
        self.new_calls.append(dict(payload))
        return {"ok": True}

    def request_move(self, payload: dict) -> dict:
        """Record the move call; raise a scripted fault or answer the scripted action."""
        self.move_calls.append(dict(payload))
        sid = payload["session_id"]
        if payload["tick"] == 0 and self.fail_first.get(sid, 0) > 0:
            self.fail_first[sid] -= 1
            raise VoidSubGame("scripted fault")
        action = self.actions[self.roles[sid]]
        return {"action": action(payload) if callable(action) else action}


def stub_clients(g1: StubAgent, g2: StubAgent) -> dict:
    """Wire two group stubs into the referee's clients mapping (one stub per group)."""
    return {"group_1": {"cop": g1, "thief": g1}, "group_2": {"cop": g2, "thief": g2}}


class FakeResp:
    """Minimal response stand-in for the WireClient's injected egress callables."""

    def __init__(self, status: int = 200, body: dict | None = None) -> None:
        """Bind a status plus an optional JSON body (None -> ``json()`` raises ValueError)."""
        self.status_code = status
        self.is_success = 200 <= status < 300
        self._body = body

    def json(self) -> dict:
        """Return the JSON body, or raise ValueError when the reply is not JSON."""
        if self._body is None:
            raise ValueError("not json")
        return self._body
