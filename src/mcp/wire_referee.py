"""§9 wire referee — drives the bonus match over HTTP; the env is the sole ground truth.

Implements ``docs/interfaces/partner_agent_brief.md`` exactly: the §9.1 role alternation
(sub-games 1-3 group_1 cop, 4-6 swapped), the P7 seed schedule with the agreed void
amendment (:class:`SeedSchedule`), P5 radius-2 masking with ``barriers_left`` to both
roles, P8 one-retry-then-void, P2 capture-before-timeout via the env transition, §3.5
Jerusalem millisecond records (feeding ``build_bonus_report``), + the per-request JSONL log.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.marl.env.actions import Action
from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.mcp.wire_client import SeedSchedule, VoidSubGame
from src.reporting.bonus import build_bonus_report, derive_totals_by_group, validate_bonus
from src.reporting.players import load_players

_ROOT = Path(__file__).resolve().parents[2]
_THIEF_ACTIONS = {"up": Action.UP, "down": Action.DOWN, "left": Action.LEFT, "right": Action.RIGHT}
_COP_ACTIONS = {**_THIEF_ACTIONS, "place_barrier": Action.PLACE_BARRIER}


def _l1(a, b) -> int:
    """Return the Manhattan distance between two (row, col) cells (the P5 metric)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def mask_payload(  # noqa: PLR0913 — the brief's payload fields are all distinct inputs
    session_id: str, tick: int, you, opponent, barriers, barriers_left: int, radius: int
) -> dict:
    """Build one P5-masked ``request_move`` payload — brief field names VERBATIM."""
    return {
        "session_id": session_id,
        "tick": tick,
        "your_pos": [you[0], you[1]],
        "opponent_pos": [opponent[0], opponent[1]] if _l1(you, opponent) <= radius else None,
        "barriers": sorted([b[0], b[1]] for b in barriers if _l1(b, you) <= radius),
        "barriers_left": int(barriers_left),
    }


class WireReferee:
    """Drive the full §9 match over the wire (referee = the environment, §3.7)."""

    def __init__(self, cfg: dict, log_path: str | Path) -> None:
        """Bind the config-derived rules + the JSONL log path (parent dirs created)."""
        self._cfg = cfg
        self._grid = int(cfg["game"]["grid_size"])
        self._radius = int(cfg["mcp"]["observation"]["view_radius"])
        self._retries = int(cfg["wire_match"]["retries"])
        self._pairs = int(cfg["game"]["num_games"]) // 2  # §9.1: ids 1..k mirror k+1..2k
        self._names = {k: cfg["wire_match"]["groups"][k]["name"] for k in ("group_1", "group_2")}
        self._tz = ZoneInfo(cfg["project"]["timezone"])
        self._env = CopsRobbersEnv(cfg, h=self._grid, w=self._grid, num_cops=1)
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event: dict) -> None:
        """Append one timestamped JSONL entry (also the WireClient ``on_event`` hook)."""
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": self._now(), **event}, ensure_ascii=False) + "\n")

    def _now(self) -> str:
        """Return an ISO-8601 millisecond timestamp in the §3.5 timezone (Asia/Jerusalem)."""
        return datetime.now(self._tz).isoformat(timespec="milliseconds")

    def _roundtrip(self, call: Callable, payload: dict, valid: Callable, what: str) -> dict:
        """Call an endpoint with ONE P8 protocol retry; raise VoidSubGame when all replies malform."""
        reply: object = None
        for _ in range(self._retries + 1):
            reply = call(payload)
            if isinstance(reply, dict) and valid(reply):
                return reply
        raise VoidSubGame(f"{what}: malformed reply {reply!r} after one retry (P8)")

    def _move(self, client, payload: dict, role: str) -> Action:
        """Fetch + validate one move; unknown strings / a thief PLACE are P8 faults."""
        allowed = _COP_ACTIONS if role == "cop" else _THIEF_ACTIONS
        checker = f"{role} request_move tick {payload['tick']}"
        reply = self._roundtrip(client.request_move, payload, lambda r: r.get("action") in allowed, checker)
        return allowed[reply["action"]]

    def _play_sub_game(self, clients: dict, gid: int, seed: int) -> dict:
        """Play sub-game ``gid`` (1..6) from ``seed``; return its §9.4 record (or void)."""
        cop_key = "group_1" if gid <= self._pairs else "group_2"
        thief_key = "group_2" if cop_key == "group_1" else "group_1"
        side = {"cop": clients[cop_key]["cop"], "thief": clients[thief_key]["thief"]}
        start = self._now()
        self._env.reset(seed=seed)
        state, session = self._env.state(), f"sg-{gid - 1}"
        shared = {
            "session_id": session,
            "grid": [self._grid, self._grid],
            "max_moves": int(self._cfg["game"]["max_moves"]),
        }
        for role, pos in (("cop", state.cop_pos[0]), ("thief", state.thief_pos)):
            hello = {**shared, "your_role": role, "your_pos": [pos[0], pos[1]]}
            self._roundtrip(side[role].new_sub_game, hello, lambda r: r.get("ok") is True, "new_sub_game")
        terminated, tick, info = False, 0, {}
        while not terminated:
            pos = {"cop": state.cop_pos[0], "thief": state.thief_pos}
            left, acts = int(self._cfg["game"]["max_barriers"]) - state.barriers_used, {}
            for role in ("cop", "thief"):
                other = pos["thief" if role == "cop" else "cop"]
                masked = mask_payload(session, tick, pos[role], other, state.barriers, left, self._radius)
                acts[role] = self._move(side[role], masked, role)
            _obs, _r, terminated, info = self._env.step({"cop_0": acts["cop"], "thief": acts["thief"]})
            state, tick = self._env.state(), tick + 1
        return {
            "id": gid,
            "seed": int(seed),
            "session_id": session,
            "start": start,
            "end": self._now(),
            "moves": tick,
            "winner": info["winner"],
            "scores": {"cop": int(info["scores"]["cop"]), "thief": int(info["scores"]["thief"])},
            "cop_group": self._names[cop_key],
            "thief_group": self._names[thief_key],
        }

    def play_match(self, clients: dict) -> dict:
        """Play the P7 schedule vs the ``clients`` mapping; return sorted records + totals_by_group."""
        wire = self._cfg["wire_match"]
        schedule = SeedSchedule(wire["seeds"], int(self._cfg["game"]["num_games"]), wire["max_void_replays"])
        records: dict[int, dict] = {}
        while (head := schedule.next_game()) is not None:
            gid, seed = head
            try:
                record = self._play_sub_game(clients, gid, seed)
            except VoidSubGame as exc:
                self.log_event({"direction": "void", "sub_game": gid, "seed": int(seed), "reason": str(exc)})
                for stale in schedule.record_void(gid):
                    records.pop(stale, None)
                continue
            schedule.record_result(gid)
            records[gid] = record
            self.log_event({"direction": "result", "sub_game": record})
        sub_games = [records[gid] for gid in sorted(records)]
        totals = derive_totals_by_group(sub_games, self._names["group_1"], self._names["group_2"])
        return {"sub_games": sub_games, "totals_by_group": totals, "log_path": str(self._log_path)}


def _partner_path() -> Path:
    """Return the partner identity file: the git-ignored local copy, else the placeholder."""
    local = _ROOT / "players.partner.local.yaml"
    return local if local.exists() else _ROOT / "players.partner.example.yaml"


def build_draft_report(cfg: dict, sub_games: list[dict]) -> dict:
    """Assemble + validate the DRAFT §9.4 body from the six match records (brief §5).

    ``mutual_agreement`` stays False by construction here: it may flip to True only after
    the §9.3 byte-compare of both groups' drafts — never as a default.
    """
    ours, partner = load_players(), load_players(_partner_path())
    report = build_bonus_report(
        groups=(ours["group_name"], partner["group_name"]),
        repos=(ours["github_repo"], partner["github_repo"]),
        students=(ours["students"], partner["students"]),
        timezone=cfg["project"]["timezone"],
        results=sub_games,
        game=cfg["game"],
        mutual_agreement=False,
    )
    validate_bonus(report, cfg["game"])
    return report
