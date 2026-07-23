"""Synthetic wire-log builder for the replay escalation/retry tests.

Drives the REAL :class:`CopsRobbersEnv` under a chosen seed with a fixed scripted rule
(each role takes its first legal directional move) and records every hello + per-tick
request/response pair in the referee's JSONL shape, plus the matching §9.4-shaped
record — so tests can fabricate fully VALID logs for ANY seed (e.g. a legitimately
escalated pair replayed under a spare seed, which the committed rehearsal never hit).
"""

from __future__ import annotations

import json

from src.marl.env.actions import Action
from src.marl.env.cops_robbers_env import CopsRobbersEnv

_MOVES = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)


def _line(direction: str, label: str, **fields) -> str:
    """Return one JSONL log line in the referee's on_event shape."""
    return json.dumps({"direction": direction, "label": label, **fields})


def synth_session(cfg: dict, sid: str, seed: int, result_event: bool = False) -> tuple[list[str], dict]:
    """Play one full scripted sub-game under ``seed``; return (jsonl lines, §9.4 record).

    With ``result_event`` the referee's ``result`` JSONL event (carrying the EXACT
    ``seed`` + ``session_id``, like :meth:`WireReferee.play_match` logs) is appended —
    the replay's PRIMARY seed source; without it the log mimics a pre-seed-event log.
    """
    grid, gid = int(cfg["game"]["grid_size"]), int(sid.rsplit("-", 1)[1]) + 1
    env = CopsRobbersEnv(cfg, h=grid, w=grid, num_cops=1)
    _obs, info = env.reset(seed=seed)
    state, lines = env.state(), []
    shared = {"session_id": sid, "grid": [grid, grid], "max_moves": int(cfg["game"]["max_moves"])}
    for role, pos in (("cop", state.cop_pos[0]), ("thief", state.thief_pos)):
        hello = {**shared, "your_role": role, "your_pos": [pos[0], pos[1]]}
        lines += [
            _line("request", f"g-{role}", url="http://x/new_sub_game", payload=hello),
            _line("response", f"g-{role}", response={"ok": True}),
        ]
    terminated, tick = False, 0
    while not terminated:
        pos = {"cop": state.cop_pos[0], "thief": state.thief_pos}
        left, acts = int(cfg["game"]["max_barriers"]) - state.barriers_used, {}
        for role in ("cop", "thief"):
            mask = info["action_mask"]["cop_0" if role == "cop" else "thief"]
            acts[role] = next(m for m in _MOVES if mask[int(m)])
            payload = {
                "session_id": sid,
                "tick": tick,
                "your_pos": list(pos[role]),
                "opponent_pos": None,
                "barriers": [],
                "barriers_left": left,
            }
            lines += [
                _line("request", f"g-{role}", url="http://x/request_move", payload=payload),
                _line("response", f"g-{role}", response={"action": acts[role].name.lower()}),
            ]
        _obs, _r, terminated, info = env.step({"cop_0": acts["cop"], "thief": acts["thief"]})
        state, tick = env.state(), tick + 1
    record = {
        "id": gid,
        "moves": tick,
        "winner": info["winner"],
        "scores": {k: int(v) for k, v in info["scores"].items()},
    }
    if result_event:  # the referee's shape: {"direction": "result", "sub_game": {...full record...}}
        lines.append(
            json.dumps({"direction": "result", "sub_game": {**record, "seed": int(seed), "session_id": sid}})
        )
    return lines, record


def write_match(
    tmp_path, cfg: dict, games: list[tuple[str, int]], result_events: bool = False
) -> tuple[object, object]:
    """Write a synthetic (log.jsonl, records.json) for ``[(sid, seed), ...]``; return paths."""
    lines, records = [], []
    for sid, seed in games:
        game_lines, record = synth_session(cfg, sid, seed, result_event=result_events)
        lines += game_lines
        records.append(record)
    log = tmp_path / "wire_log_synth.jsonl"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    recs = tmp_path / "records_synth.json"
    recs.write_text(json.dumps({"sub_games": records}), encoding="utf-8")
    return log, recs
