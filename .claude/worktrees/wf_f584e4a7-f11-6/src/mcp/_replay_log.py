"""Wire-log parsing for the §9.3 replay (kept out of wire_replay.py for the 150-LOC cap).

Input: the referee's per-request JSONL log path.
Output: per-session dicts — spawns, per-tick action pairs, and per-tick GROUND-TRUTH
request payloads (``your_pos`` / ``barriers_left`` per role) used to verify EVERY replayed
tick, not just the terminal summary.
Setup: none — pure parsing; imported by :mod:`src.mcp.wire_replay`.
"""

from __future__ import annotations

import json
from pathlib import Path

_ROLES = ("cop", "thief")


class ReplayMismatchError(RuntimeError):
    """The deterministic replay diverged from the log/records — the evidence is invalid."""


def gid_of(session_id: str) -> int:
    """Return the 1-based sub-game id for a referee session id (``sg-0`` -> 1)."""
    return int(session_id.rsplit("-", 1)[1]) + 1


def parse_wire_log(path: str | Path) -> dict[str, dict]:
    """Parse the JSONL log into per-session spawns + per-tick actions AND ground truth.

    Returns ``{sid: {"spawns": {role: pos}, "actions": {tick: {role: str}},
    "states": {tick: {role: {"your_pos": pos, "barriers_left": int}}}}`` plus, when the
    referee logged a ``result`` event for the session, ``"seed"`` — the EXACT seed the
    completed run was played under (last result wins, matching the last-hello-wins run).
    A void re-hello supersedes the session's earlier run (spawns, actions AND states are
    wiped) — but an IDENTICAL re-hello (same role, same pos) with NO moves recorded
    since is a P8 TRANSPORT RETRY of the hello itself: an idempotent no-op that must
    NOT wipe the other role's spawn.
    """
    sessions: dict[str, dict] = {}
    pending: dict[str, tuple[str, int, str]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        direction, label = entry.get("direction"), entry.get("label")
        if direction == "request":
            payload = entry["payload"]
            sess = sessions.setdefault(payload["session_id"], {"spawns": {}, "actions": {}, "states": {}})
            if entry["url"].endswith("/new_sub_game"):
                role, pos = payload["your_role"], tuple(payload["your_pos"])
                retry = sess["spawns"].get(role) == pos and not sess["actions"] and not sess["states"]
                if role in sess["spawns"] and not retry:  # void re-hello: last run wins
                    sess["spawns"], sess["actions"], sess["states"] = {}, {}, {}
                sess["spawns"][role] = pos
            else:
                tick, role = int(payload["tick"]), label.rsplit("-", 1)[1]
                sess["states"].setdefault(tick, {})[role] = {
                    "your_pos": tuple(payload["your_pos"]),
                    "barriers_left": int(payload["barriers_left"]),
                }
                pending[label] = (payload["session_id"], tick, role)
        elif direction == "response" and label in pending:
            sid, tick, role = pending.pop(label)
            if isinstance(reply := entry.get("response"), dict) and isinstance(reply.get("action"), str):
                sessions[sid]["actions"].setdefault(tick, {})[role] = reply["action"]
        elif direction == "result" and isinstance(sub := entry.get("sub_game"), dict):
            if "session_id" in sub and "seed" in sub:  # the referee's EXACT per-run seed (last wins)
                sess = sessions.setdefault(sub["session_id"], {"spawns": {}, "actions": {}, "states": {}})
                sess["seed"] = int(sub["seed"])
    return sessions


def ordered_actions(sid: str, sess: dict) -> list[dict[str, str]]:
    """Return the session's action pairs ordered by tick; raise on gaps/missing roles."""
    ticks = sorted(sess["actions"])
    if ticks != list(range(len(ticks))):
        raise ReplayMismatchError(f"{sid}: non-contiguous logged ticks {ticks}")
    pairs = [sess["actions"][t] for t in ticks]
    for tick, pair in enumerate(pairs):
        if set(pair) != set(_ROLES):
            raise ReplayMismatchError(f"{sid} tick {tick}: roles {sorted(pair)} != ['cop', 'thief']")
    return pairs
