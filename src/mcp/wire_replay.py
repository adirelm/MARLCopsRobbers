"""§9.3 wire-match replay — re-run the logged match deterministically, render evidence.

Works from exactly (JSONL log + §9.4 records + the P7 ``wire_match.seeds`` k/k+3 schedule).
ANY divergence of the re-run :class:`CopsRobbersEnv` from the log/records raises
:class:`ReplayMismatchError` loudly — a silently diverging replay is worthless as §9.3
evidence. A void re-hello supersedes the session's earlier run; a P8 retry keeps the last
valid reply. Each sub-game's seed comes PRIMARILY from the referee's logged ``result``
event (spawn-verified against both hellos); for logs predating seed events an ESCALATED
pair (P7: spare after 3 consecutive voids) falls back to spawn-matching s_k then the
spares in order. Mirror games must resolve to ONE seed and no spare may serve two pairs.
PNG rendering lives in :mod:`src.mcp.wire_screens` (re-exported here).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.gui.spectator import SpectatorFrame
from src.marl.env.actions import Action
from src.marl.env.scorer import Scorer
from src.mcp._replay_log import ReplayMismatchError, gid_of, ordered_actions, parse_wire_log
from src.mcp._replay_outcome import verify_body_metadata, verify_published_outcome
from src.mcp._replay_records import (
    agreed_group_names,
    verify_record_scores,
    verify_record_structure,
    verify_result_event,
)
from src.mcp._replay_seed import seeded_env, verify_void_attempts
from src.mcp._replay_verify import verify_escalation_budget, verify_session_voids, verify_tick
from src.mcp.wire_screens import mid_frame_index, save_screens  # noqa: F401 — re-export (150-LOC split)

_MOVES = {a.name.lower(): a for a in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)}
_ALLOWED = {"cop": {**_MOVES, "place_barrier": Action.PLACE_BARRIER}, "thief": _MOVES}
_ROLES = ("cop", "thief")


def _frame(cfg, env, gid, totals, winner, last) -> SpectatorFrame:  # noqa: PLR0913 - one arg per HUD input
    """Snapshot the replayed env into the frozen god-view frame the GUI renders."""
    state = env.state()
    action_names = (
        {"cop_0": _ALLOWED["cop"][last["cop"]].name, "thief": _MOVES[last["thief"]].name} if last else None
    )
    return SpectatorFrame(
        grid=(state.h, state.w),
        cop_positions=tuple(tuple(p) for p in state.cop_pos),
        thief_position=tuple(state.thief_pos),
        barriers=tuple(sorted(state.barriers)),
        view_radius=int(cfg["env"]["view_radius_by_grid"][min(state.h, state.w)]),
        move=state.step,
        max_moves=int(cfg["game"]["max_moves"]),
        sub_game=gid,
        num_games=int(cfg["game"]["num_games"]),
        scores=Scorer(cfg).score(winner) if winner else {"cop": 0, "thief": 0},
        totals=dict(totals),
        winner=winner,
        last_action=action_names,
        max_barriers=int(cfg["game"]["max_barriers"]),
    )


def replay_sub_game(cfg: dict, sid: str, sess: dict, record: dict, totals: dict) -> tuple[list, dict]:
    """Re-run one logged sub-game from its P7 (possibly escalated) seed; every tick verified."""
    gid = gid_of(sid)
    env, seed = seeded_env(cfg, sid, sess, gid)
    verify_void_attempts(cfg, sid, sess, gid)
    verify_session_voids(cfg, sid, int(sess.get("voids", 0)), seed)
    frames, terminated, info = [_frame(cfg, env, gid, totals, None, None)], False, {}
    for tick, pair in enumerate(ordered_actions(sid, sess)):
        if terminated:
            raise ReplayMismatchError(f"{sid}: log continues past the terminal state at tick {tick}")
        if bad := [(role, pair[role]) for role in _ROLES if pair[role] not in _ALLOWED[role]]:
            raise ReplayMismatchError(f"{sid} tick {tick}: illegal logged action(s) {bad}")
        verify_tick(cfg, sid, tick, sess, env.state())
        joint = {"cop_0": _ALLOWED["cop"][pair["cop"]], "thief": _MOVES[pair["thief"]]}
        _obs, _r, terminated, info = env.step(joint)
        frames.append(_frame(cfg, env, gid, totals, info.get("winner"), pair))
    got = {"moves": len(frames) - 1, "winner": info.get("winner"), "scores": info.get("scores")}
    want = {"moves": int(record["moves"]), "winner": record["winner"], "scores": dict(record["scores"])}
    if not terminated or got != want:
        raise ReplayMismatchError(f"{sid}: replay {got} (terminated={terminated}) != record {want}")
    return frames, {"gid": gid, "seed": seed, **got}


def replay_match(
    cfg: dict, log_path: str | Path, records_path: str | Path, full_match: bool = True
) -> list[dict]:
    """Replay every logged sub-game against the records; return per-game frames + summaries.

    ``full_match`` asserts these artifacts ARE a complete §9.1 match (all
    ``game.num_games`` ids). Only unit tests that replay a deliberate subset pass False.
    """
    sessions = parse_wire_log(log_path)
    published = json.loads(Path(records_path).read_text(encoding="utf-8"))
    records = {int(r["id"]): r for r in published["sub_games"]}
    # FIRST: the records carry DERIVED fields the move-replay can never see — the id set, who
    # played cop, the Table-1 row. Ordered ahead of the log/records comparison because a match
    # missing a sub-game fails BOTH, and "the ids are not 1..6" names the defect while "log
    # sub-games != record ids" only says the two agree on being wrong together.
    verify_record_structure(cfg, records, published.get("groups"), full_match)
    if full_match:  # a subset body has no meaningful match TOTALS to re-derive
        verify_body_metadata(published)
        verify_published_outcome(cfg, published, agreed_group_names(cfg, published.get("groups")))
    if sorted(gid_of(s) for s in sessions) != sorted(records):
        raise ReplayMismatchError(f"log sub-games {sorted(sessions)} != record ids {sorted(records)}")
    replays, totals = [], dict.fromkeys(_ROLES, 0)
    for sid in sorted(sessions, key=gid_of):
        verify_record_scores(cfg, gid_of(sid), records[gid_of(sid)])
        verify_result_event(sid, sessions[sid], records[gid_of(sid)])
        frames, summary = replay_sub_game(cfg, sid, sessions[sid], records[gid_of(sid)], totals)
        replays.append({**summary, "frames": frames})
        for role in _ROLES:
            totals[role] += int(summary["scores"][role])
    seed_of, pairs = {g["gid"]: g["seed"] for g in replays}, int(cfg["game"]["num_games"]) // 2
    spares = {int(s) for s in cfg["wire_match"]["seeds"][pairs:]}
    owner: dict[int, int] = {}  # spare seed -> the ONE pair it escalated (SeedSchedule consumes each once)
    for k in range(1, pairs + 1):  # §9.1 mirror consistency: k and k+pairs must share ONE seed
        if {k, k + pairs} <= set(seed_of) and seed_of[k] != seed_of[k + pairs]:
            raise ReplayMismatchError(
                f"mirror pair {k}/{k + pairs} resolved to seeds {seed_of[k]} != {seed_of[k + pairs]}"
            )
        seed = seed_of.get(k, seed_of.get(k + pairs))
        if seed in spares and owner.setdefault(seed, k) != k:
            raise ReplayMismatchError(
                f"spare seed {seed} resolved for pairs {owner[seed]} and {k} — "
                f"the P7 schedule consumes each spare at most once"
            )
    # LAST, so a structurally impossible schedule is reported as such rather than as an
    # accounting shortfall: only a schedule that could exist is worth billing for.
    verify_escalation_budget(cfg, list(seed_of.values()), sum(s["voids"] for s in sessions.values()))
    return replays
