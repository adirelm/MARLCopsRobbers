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
from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.marl.env.scorer import Scorer
from src.mcp._replay_log import ReplayMismatchError, gid_of, ordered_actions, parse_wire_log
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


def _verify_tick(cfg: dict, sid: str, tick: int, sess: dict, state) -> None:
    """Check the replayed PRE-MOVE state against the tick's logged request payloads.

    The log carries per-tick ground truth (each role's ``your_pos`` + ``barriers_left``),
    so a divergence ANYWHERE in a 25-move game is caught at the tick it happens — not
    only when it survives to the terminal summary (the silent-divergence hole).
    """
    truth = sess["states"].get(tick, {})
    env_pos = {"cop": tuple(state.cop_pos[0]), "thief": tuple(state.thief_pos)}
    left = int(cfg["game"]["max_barriers"]) - int(state.barriers_used)
    for role in _ROLES:
        logged = truth.get(role)
        if logged is None:
            raise ReplayMismatchError(f"{sid} tick {tick}: no logged request payload for {role}")
        if logged["your_pos"] != env_pos[role] or logged["barriers_left"] != left:
            raise ReplayMismatchError(
                f"{sid} tick {tick} {role}: logged {logged} != replayed "
                f"{{'your_pos': {env_pos[role]}, 'barriers_left': {left}}}"
            )


def _seeded_env(cfg: dict, sid: str, sess: dict, gid: int) -> tuple[CopsRobbersEnv, int]:
    """Return the env + seed for ``sid``, spawn-verified against BOTH logged hellos.

    PRIMARY source: the seed the referee RECORDED in the session's JSONL ``result`` event
    — exact, and still cross-checked against the logged spawns (the authoritative tamper
    guard). FALLBACK, for logs predating seed events only: s_k then the spares in order
    by spawn match — ambiguous in principle, because distinct seeds can collide on the
    (cop, thief) spawn pair (~1/396 per candidate on the 5x5 board), so a decoy spare
    earlier in the order could silently win; the recorded seed removes that risk.
    """
    seeds, grid = [int(s) for s in cfg["wire_match"]["seeds"]], int(cfg["game"]["grid_size"])
    pairs = int(cfg["game"]["num_games"]) // 2
    allowed = (seeds[(gid - 1) % pairs], *seeds[pairs:])  # P7: s_k or a spare — nothing else is legal
    recorded = sess.get("seed")
    if recorded is not None and recorded not in allowed:
        raise ReplayMismatchError(
            f"{sid}: recorded result seed {recorded} is neither s_k nor a spare in {seeds}"
        )
    for seed in allowed if recorded is None else (recorded,):
        env = CopsRobbersEnv(cfg, h=grid, w=grid, num_cops=1)
        env.reset(seed=seed)
        state = env.state()
        spawn_of = {"cop": tuple(state.cop_pos[0]), "thief": tuple(state.thief_pos)}
        if all(sess["spawns"].get(role) == spawn_of[role] for role in _ROLES):
            return env, seed
    raise ReplayMismatchError(
        f"{sid}: logged spawns {sess['spawns']} match neither s_k nor any spare seed in {seeds}"
        if recorded is None
        else f"{sid}: logged spawns {sess['spawns']} do not match the recorded seed {recorded}"
    )


def replay_sub_game(cfg: dict, sid: str, sess: dict, record: dict, totals: dict) -> tuple[list, dict]:
    """Re-run one logged sub-game from its P7 (possibly escalated) seed; every tick verified."""
    gid = gid_of(sid)
    env, seed = _seeded_env(cfg, sid, sess, gid)
    frames, terminated, info = [_frame(cfg, env, gid, totals, None, None)], False, {}
    for tick, pair in enumerate(ordered_actions(sid, sess)):
        if terminated:
            raise ReplayMismatchError(f"{sid}: log continues past the terminal state at tick {tick}")
        if bad := [(role, pair[role]) for role in _ROLES if pair[role] not in _ALLOWED[role]]:
            raise ReplayMismatchError(f"{sid} tick {tick}: illegal logged action(s) {bad}")
        _verify_tick(cfg, sid, tick, sess, env.state())
        joint = {"cop_0": _ALLOWED["cop"][pair["cop"]], "thief": _MOVES[pair["thief"]]}
        _obs, _r, terminated, info = env.step(joint)
        frames.append(_frame(cfg, env, gid, totals, info.get("winner"), pair))
    got = {"moves": len(frames) - 1, "winner": info.get("winner"), "scores": info.get("scores")}
    want = {"moves": int(record["moves"]), "winner": record["winner"], "scores": dict(record["scores"])}
    if not terminated or got != want:
        raise ReplayMismatchError(f"{sid}: replay {got} (terminated={terminated}) != record {want}")
    return frames, {"gid": gid, "seed": seed, **got}


def replay_match(cfg: dict, log_path: str | Path, records_path: str | Path) -> list[dict]:
    """Replay every logged sub-game against the records; return per-game frames + summaries."""
    sessions = parse_wire_log(log_path)
    body = json.loads(Path(records_path).read_text(encoding="utf-8"))["sub_games"]
    records = {int(r["id"]): r for r in body}
    if sorted(gid_of(s) for s in sessions) != sorted(records):
        raise ReplayMismatchError(f"log sub-games {sorted(sessions)} != record ids {sorted(records)}")
    replays, totals = [], dict.fromkeys(_ROLES, 0)
    for sid in sorted(sessions, key=gid_of):
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
    return replays
