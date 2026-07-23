"""§9.3 wire-match replay — re-run the logged match deterministically, render evidence.

Works from exactly (JSONL log + §9.4 records + the P7 ``wire_match.seeds`` k/k+3 schedule).
ANY divergence of the re-run :class:`CopsRobbersEnv` from the log/records raises
:class:`ReplayMismatchError` loudly — a silently diverging replay is worthless as §9.3
evidence. A void re-hello supersedes the session's earlier run; a P8 retry keeps the last
valid reply; an ESCALATED pair (P7: spare seed after 3 consecutive voids) is resolved by
spawn-matching s_k first, then the spares in order — mirror games must resolve to ONE seed.
Rendering calls INTO the god-view GUI path headlessly
(``src.gui.render.render_frame``; src/gui itself gains no imports).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.gui.spectator import SpectatorFrame
from src.marl.env.actions import Action
from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.marl.env.scorer import Scorer
from src.mcp._replay_log import ReplayMismatchError, gid_of, ordered_actions, parse_wire_log

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
    """Return the env + seed whose spawns match BOTH logged hellos: s_k, else a spare in order.

    A legitimately ESCALATED pair (P7: 3 consecutive voids) replayed under the next unused
    spare seed, so s_k is tried first, then every spare; the accepted seed goes in the summary.
    """
    seeds, grid = [int(s) for s in cfg["wire_match"]["seeds"]], int(cfg["game"]["grid_size"])
    pairs = int(cfg["game"]["num_games"]) // 2
    for seed in (seeds[(gid - 1) % pairs], *seeds[pairs:]):
        env = CopsRobbersEnv(cfg, h=grid, w=grid, num_cops=1)
        env.reset(seed=seed)
        state = env.state()
        spawn_of = {"cop": tuple(state.cop_pos[0]), "thief": tuple(state.thief_pos)}
        if all(sess["spawns"].get(role) == spawn_of[role] for role in _ROLES):
            return env, seed
    raise ReplayMismatchError(
        f"{sid}: logged spawns {sess['spawns']} match neither s_k nor any spare seed "
        f"in {seeds} — wrong seed schedule or tampered log"
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
    for k in range(1, pairs + 1):  # §9.1 mirror consistency: k and k+pairs must share ONE seed
        if {k, k + pairs} <= set(seed_of) and seed_of[k] != seed_of[k + pairs]:
            raise ReplayMismatchError(
                f"mirror pair {k}/{k + pairs} resolved to seeds {seed_of[k]} != {seed_of[k + pairs]}"
            )
    return replays


def mid_frame_index(frames: list, radius: int) -> int:
    """Pick the most informative mid frame: first barrier, else first mutual visibility, else middle."""
    inner = range(1, max(len(frames) - 1, 1))
    for i in inner:
        if frames[i].barriers:
            return i
    for i in inner:
        (cr, cc), (tr, tc) = frames[i].cop_positions[0], frames[i].thief_position
        if abs(cr - tr) + abs(cc - tc) <= radius:
            return i
    return len(frames) // 2


def save_screens(cfg: dict, replays: list[dict], out_dir: str | Path | None = None) -> list[Path]:
    """Render t00/mid/final PNGs per sub-game via the EXISTING GUI path (headless pygame)."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame  # noqa: PLC0415 - lazy: pygame is the optional gui extra

    from src.gui.render import render_frame  # noqa: PLC0415 - lazy with pygame

    out = Path(cfg["gui"]["bonus_screenshot_dir"] if out_dir is None else out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pygame.init()
    font, saved = pygame.font.SysFont(None, 24), []
    for game in replays:
        frames = game["frames"]
        mid = mid_frame_index(frames, int(cfg["mcp"]["observation"]["view_radius"]))
        picks = (("t00", 0), ("mid", mid), ("final", len(frames) - 1))
        for tag, idx in picks:
            surface = pygame.Surface((720, 560))
            render_frame(surface, font, frames[idx])
            path = out / f"bonus_sg{game['gid']}_{tag}.png"
            pygame.image.save(surface, str(path))
            saved.append(path)
    pygame.quit()
    return saved
