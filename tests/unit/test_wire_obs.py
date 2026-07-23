"""Bit-equivalence proof for the §9 wire-obs reconstruction (the whole ballgame).

Drives a REAL CopsRobbersEnv sub-game (both roles scripted), at every tick builds
the wire ``request_move`` payload the referee WOULD send (radius-2 masking per the
partner brief §2 / P5), rebuilds the Observation via ``src.mcp.wire_obs``, and
asserts it equals the env-emitted Observation EXACTLY (``np.array_equal`` per
array + dtype) for BOTH roles across full episodes and 3 seeds. Also proves the
legality-mask parity and the constructor guards.
"""

from __future__ import annotations

from copy import deepcopy
from random import Random

import numpy as np
import pytest

from src.marl.env.actions import DELTAS, Action
from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.marl.env.grid import manhattan
from src.mcp import wire_obs

_MOVES = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)
_SEEDS = (7, 17, 37)


def _payload(state, role, tick, cfg):
    """Build the request_move payload the referee would send for `role` (P5 masking)."""
    you = state.cop_pos[0] if role == "cop" else state.thief_pos
    opp = state.thief_pos if role == "cop" else state.cop_pos[0]
    radius = int(cfg["mcp"]["observation"]["view_radius"])
    return {
        "session_id": "sg-0",
        "tick": tick,
        "your_pos": list(you),
        "opponent_pos": list(opp) if manhattan(you, opp) <= radius else None,
        "barriers": sorted(list(b) for b in state.barriers if manhattan(you, b) <= radius),
        "barriers_left": cfg["game"]["max_barriers"] - state.barriers_used,
    }


def _steer(pos, target, mask, toward):
    """Deterministic legal move minimizing (toward) / maximizing (flee) the distance."""
    scored = sorted(
        (manhattan((pos[0] + DELTAS[mv][0], pos[1] + DELTAS[mv][1]), target), i, mv)
        for i, mv in enumerate(_MOVES)
        if mask[i]
    )
    return scored[0][2] if toward else scored[-1][2]


def _assert_tick(sessions, env_out, state, tick, cfg):
    """Assert wire-rebuilt obs + mask == env-emitted obs + mask for BOTH roles."""
    obs, info = env_out
    flags = {"seen": False, "unseen": False, "barrier": False}
    for role, key in (("cop", "cop_0"), ("thief", "thief")):
        pay = _payload(state, role, tick, cfg)
        rebuilt = wire_obs.build_observation(sessions[role], pay, cfg)
        want = obs[key]
        assert rebuilt["image"].dtype == want["image"].dtype == np.float32
        assert np.array_equal(rebuilt["image"], want["image"]), (role, tick)
        assert rebuilt["scalars"].dtype == want["scalars"].dtype == np.float32
        assert np.array_equal(rebuilt["scalars"], want["scalars"]), (role, tick)
        mask = wire_obs.build_mask(sessions[role], pay, cfg)
        assert np.array_equal(mask, info["action_mask"][key]), (role, tick)
        flags["seen" if pay["opponent_pos"] is not None else "unseen"] = True
        flags["barrier"] |= bool(pay["barriers"])
    return flags


def test_wire_obs_bit_equivalent_full_episodes(cfg):
    """Every tick of full episodes across 3 seeds rebuilds bit-identical obs."""
    seen_any, unseen_any, barrier_any, ticks_checked = False, False, False, 0
    for seed in _SEEDS:
        env = CopsRobbersEnv(cfg, h=5, w=5, num_cops=1)
        obs, info = env.reset(seed=seed)
        grid, max_moves = (5, 5), cfg["game"]["max_moves"]
        sessions = {r: wire_obs.new_session(r, grid, max_moves, cfg) for r in ("cop", "thief")}
        rng, tick, terminated = Random(seed), 0, False
        while not terminated:
            state = env.state()
            flags = _assert_tick(sessions, (obs, info), state, tick, cfg)
            seen_any |= flags["seen"]
            unseen_any |= flags["unseen"]
            barrier_any |= flags["barrier"]
            ticks_checked += 1
            cop, thief, masks = state.cop_pos[0], state.thief_pos, info["action_mask"]
            if masks["cop_0"][int(Action.PLACE_BARRIER)] and rng.random() < 0.4:
                cop_a = Action.PLACE_BARRIER
            else:
                cop_a = _steer(cop, thief, masks["cop_0"], toward=True)
            thief_a = _steer(thief, cop, masks["thief"], toward=tick % 2 == 0)
            obs, _r, terminated, info = env.step({"cop_0": cop_a, "thief": thief_a})
            tick += 1
    assert ticks_checked >= len(_SEEDS)  # at least one full tick asserted per episode
    assert seen_any and unseen_any and barrier_any  # all wire branches exercised


def test_new_session_rejects_unknown_role(cfg):
    with pytest.raises(ValueError, match="role"):
        wire_obs.new_session("referee", (5, 5), cfg["game"]["max_moves"], cfg)


def test_new_session_rejects_max_moves_mismatch(cfg):
    with pytest.raises(ValueError, match="max_moves"):
        wire_obs.new_session("cop", (5, 5), cfg["game"]["max_moves"] + 1, cfg)


def test_new_session_rejects_narrow_wire_masking(cfg):
    narrow = deepcopy(cfg)
    narrow["mcp"]["observation"]["view_radius"] = 1  # < env view radius 2 -> info lost
    with pytest.raises(ValueError, match="radius"):
        wire_obs.new_session("cop", (5, 5), cfg["game"]["max_moves"], narrow)


def test_build_observation_rejects_unreachable_out_of_view_stand_in(cfg):
    wide = deepcopy(cfg)
    wide["env"]["view_radius_by_grid"][5] = 4  # whole-board view: no out-of-view cell
    wide["env"]["view_radius_max"] = 4  # keep the footprint consistent — a radius above
    # view_radius_max now fails loudly instead of silently truncating (codex W2 M4)
    wide["mcp"]["observation"]["view_radius"] = 4
    session = wire_obs.new_session("cop", (5, 5), cfg["game"]["max_moves"], wide)
    payload = {"tick": 0, "your_pos": [2, 2], "opponent_pos": None, "barriers": [], "barriers_left": 5}
    with pytest.raises(ValueError, match="stand-in"):
        wire_obs.build_observation(session, payload, wide)


def test_synth_state_rejects_bad_barriers_left(cfg):
    session = wire_obs.new_session("thief", (5, 5), cfg["game"]["max_moves"], cfg)
    payload = {"tick": 0, "your_pos": [0, 0], "opponent_pos": None, "barriers": [], "barriers_left": 99}
    with pytest.raises(ValueError, match="barriers_left"):
        wire_obs.build_mask(session, payload, cfg)
