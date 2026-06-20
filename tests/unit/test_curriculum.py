"""Unit tests for src/marl/env/curriculum.py (T2.4).

Covers the graduated Sanity-Check ladder: ``current()`` returns the right
``(h, w, num_cops)`` per stage; ``maybe_promote`` advances on a capture-rate
>= ``env.curriculum.promotion_threshold`` and clamps at the final stage;
``make_env`` builds a CopsRobbersEnv for the active stage; and a full
``<= game.max_moves`` episode runs at every stage size with a winner in
``{cop, thief}`` and a placed-barrier count never exceeding the team budget.
All bounds come from config — nothing is hardcoded (CLAUDE.md §4).
"""

from __future__ import annotations

import copy

import pytest

from src.marl.env.actions import Action
from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.marl.env.curriculum import Curriculum


def _stages(cfg: dict) -> list[list[int]]:
    return cfg["env"]["curriculum"]["stages"]


def _num_cops_by_stage(cfg: dict) -> list[int]:
    return cfg["env"]["curriculum"]["num_cops_by_stage"]


def _with_stages(cfg: dict, stages: list[list[int]], cops: list[int]) -> dict:
    """Return a deep-copied cfg whose curriculum ladder is replaced (synthetic)."""
    synthetic = copy.deepcopy(cfg)
    synthetic["env"]["curriculum"]["stages"] = stages
    synthetic["env"]["curriculum"]["num_cops_by_stage"] = cops
    return synthetic


def test_current_returns_first_stage(cfg):
    cur = Curriculum(cfg)
    h, w, num_cops = cur.current()
    first_h, first_w = _stages(cfg)[0]
    assert (h, w, num_cops) == (first_h, first_w, _num_cops_by_stage(cfg)[0])


def test_current_matches_config_per_stage(cfg):
    cur = Curriculum(cfg)
    stages = _stages(cfg)
    cops = _num_cops_by_stage(cfg)
    threshold = cfg["env"]["curriculum"]["promotion_threshold"]
    for i, (h, w) in enumerate(stages):
        assert cur.current() == (h, w, cops[i])
        if i < len(stages) - 1:
            assert cur.maybe_promote(threshold) is True


def test_promotes_at_exactly_threshold(cfg):
    cur = Curriculum(cfg)
    threshold = cfg["env"]["curriculum"]["promotion_threshold"]
    before = cur.current()
    assert cur.maybe_promote(threshold) is True
    assert cur.current() != before


def test_does_not_promote_below_threshold(cfg):
    cur = Curriculum(cfg)
    threshold = cfg["env"]["curriculum"]["promotion_threshold"]
    before = cur.current()
    # Just below threshold (clamped at 0) must NOT advance the stage.
    assert cur.maybe_promote(max(0.0, threshold - 0.01)) is False
    assert cur.current() == before


def test_clamps_at_final_stage(cfg):
    cur = Curriculum(cfg)
    stages = _stages(cfg)
    cops = _num_cops_by_stage(cfg)
    # Promote past the end; the final promotion attempt returns False (clamped).
    for _ in range(len(stages) - 1):
        assert cur.maybe_promote(1.0) is True
    assert cur.maybe_promote(1.0) is False
    last_h, last_w = stages[-1]
    assert cur.current() == (last_h, last_w, cops[-1])


def test_make_env_for_current_stage(cfg):
    cur = Curriculum(cfg)
    env = cur.make_env(cfg)
    assert isinstance(env, CopsRobbersEnv)
    h, w, num_cops = cur.current()
    obs_dict, _ = env.reset(seed=1)
    expected = {f"cop_{i}" for i in range(num_cops)} | {"thief"}
    assert set(obs_dict.keys()) == expected
    state = env.state()
    assert (state.h, state.w) == (h, w)
    assert len(state.cop_pos) == num_cops


def _run_episode(env: CopsRobbersEnv, num_cops: int, max_moves: int) -> dict:
    env.reset(seed=5)
    info: dict = {}
    for _ in range(max_moves):
        joint = {f"cop_{i}": Action.DOWN for i in range(num_cops)}
        joint["thief"] = Action.UP
        _, _, terminated, info = env.step(joint)
        if terminated:
            break
    return info


def test_full_episode_runs_at_each_stage(cfg):
    cur = Curriculum(cfg)
    stages = _stages(cfg)
    cops = _num_cops_by_stage(cfg)
    max_moves = cfg["game"]["max_moves"]
    max_barriers = cfg["game"]["max_barriers"]
    threshold = cfg["env"]["curriculum"]["promotion_threshold"]
    for i in range(len(stages)):
        env = cur.make_env(cfg)
        info = _run_episode(env, cops[i], max_moves)
        assert info["winner"] in ("cop", "thief")
        assert env.state().barriers_used <= max_barriers
        assert len(env.state().barriers) <= max_barriers
        if i < len(stages) - 1:
            cur.maybe_promote(threshold)


def test_promotion_window_exposes_config_value(cfg):
    cur = Curriculum(cfg)
    assert cur.promotion_window == cfg["env"]["curriculum"]["promotion_window"]


def test_init_rejects_mismatched_parallel_lists(cfg):
    # Three stages but only two cop counts is an invalid (non-parallel) ladder.
    bad = _with_stages(cfg, [[2, 2], [3, 3], [4, 4]], [1, 1])
    with pytest.raises(ValueError, match="parallel"):
        Curriculum(bad)


def test_current_on_synthetic_non_square_stages(cfg):
    # A non-square ladder (h != w) is supported by the egocentric obs pad.
    synthetic = _with_stages(cfg, [[3, 5], [4, 6]], [1, 2])
    cur = Curriculum(synthetic)
    assert cur.current() == (3, 5, 1)
    assert cur.maybe_promote(1.0) is True
    assert cur.current() == (4, 6, 2)


def test_make_env_defaults_to_bound_cfg(cfg):
    # make_env with no arg uses the single config bound at construction.
    cur = Curriculum(cfg)
    env = cur.make_env()
    assert isinstance(env, CopsRobbersEnv)
    h, w, num_cops = cur.current()
    env.reset(seed=2)
    state = env.state()
    assert (state.h, state.w) == (h, w)
    assert len(state.cop_pos) == num_cops
