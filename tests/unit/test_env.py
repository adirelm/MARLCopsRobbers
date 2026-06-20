"""Unit tests for src/marl/env/cops_robbers_env.py + render_state.py (T2.3).

Covers the CopsRobbersEnv runtime: reset() returns per-agent LOCAL observations
plus an action_mask + ISO started_at in info; step() returns the 4-tuple
(obs_dict, reward_dict, terminated, info) with reward keys matching the agents;
a full 2x2 episode runs to a capture OR a timeout (winner in {cop, thief});
state() returns a GlobalState (the sanctioned train-only accessor); and
render_state() returns a plain serializable god-view dict carrying NO
GlobalState instance. The newer P2-fix tests live in test_env_p2_fixes.py.
"""

from __future__ import annotations

import datetime as dt

from src.marl.env.actions import Action
from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.marl.env.render_state import render_state
from src.marl.env.types import GlobalState


def _agent_keys(num_cops: int) -> set[str]:
    return {f"cop_{i}" for i in range(num_cops)} | {"thief"}


def test_reset_returns_local_obs_and_action_mask(cfg):
    env = CopsRobbersEnv(cfg, h=5, w=5, num_cops=1)
    obs_dict, info = env.reset(seed=7)
    assert set(obs_dict.keys()) == _agent_keys(1)
    for obs in obs_dict.values():
        assert set(obs.keys()) == {"image", "scalars"}
    assert set(info["action_mask"].keys()) == _agent_keys(1)
    # started_at is a parseable ISO-8601 timestamp.
    dt.datetime.fromisoformat(info["started_at"])


def test_reset_obs_shapes_match_config(cfg):
    env = CopsRobbersEnv(cfg, h=5, w=5, num_cops=1)
    obs_dict, _ = env.reset(seed=1)
    wv = 2 * cfg["env"]["view_radius_max"] + 1
    channels = cfg["env"]["obs_channels"]
    for obs in obs_dict.values():
        assert obs["image"].shape == (channels, wv, wv)
        assert obs["scalars"].shape == (cfg["env"]["obs_scalars"],)


def test_reset_is_deterministic_given_seed(cfg):
    env = CopsRobbersEnv(cfg, h=5, w=5, num_cops=1)
    env.reset(seed=42)
    s_a = env.state()
    env.reset(seed=42)
    s_b = env.state()
    assert s_a == s_b


def test_step_returns_four_tuple(cfg):
    env = CopsRobbersEnv(cfg, h=5, w=5, num_cops=1)
    env.reset(seed=7)
    out = env.step({"cop_0": Action.DOWN, "thief": Action.UP})
    assert isinstance(out, tuple)
    assert len(out) == 4
    obs_dict, _reward_dict, terminated, info = out
    assert set(obs_dict.keys()) == _agent_keys(1)
    assert isinstance(terminated, bool)
    assert "action_mask" in info
    assert "winner" in info
    assert "capture" in info


def test_step_reward_keys_match_agents(cfg):
    env = CopsRobbersEnv(cfg, h=5, w=5, num_cops=1)
    env.reset(seed=7)
    _, reward_dict, _, _ = env.step({"cop_0": Action.DOWN, "thief": Action.UP})
    assert set(reward_dict.keys()) == _agent_keys(1)
    for value in reward_dict.values():
        assert isinstance(value, float)


def test_full_2x2_episode_terminates(cfg):
    env = CopsRobbersEnv(cfg, h=2, w=2, num_cops=1)
    env.reset(seed=3)
    max_moves = cfg["game"]["max_moves"]
    terminated = False
    info: dict = {}
    for _ in range(max_moves):
        joint = {"cop_0": Action.DOWN, "thief": Action.UP}
        _, _, terminated, info = env.step(joint)
        if terminated:
            break
    assert terminated is True
    assert info["winner"] in ("cop", "thief")
    assert info["scores"] is not None
    assert set(info["scores"].keys()) == {"cop", "thief"}


def test_state_returns_globalstate(cfg):
    env = CopsRobbersEnv(cfg, h=5, w=5, num_cops=1)
    env.reset(seed=7)
    assert isinstance(env.state(), GlobalState)


def test_two_cop_env_keys(cfg):
    env = CopsRobbersEnv(cfg, h=4, w=4, num_cops=2)
    obs_dict, info = env.reset(seed=9)
    assert set(obs_dict.keys()) == _agent_keys(2)
    assert set(info["action_mask"].keys()) == _agent_keys(2)


def test_render_state_has_no_globalstate(cfg):
    env = CopsRobbersEnv(cfg, h=5, w=5, num_cops=1)
    env.reset(seed=7)
    rendered = render_state(env.state(), cfg)
    assert not isinstance(rendered, GlobalState)
    # god-view DRAW payload may carry dims+positions, but never a GlobalState.
    for value in rendered.values():
        assert not isinstance(value, GlobalState)
