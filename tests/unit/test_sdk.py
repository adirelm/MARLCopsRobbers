"""Unit tests for src.sdk.sdk.MarlSDK (the single business-logic entry, T0.7/§4).

Pins the minimal P3 facade surface: ``build_env`` returns a real P2
:class:`CopsRobbersEnv`, ``collect_episode`` returns a buffer-schema padded
episode dict, and ``write_subgame_json`` writes a minimal sub-game record. Every
method must ROUTE through ``src`` (no business logic re-implemented in the SDK).
"""

from __future__ import annotations

import inspect
import json

from src.marl.data.heuristics import cop_expert, thief_expert
from src.marl.data.schemas import SourceTag
from src.marl.env.actions import Action
from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.marl.env.types import GlobalState
from src.marl.replay import _smoke_helpers as helpers
from src.sdk.sdk import MarlSDK


def _heuristic_policies():
    """Return (cop_policy, thief_policy) closures over the env experts (DRY)."""

    def cop_policy(state, _mask, cfg):
        return cop_expert(state, cfg, idx=0)

    def thief_policy(state, _mask, cfg):
        return thief_expert(state, cfg)

    return cop_policy, thief_policy


def test_build_env_returns_real_env(cfg):
    """build_env constructs a real P2 CopsRobbersEnv at the requested size."""
    sdk = MarlSDK(cfg)
    env = sdk.build_env(h=2, w=2, num_cops=1)
    assert isinstance(env, CopsRobbersEnv)
    obs, info = env.reset(seed=0)
    assert set(obs) == {"cop_0", "thief"}
    assert "action_mask" in info


def test_build_env_defaults_from_config(cfg):
    """build_env with no args falls back to the configured grid + cop count."""
    sdk = MarlSDK(cfg)
    env = sdk.build_env()
    state = (env.reset(seed=1), env.state())[1]
    assert state.h == cfg["game"]["grid_size"]
    assert len(state.cop_pos) == cfg["env"]["num_cops"]


def test_collect_episode_matches_buffer_schema(cfg):
    """collect_episode returns a padded episode dict with the buffer keys/shapes."""
    sdk = MarlSDK(cfg)
    env = sdk.build_env(h=2, w=2, num_cops=1)
    cop_policy, thief_policy = _heuristic_policies()
    episode = sdk.collect_episode(env, cop_policy, thief_policy, seed=7)
    t_max = cfg["game"]["max_moves"]
    a_cop = cfg["env"]["actions"]["a_cop"]
    c = cfg["env"]["obs_channels"]
    w_v = 2 * cfg["env"]["view_radius_max"] + 1
    n_scalars = cfg["env"]["obs_scalars"]
    assert episode["filled"].shape == (t_max,)
    assert episode["actions"].shape == (t_max, 1)
    assert episode["next_legal_mask"].shape == (t_max, 1, a_cop)
    # obs/scalars/global_state live on the T+1 axis (keep the terminal next-obs
    # frame so P4's recurrent target can unroll obs[1..T]) — matches the buffer.
    assert episode["obs"].shape == (t_max + 1, 1, c, w_v, w_v)
    assert episode["scalars"].shape == (t_max + 1, 1, n_scalars)
    assert episode["global_state"].shape[0] == t_max + 1
    assert bool(episode["filled"][0]) is True  # at least one real step


def test_collect_episode_feeds_real_buffer(cfg):
    """The padded episode is ingestible by the real CentralizedReplayBuffer."""
    sdk = MarlSDK(cfg)
    env = sdk.build_env(h=2, w=2, num_cops=1)
    cop_policy, thief_policy = _heuristic_policies()
    episode = sdk.collect_episode(env, cop_policy, thief_policy, seed=3)
    buffer = helpers.make_buffer(cfg, seed=0)
    buffer.add_episode(episode, SourceTag.LIVE_CTDE)
    assert len(buffer) == 1


def test_write_subgame_json_writes_minimal_record(cfg, tmp_path):
    """write_subgame_json serializes the minimal sub-game record to disk."""
    sdk = MarlSDK(cfg)
    result = {
        "game_id": "g1",
        "grid": [2, 2],
        "winner": "cop",
        "capture": True,
        "steps": 1,
        "scores": {"cop": 20, "thief": 5},
        "seed": 7,
    }
    path = tmp_path / "subgames" / "g1.json"
    sdk.write_subgame_json(result, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == result


def test_facade_routes_through_src_single_entry():
    """The SDK module imports its building blocks from src (single-entry, §4)."""
    src = inspect.getsource(MarlSDK)
    # The facade must not re-implement env/replay logic — it delegates to src.*.
    assert "CopsRobbersEnv" in src or "_helpers" in src
    # Sanity: the heuristic experts return real Actions used by the policies.
    cop_policy, _ = _heuristic_policies()
    state = GlobalState(((0, 0),), (1, 1), frozenset(), 0, 0, 2, 2, False)
    cfg_min = {"env": {"actions": {"a_cop": 5}}}  # cop_expert ignores cfg geometry
    assert isinstance(cop_policy(state, None, cfg_min), Action)
