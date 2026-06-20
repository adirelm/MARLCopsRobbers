"""Unit tests for src.sdk.sdk.MarlSDK (the single business-logic entry, T0.7/§4).

Pins the facade surface that does NOT require a trained net: ``build_env`` returns
a real P2 :class:`CopsRobbersEnv`, and ``write_subgame_json`` serializes a minimal
sub-game record. Every method must ROUTE through ``src`` (no business logic
re-implemented in the SDK). The training surface (train/finetune/build_policy/
export) is covered by test_trainer / test_finetune / test_sweep.
"""

from __future__ import annotations

import inspect
import json

from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.sdk.sdk import MarlSDK


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
    # The facade must not re-implement env/training logic — it delegates to src.*.
    assert "CopsRobbersEnv" in src
    assert "SelfPlayTrainer" in src
