"""Cross-module invariants for src.marl.nets.dims (P4a).

dims must equal the ACTUAL encode_state / local-obs that the P4b CTDE learner
feeds the QMIX hypernet and the RecurrentQNet — otherwise the Linears size-
mismatch only at train time. These tests tie the config-derived dims to the
real producers (obs_encoder.encode_state, env.build_observation).
"""

from __future__ import annotations

import pytest

from src.marl.data.obs_encoder import encode_state
from src.marl.env.observation import VisibilityMemory, build_observation
from src.marl.nets.dims import action_dim, obs_dim, state_dim


def test_action_dim_unknown_role_raises(cfg: dict) -> None:
    """action_dim rejects any role other than cop/thief (covers the guard)."""
    with pytest.raises(ValueError, match="role"):
        action_dim(cfg, "robber")


def test_state_dim_matches_encode_state(cfg: dict, make_state) -> None:
    """dims.state_dim equals the real encode_state width (QMIX hypernet input)."""
    state = make_state(cop_pos=(0, 0), thief_pos=(2, 2), h=5, w=5)
    assert encode_state(state, cfg).shape[0] == state_dim(cfg)


def test_obs_dim_matches_local_obs(cfg: dict, make_state) -> None:
    """dims.obs_dim equals the flattened real local obs (RecurrentQNet input)."""
    state = make_state(cop_pos=(0, 0), thief_pos=(2, 2), h=5, w=5)
    obs = build_observation(state, "cop_0", {"cop_0": VisibilityMemory()}, cfg)
    assert obs["image"].size + obs["scalars"].size == obs_dim(cfg)
