"""Shared fixtures for the episode-buffer unit tests (T3.1 / T4.2).

Builds a small buffer-only ``CentralizedReplayBuffer`` and UNPADDED episode
dicts decoupled from the real grid (``(C,W,W)=(5,5,5)``). Lives in a helper
module so the core and active-mask test files each stay well under the 150-LOC
gate while sharing one episode-builder (DRY).
"""

from __future__ import annotations

import numpy as np

from src.marl.replay.episode_buffer import CentralizedReplayBuffer

_CAP = 4
_T_MAX = 6
_N = 2
_C = 5
_W = 5
_SCALARS = 6
_STATE_DIM = 77
_N_ACTIONS = 5


def make_buffer(capacity: int = _CAP, n_agents: int = _N, seed: int = 0) -> CentralizedReplayBuffer:
    """Build a small buffer-only ring (dims decoupled from the real grid)."""
    return CentralizedReplayBuffer(
        capacity=capacity,
        t_max=_T_MAX,
        n_agents=n_agents,
        obs_channels=_C,
        w_v=_W,
        obs_scalars=_SCALARS,
        state_dim=_STATE_DIM,
        n_actions=_N_ACTIONS,
        seed=seed,
    )


def make_episode(length: int, n_agents: int = _N, fill: float = 1.0) -> dict:
    """Build an UNPADDED episode dict of real-step length ``length``.

    Carries an episode-constant ``active`` mask of shape ``(n_agents,)`` (all
    ``True`` by default == a fully-occupied N-wide episode).
    """
    t = length
    rng = np.random.default_rng(int(fill * 100) + length)
    return {
        "obs": np.full((t + 1, n_agents, _C, _W, _W), fill, dtype=np.float32),
        "scalars": np.full((t + 1, n_agents, _SCALARS), fill, dtype=np.float32),
        "global_state": np.full((t + 1, _STATE_DIM), fill, dtype=np.float32),
        "actions": rng.integers(0, _N_ACTIONS, size=(t, n_agents)).astype(np.int64),
        "reward": np.full((t, n_agents), fill, dtype=np.float32),
        "done": np.array([i == t - 1 for i in range(t)], dtype=bool),
        "filled": np.ones(t, dtype=bool),
        "next_legal_mask": np.ones((t, n_agents, _N_ACTIONS), dtype=bool),
        "hidden_seed": np.int64(123),
        "active": np.ones(n_agents, dtype=bool),
    }
