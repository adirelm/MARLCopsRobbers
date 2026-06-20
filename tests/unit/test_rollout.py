"""Self-play rollout + per-role episode padding tests (T4.6).

Rolls a real 2x2 episode with two recurrent policies and asserts the per-role
records pad into the CentralizedReplayBuffer schema and ingest without error
(cop widened to N=2 with an active phantom slot; thief N=1). torch + RNG seeded.
"""

from __future__ import annotations

from random import Random

import numpy as np
import torch

from src.marl.data.schemas import SourceTag
from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.marl.nets.agent_net import RecurrentQNet
from src.marl.replay.episode_buffer import CentralizedReplayBuffer
from src.services.episode_pad import pad_episode
from src.services.policy import RecurrentPolicy
from src.services.rollout import collect_episode

SEED = 7


def _rollout(cfg):
    """Collect one 2x2 self-play episode (1 cop vs thief), seeded."""
    torch.manual_seed(SEED)
    env = CopsRobbersEnv(cfg, h=2, w=2, num_cops=1)
    cop = RecurrentPolicy(RecurrentQNet(cfg, "cop", 2), n_agents=1)
    thief = RecurrentPolicy(RecurrentQNet(cfg, "thief", 1), n_agents=1)
    return collect_episode(env, cop, thief, cfg, seed=1, epsilon=0.3, rng=Random(0))


def test_collect_episode_returns_both_role_tracks(cfg):
    """One rollout yields equal-length cop + thief step tracks and a capture flag."""
    ep = _rollout(cfg)
    assert len(ep["cop"]) == len(ep["thief"]) >= 1
    assert isinstance(ep["capture"], bool)
    assert ep["cop"][-1]["done"] is True  # last real step is terminal


def test_pad_cop_episode_widens_to_two_slots_with_active_mask(cfg):
    """A 1-cop episode pads to the buffer's N=2 width with active=[True, False]."""
    ep = _rollout(cfg)
    a_cop = cfg["env"]["actions"]["a_cop"]
    cop_ep = pad_episode(ep["cop"], n_slots=2, n_actions=a_cop, cfg=cfg)
    t = len(ep["cop"])
    assert cop_ep["obs"].shape[1] == 2
    assert cop_ep["active"].tolist() == [True, False]
    assert bool(cop_ep["filled"][:t].all()) and not bool(cop_ep["filled"][t:].any())
    assert bool(cop_ep["done"][t - 1])


def test_padded_episodes_ingest_into_role_buffers(cfg):
    """Padded cop (N=2) + thief (N=1) episodes add to their buffers without error."""
    ep = _rollout(cfg)
    env_cfg = cfg["env"]
    w_v = 2 * env_cfg["view_radius_max"] + 1
    state_dim = 3 * w_v**2 + 2
    # Both buffers store the env's uniform a_cop-wide mask; ThiefLearner slices it.
    mask_w = env_cfg["actions"]["a_cop"]
    cop_ep = pad_episode(ep["cop"], 2, mask_w, cfg)
    thief_ep = pad_episode(ep["thief"], 1, mask_w, cfg)
    cop_buf = CentralizedReplayBuffer(
        4,
        cfg["game"]["max_moves"],
        2,
        env_cfg["obs_channels"],
        w_v,
        env_cfg["obs_scalars"],
        state_dim,
        mask_w,
        seed=0,
    )
    thief_buf = CentralizedReplayBuffer(
        4,
        cfg["game"]["max_moves"],
        1,
        env_cfg["obs_channels"],
        w_v,
        env_cfg["obs_scalars"],
        state_dim,
        mask_w,
        seed=0,
    )
    cop_buf.add_episode(cop_ep, SourceTag.SELF_PLAY)
    thief_buf.add_episode(thief_ep, SourceTag.SELF_PLAY)
    assert len(cop_buf) == 1
    assert len(thief_buf) == 1
    batch = cop_buf.sample(2)
    assert batch["obs"].shape[0] == 2
    assert np.array_equal(batch["active"][0], np.array([True, False]))
