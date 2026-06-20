"""Tests for the recurrent acting policy (T4.6 — GRU hidden carry + ε-greedy).

Drives a real 2x2 env's local observations through a CopNet-shaped RecurrentQNet
so the encode->forward->mask->select path is exercised end to end. torch + the
exploration RNG are seeded for determinism.
"""

from __future__ import annotations

from random import Random

import torch

from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.marl.nets.agent_net import RecurrentQNet
from src.services.policy import RecurrentPolicy

SEED = 7


def _env_and_policy(cfg):
    """Build a reset 2x2 single-cop env + a fresh CopNet-backed policy."""
    torch.manual_seed(SEED)
    env = CopsRobbersEnv(cfg, h=2, w=2, num_cops=1)
    obs, info = env.reset(seed=1)
    policy = RecurrentPolicy(RecurrentQNet(cfg, "cop", n_agents=2), n_agents=1)
    return env, obs, info, policy


def test_act_returns_one_legal_action_per_agent(cfg):
    """Greedy act returns exactly one action per agent, always within the mask."""
    _env, obs, info, policy = _env_and_policy(cfg)
    mask = info["action_mask"]["cop_0"]
    actions = policy.act([obs["cop_0"]], [mask], epsilon=0.0, rng=Random(0))
    legal = [i for i, ok in enumerate(mask) if ok]
    assert len(actions) == 1
    assert int(actions[0]) in legal


def test_epsilon_one_still_legal(cfg):
    """Pure exploration (ε=1) still only ever returns LEGAL actions."""
    _env, obs, info, policy = _env_and_policy(cfg)
    mask = info["action_mask"]["cop_0"]
    legal = {i for i, ok in enumerate(mask) if ok}
    for _ in range(20):
        action = policy.act([obs["cop_0"]], [mask], epsilon=1.0, rng=Random(0))[0]
        assert int(action) in legal


def test_hidden_state_advances_then_resets(cfg):
    """act() advances the GRU hidden state; reset() restores it to zero (eq 8)."""
    _env, obs, info, policy = _env_and_policy(cfg)
    mask = info["action_mask"]["cop_0"]
    assert torch.count_nonzero(policy._hidden) == 0
    policy.act([obs["cop_0"]], [mask], epsilon=0.0, rng=Random(0))
    assert torch.count_nonzero(policy._hidden) > 0
    policy.reset()
    assert torch.count_nonzero(policy._hidden) == 0


def test_greedy_is_deterministic(cfg):
    """ε=0 acting is deterministic for a fixed net + obs (no RNG dependence)."""
    _env, obs, info, policy_a = _env_and_policy(cfg)
    _env2, obs2, info2, policy_b = _env_and_policy(cfg)
    a = policy_a.act([obs["cop_0"]], [info["action_mask"]["cop_0"]], 0.0, Random(1))
    b = policy_b.act([obs2["cop_0"]], [info2["action_mask"]["cop_0"]], 0.0, Random(99))
    assert int(a[0]) == int(b[0])
