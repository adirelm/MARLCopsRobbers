"""Tests for the self-play schedule, opponent pool, and heuristic policy (T4.6).

Covers the best-response role alternation, the heuristic-seeded FIFO opponent
pool, and the privileged Manhattan-heuristic acting policy. torch + RNG seeded.
"""

from __future__ import annotations

from random import Random

import torch

from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.marl.nets.agent_net import RecurrentQNet
from src.services.heuristic_policy import HeuristicPolicy
from src.services.selfplay import OpponentPool, training_role

SEED = 7


def test_training_role_alternates_in_window_blocks():
    """window_k=1 alternates every round; window_k=2 trains each role two rounds."""
    assert [training_role(i, 1) for i in range(4)] == ["cop", "thief", "cop", "thief"]
    assert [training_role(i, 2) for i in range(4)] == ["cop", "cop", "thief", "thief"]


def test_opponent_pool_seeded_with_heuristic(cfg):
    """A fresh pool holds exactly the heuristic seed and samples an acting policy."""
    pool = OpponentPool("thief", cfg, 1, seed=SEED)
    assert len(pool) == 1
    opponent = pool.sample()
    assert hasattr(opponent, "act") and hasattr(opponent, "reset")


def test_opponent_pool_fifo_caps_net_snapshots(cfg):
    """Net snapshots are FIFO-capped at pool_size; the heuristic seed is kept."""
    torch.manual_seed(SEED)
    pool = OpponentPool("cop", cfg, 1, seed=SEED)
    for _ in range(int(cfg["selfplay"]["pool_size"]) + 3):
        pool.add(RecurrentQNet(cfg, "cop", 2))
    assert len(pool) == int(cfg["selfplay"]["pool_size"]) + 1  # heuristic + pool_size nets


def test_heuristic_policy_returns_one_legal_action_per_cop(cfg):
    """The privileged heuristic returns a legal action per cop from the state."""
    env = CopsRobbersEnv(cfg, h=2, w=2, num_cops=1)
    _obs, info = env.reset(seed=1)
    actions = HeuristicPolicy("cop", cfg, 1).act([], [], epsilon=0.0, rng=Random(0), state=env.state())
    legal = [i for i, ok in enumerate(info["action_mask"]["cop_0"]) if ok]
    assert len(actions) == 1
    assert int(actions[0]) in legal
