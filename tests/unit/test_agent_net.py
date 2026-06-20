"""Unit tests for dims helpers + RecurrentQNet (T4.0b, P4a Stage 1).

Pure-tensor forward-shape, hidden-carry, weight-sharing, and target-clone
gates. No training loop. torch is seeded for determinism.
"""

from __future__ import annotations

import torch

from src.marl.nets.agent_net import RecurrentQNet
from src.marl.nets.dims import action_dim, hidden_dim, obs_dim, state_dim

SEED = 7


def test_dims_match_config(cfg: dict) -> None:
    """Config-derived dims equal the spec constants (131/64/5/4/77)."""
    assert obs_dim(cfg) == 131
    assert hidden_dim(cfg) == 64
    assert action_dim(cfg, "cop") == 5
    assert action_dim(cfg, "thief") == 4
    assert state_dim(cfg) == 77


def test_forward_shapes_cop(cfg: dict) -> None:
    """Cop forward returns q[B,n,5] and h'[B,n,H]."""
    torch.manual_seed(SEED)
    b, n = 3, 1
    net = RecurrentQNet(cfg, "cop", n_agents=n)
    obs = torch.randn(b, n, obs_dim(cfg))
    h = net.initial_hidden(b, n)
    q, h2 = net.forward(obs, h)
    assert q.shape == (b, n, action_dim(cfg, "cop"))
    assert h2.shape == (b, n, hidden_dim(cfg))


def test_forward_shapes_thief(cfg: dict) -> None:
    """Thief forward returns q[B,n,4] (action_dim differs from cop)."""
    torch.manual_seed(SEED)
    b, n = 2, 1
    net = RecurrentQNet(cfg, "thief", n_agents=n)
    obs = torch.randn(b, n, obs_dim(cfg))
    q, _ = net.forward(obs, net.initial_hidden(b, n))
    assert q.shape == (b, n, action_dim(cfg, "thief"))
    assert action_dim(cfg, "thief") != action_dim(cfg, "cop")


def test_initial_hidden_is_zeros(cfg: dict) -> None:
    """initial_hidden returns an all-zero [B,n,H] tensor."""
    net = RecurrentQNet(cfg, "cop", n_agents=2)
    h = net.initial_hidden(4, 2)
    assert h.shape == (4, 2, hidden_dim(cfg))
    assert torch.count_nonzero(h) == 0


def test_hidden_carries_on_nonzero_input(cfg: dict) -> None:
    """h' differs from the zero initial h on a nonzero observation."""
    torch.manual_seed(SEED)
    net = RecurrentQNet(cfg, "cop", n_agents=1)
    obs = torch.randn(2, 1, obs_dim(cfg))
    h0 = net.initial_hidden(2, 1)
    _, h1 = net.forward(obs, h0)
    assert not torch.allclose(h1, h0)


def test_weight_sharing_distinct_agent_ids(cfg: dict) -> None:
    """n=2 weight-sharing with distinct agent-id one-hots yields differing q."""
    torch.manual_seed(SEED)
    net = RecurrentQNet(cfg, "cop", n_agents=2)
    # Identical per-agent obs: any q difference comes purely from the agent-id
    # one-hot the net prepends, proving agents are distinguishable.
    base = torch.randn(3, 1, obs_dim(cfg))
    obs = base.repeat(1, 2, 1)
    q, _ = net.forward(obs, net.initial_hidden(3, 2))
    assert not torch.allclose(q[:, 0, :], q[:, 1, :])


def test_no_cross_agent_contamination(cfg: dict) -> None:
    """Perturbing agent 1's obs/hidden leaves agent 0's q,h bit-identical.

    Guards the (B*n) flatten->GRUCell->reshape round-trip: a transposed/wrong
    reshape would silently mix agent rows yet still pass a mere distinctness
    check, so we assert true per-agent isolation.
    """
    torch.manual_seed(SEED)
    net = RecurrentQNet(cfg, "cop", n_agents=2)
    obs = torch.randn(3, 2, obs_dim(cfg))
    h0 = torch.randn(3, 2, hidden_dim(cfg))
    q, h1 = net.forward(obs, h0)
    obs2 = obs.clone()
    obs2[:, 1, :] = torch.randn(3, obs_dim(cfg))  # change ONLY agent 1's obs
    h0b = h0.clone()
    h0b[:, 1, :] = torch.randn(3, hidden_dim(cfg))  # and ONLY agent 1's hidden
    q2, h2 = net.forward(obs2, h0b)
    assert torch.equal(q[:, 0, :], q2[:, 0, :])  # agent 0 untouched
    assert torch.equal(h1[:, 0, :], h2[:, 0, :])
    assert not torch.equal(q[:, 1, :], q2[:, 1, :])  # agent 1 did change


def test_clone_target_equal_and_detached(cfg: dict) -> None:
    """clone_target is a frozen, storage-independent copy that never tracks online."""
    torch.manual_seed(SEED)
    net = RecurrentQNet(cfg, "cop", n_agents=1)
    target = net.clone_target()
    assert target is not net
    for p_online, p_target in zip(net.parameters(), target.parameters(), strict=True):
        assert torch.equal(p_online, p_target)
        assert p_target.requires_grad is False
        assert p_online.data_ptr() != p_target.data_ptr()  # no shared storage
    # Behavioral isolation: mutating online must NOT change the frozen target,
    # and online must stay trainable (the silent Double-DQN tracking bug).
    before = target.head.weight.clone()
    with torch.no_grad():
        net.head.weight.add_(1.0)
    assert torch.equal(target.head.weight, before)
    assert not torch.equal(net.head.weight, target.head.weight)
    assert all(p.requires_grad for p in net.parameters())
