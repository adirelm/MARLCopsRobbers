"""Unit tests for wrap_encoder attach scope (T4.3, P4c Stage 1).

Asserts that wrap_encoder converts ONLY the encoder's nn.Linear children to
OLoRALinear, leaving the GRU cell and the Q-head as plain modules, and that
the post-wrap trainable set is restricted to the adapters {A, B} (+ head).
torch is seeded for determinism.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.marl.nets.agent_net import RecurrentQNet
from src.marl.nets.dims import obs_dim
from src.marl.nets.olora_linear import OLoRALinear, wrap_encoder

SEED = 7


def test_wrap_encoder_replaces_only_encoder_linears(cfg: dict) -> None:
    """Every encoder nn.Linear becomes OLoRALinear; none are left plain."""
    torch.manual_seed(SEED)
    net = RecurrentQNet(cfg, "cop", n_agents=2)
    n_linears = sum(isinstance(m, nn.Linear) for m in net.encoder)
    assert n_linears > 0
    wrap_encoder(net, cfg)
    assert sum(isinstance(m, OLoRALinear) for m in net.encoder) == n_linears
    assert not any(isinstance(m, nn.Linear) for m in net.encoder)


def test_wrap_encoder_leaves_gru_and_head_plain(cfg: dict) -> None:
    """gru stays nn.GRUCell and head stays a plain nn.Linear after wrap."""
    torch.manual_seed(SEED)
    net = RecurrentQNet(cfg, "cop", n_agents=2)
    wrap_encoder(net, cfg)
    assert isinstance(net.gru, nn.GRUCell)
    assert isinstance(net.head, nn.Linear)
    assert not isinstance(net.head, OLoRALinear)


def test_wrap_encoder_preserves_forward(cfg: dict) -> None:
    """Wrapping is function-preserving end-to-end at init (<1e-4 over forward)."""
    torch.manual_seed(SEED)
    net = RecurrentQNet(cfg, "cop", n_agents=2)
    obs = torch.randn(3, 2, obs_dim(cfg))
    h0 = net.initial_hidden(3, 2)
    q_before, _ = net.forward(obs, h0)
    wrap_encoder(net, cfg)
    q_after, _ = net.forward(obs, h0)
    assert torch.linalg.norm(q_after - q_before) < 1e-4


def test_wrap_encoder_trainable_set_is_adapters_plus_head(cfg: dict) -> None:
    """After wrap only {A,B} (encoder) and the plain head require grad."""
    torch.manual_seed(SEED)
    net = RecurrentQNet(cfg, "cop", n_agents=2)
    wrap_encoder(net, cfg)
    for name, p in net.named_parameters():
        if name.startswith("encoder"):
            short = name.rsplit(".", 1)[-1]
            assert p.requires_grad == (short in {"A", "B"}), name
    head_grads = [p.requires_grad for n, p in net.named_parameters() if n.startswith("head")]
    assert all(head_grads)
    gru_grads = [p.requires_grad for n, p in net.named_parameters() if n.startswith("gru")]
    assert not any(gru_grads)  # gru FROZEN: transferred backbone; only encoder {A,B} + head train


def test_wrap_encoder_rejects_unsupported_target_layers(cfg: dict) -> None:
    """target_layers other than ['encoder'] is rejected (no silent no-op knob)."""
    torch.manual_seed(SEED)
    net = RecurrentQNet(cfg, "cop", n_agents=2)
    bad = {**cfg, "olora": {**cfg["olora"], "target_layers": ["encoder", "head"]}}
    with pytest.raises(ValueError, match="target_layers"):
        wrap_encoder(net, bad)


def test_wrap_encoder_fewer_trainable_per_linear(cfg: dict) -> None:
    """Total trainable params drop after wrap (frozen encoder W0/bias removed)."""
    torch.manual_seed(SEED)
    net = RecurrentQNet(cfg, "cop", n_agents=2)
    before = sum(p.numel() for p in net.parameters() if p.requires_grad)
    wrap_encoder(net, cfg)
    after = sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert after < before
