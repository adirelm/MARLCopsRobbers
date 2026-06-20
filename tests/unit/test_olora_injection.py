"""Unit tests for the QmixLearner OLoRA net-injection seam (T4.5, P4c Stage 3).

Validates the backward-compatible ``net=`` seam re-opened on the P4b learner:
an OLoRA-wrapped net can be injected so the optimizer trains ONLY the
``{A, B, head, mixer}`` params (the frozen base ``W0``/``bias`` receive zero
gradient and never update), while ``net=None`` reproduces the P4b path that
builds and trains the full dense net. torch is seeded for determinism.
"""

from __future__ import annotations

import torch

from src.marl.learner.learner_base import QmixLearner
from src.marl.mixers.qmix_mixer import QmixMixer
from src.marl.nets.agent_net import RecurrentQNet
from src.marl.nets.olora_linear import OLoRALinear, wrap_encoder
from tests.unit._learner_fixtures import make_batch

SEED = 7


def _wrapped_net(cfg: dict) -> RecurrentQNet:
    """Build a seeded OLoRA-wrapped cop net (encoder Linears wrapped)."""
    torch.manual_seed(SEED)
    return wrap_encoder(RecurrentQNet(cfg, "cop", 2), cfg)


def _injected_learner(cfg: dict) -> QmixLearner:
    """Build a QMIX learner around an injected OLoRA-wrapped net."""
    torch.manual_seed(SEED)
    return QmixLearner(cfg, role="cop", n_agents=2, mixer=QmixMixer(cfg, 2), net=_wrapped_net(cfg))


def test_injected_net_is_used_as_online_net(cfg: dict) -> None:
    """An injected net becomes the learner's online net (no internal rebuild)."""
    net = _wrapped_net(cfg)
    learner = QmixLearner(cfg, role="cop", n_agents=2, mixer=QmixMixer(cfg, 2), net=net)
    assert learner.online_net is net


def test_only_adapters_head_mixer_have_gradient(cfg: dict) -> None:
    """After one update only {A,B,head,mixer} have grad; W0/bias are zero/None."""
    learner = _injected_learner(cfg)
    learner.update(make_batch(b=2, t=3, n=2, active=[True, True], seed=1))
    for module in learner.online_net.encoder:
        if isinstance(module, OLoRALinear):
            assert module.A.grad is not None and torch.any(module.A.grad != 0)
            assert module.B.grad is not None and torch.any(module.B.grad != 0)
            assert module.W0.grad is None  # frozen buffer: never accumulates grad
            assert module.bias.grad is None  # frozen base bias
    assert learner.online_net.head.weight.grad is not None
    gru_grads = [p.grad for p in learner.online_net.gru.parameters()]
    assert all(g is None for g in gru_grads)  # GRU frozen: transferred backbone never updates
    mixer_grads = [p.grad for p in learner.mixer.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in mixer_grads)


def test_frozen_base_weights_do_not_change_after_update(cfg: dict) -> None:
    """Optimizer excludes frozen base params: W0 is bit-identical after a step."""
    learner = _injected_learner(cfg)
    before = [m.W0.clone() for m in learner.online_net.encoder if isinstance(m, OLoRALinear)]
    learner.update(make_batch(b=2, t=3, n=2, active=[True, True], seed=2))
    after = [m.W0 for m in learner.online_net.encoder if isinstance(m, OLoRALinear)]
    assert before and all(torch.equal(b, a) for b, a in zip(before, after, strict=True))


def test_adapters_change_after_update(cfg: dict) -> None:
    """The trainable A/B adapters actually move under the optimizer step."""
    learner = _injected_learner(cfg)
    before = [m.A.clone() for m in learner.online_net.encoder if isinstance(m, OLoRALinear)]
    learner.update(make_batch(b=2, t=3, n=2, active=[True, True], seed=3))
    after = [m.A for m in learner.online_net.encoder if isinstance(m, OLoRALinear)]
    assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))


def test_net_none_builds_and_trains_full_dense_net(cfg: dict) -> None:
    """net=None reproduces P4b: a full dense net is built and every weight trains."""
    torch.manual_seed(SEED)
    learner = QmixLearner(cfg, role="cop", n_agents=2, mixer=QmixMixer(cfg, 2))
    assert not any(isinstance(m, OLoRALinear) for m in learner.online_net.encoder)
    first_linear = next(m for m in learner.online_net.encoder if isinstance(m, torch.nn.Linear))
    before = first_linear.weight.clone()
    learner.update(make_batch(b=2, t=3, n=2, active=[True, True], seed=4))
    assert not torch.equal(before, first_linear.weight)  # dense weight trains (P4b)
