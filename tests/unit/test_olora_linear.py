"""Unit tests for the OLoRA QR init + OLoRALinear wrapper (T4.3, P4c Stage 1).

Numerically validates the paper-faithful OLoRA math ([7] eqs 3-5): thin-QR of
the PRETRAINED weight, function-preserving re-base at init, orthonormal B
columns, and the rank guard. torch is seeded for determinism.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.marl.nets.olora_init import olora_init
from src.marl.nets.olora_linear import OLoRALinear

SEED = 7
RANK = 4
SCALE = 8.0
FN_TOL = 1e-5
ORTHO_TOL = 1e-5


def test_olora_init_shapes_and_factorization() -> None:
    """olora_init returns (W0, B, A) with B=(m,r), A=(r,n), W0=(m,n)."""
    torch.manual_seed(SEED)
    m, n = 64, 40
    weight = torch.randn(m, n)
    w0, b, a = olora_init(weight, RANK, SCALE)
    assert w0.shape == (m, n)
    assert b.shape == (m, RANK)
    assert a.shape == (RANK, n)


def test_olora_init_function_preserving_vs_saved_weight() -> None:
    """W0 + scale*(B@A) reproduces the SAVED pretrained weight (<1e-5)."""
    torch.manual_seed(SEED)
    weight = torch.randn(64, 50)
    w0, b, a = olora_init(weight, RANK, SCALE)
    w_eff = w0 + SCALE * (b @ a)
    assert torch.linalg.norm(w_eff - weight) < FN_TOL


def test_olora_init_orthonormal_columns() -> None:
    """B has orthonormal columns: BᵀB ≈ I_r."""
    torch.manual_seed(SEED)
    weight = torch.randn(64, 48)
    _, b, _ = olora_init(weight, RANK, SCALE)
    gram = b.t() @ b
    assert torch.linalg.norm(gram - torch.eye(RANK)) < ORTHO_TOL


def test_olora_init_rank_guard_raises() -> None:
    """rank > min(m,n)//2 raises (degenerate-rank guard)."""
    weight = torch.randn(8, 8)  # min//2 == 4
    with pytest.raises((AssertionError, ValueError)):
        olora_init(weight, rank=5, scale=SCALE)


def test_olora_init_rejects_nonpositive_rank() -> None:
    """rank < 1 raises (a non-positive rank silently slices the whole QR basis)."""
    weight = torch.randn(8, 8)
    with pytest.raises(ValueError, match="rank"):
        olora_init(weight, rank=0, scale=SCALE)


def test_olora_linear_handles_bias_free_linear() -> None:
    """A wrapped bias-free nn.Linear keeps bias=None and still forwards."""
    torch.manual_seed(SEED)
    pretrained = nn.Linear(8, 8, bias=False)
    wrapped = OLoRALinear(pretrained, rank=2, scale=SCALE)
    assert wrapped.bias is None
    assert wrapped(torch.randn(3, 8)).shape == (3, 8)


def test_olora_linear_function_preserving_vs_pretrained() -> None:
    """OLoRALinear(x) matches the wrapped pretrained nn.Linear(x) at init."""
    torch.manual_seed(SEED)
    pretrained = nn.Linear(50, 64)
    wrapped = OLoRALinear(pretrained, RANK, SCALE)
    x = torch.randn(8, 50)
    assert torch.linalg.norm(wrapped(x) - pretrained(x)) < FN_TOL


def test_olora_linear_matches_saved_weight_after_pretrained_mutation() -> None:
    """Wrapper holds its own W0/bias copy: mutating pretrained AFTER wrap is ignored.

    Function-preservation is tested against the weight SAVED at construction, so
    a later in-place edit of the source Linear must not change the wrapper output.
    """
    torch.manual_seed(SEED)
    pretrained = nn.Linear(40, 64)
    x = torch.randn(5, 40)
    saved = pretrained.weight.detach().clone()
    saved_bias = pretrained.bias.detach().clone()
    wrapped = OLoRALinear(pretrained, RANK, SCALE)
    with torch.no_grad():
        pretrained.weight.add_(1.0)  # corrupt the source AFTER wrapping
        pretrained.bias.add_(1.0)
    reference = nn.functional.linear(x, saved, saved_bias)
    assert torch.linalg.norm(wrapped(x) - reference) < FN_TOL


def test_olora_linear_duck_types_nn_linear() -> None:
    """OLoRALinear exposes in_features/out_features like nn.Linear."""
    pretrained = nn.Linear(50, 64)
    wrapped = OLoRALinear(pretrained, RANK, SCALE)
    assert wrapped.in_features == 50
    assert wrapped.out_features == 64


def test_olora_linear_only_ab_trainable() -> None:
    """Only A and B require grad; W0 and bias are frozen."""
    torch.manual_seed(SEED)
    pretrained = nn.Linear(50, 64)
    wrapped = OLoRALinear(pretrained, RANK, SCALE)
    assert wrapped.A.requires_grad is True
    assert wrapped.B.requires_grad is True
    trainable = {name for name, p in wrapped.named_parameters() if p.requires_grad}
    assert trainable == {"A", "B"}


def test_olora_linear_far_fewer_trainable_params() -> None:
    """A wrapped Linear has ~8x fewer trainable params than the dense weight."""
    torch.manual_seed(SEED)
    pretrained = nn.Linear(64, 64)
    wrapped = OLoRALinear(pretrained, RANK, SCALE)
    dense = pretrained.weight.numel()
    trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    assert trainable < dense
    assert dense / trainable > 7.0  # r=4 over 64x64 -> ~8x reduction
