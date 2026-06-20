"""Tests for compute resource governance (T4.6 — keep the host responsive).

PyTorch defaults to grabbing every CPU core, which freezes the laptop UI during
training. ``apply_compute_limits`` caps the torch thread pools from the config
``compute`` block. torch is process-global, so these assertions read back the
caps it set.
"""

from __future__ import annotations

import os

import torch

from src.utils.compute import apply_compute_limits


def test_apply_compute_limits_caps_intra_op_threads(cfg):
    """torch intra-op threads are pinned to compute.num_threads (not all cores)."""
    apply_compute_limits(cfg)
    assert torch.get_num_threads() == int(cfg["compute"]["num_threads"])


def test_apply_compute_limits_returns_effective_count(cfg):
    """The helper returns the effective torch thread count it applied."""
    assert apply_compute_limits(cfg) == torch.get_num_threads()


def test_apply_compute_limits_exports_omp_env(cfg):
    """OMP/MKL thread caps are exported so spawned workers inherit the limit."""
    apply_compute_limits(cfg)
    assert os.environ["OMP_NUM_THREADS"] == str(cfg["compute"]["num_threads"])
    assert os.environ["MKL_NUM_THREADS"] == str(cfg["compute"]["num_threads"])


def test_apply_compute_limits_is_idempotent(cfg):
    """A second call must not raise (inter-op pool can only be sized once)."""
    apply_compute_limits(cfg)
    apply_compute_limits(cfg)  # interop re-set is suppressed, not fatal
    assert torch.get_num_threads() == int(cfg["compute"]["num_threads"])
