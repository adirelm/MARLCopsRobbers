"""Compute resource governance — keep the host responsive during training (T4.6).

PyTorch defaults to grabbing EVERY CPU core for intra-op parallelism, which on a
laptop spikes all cores and freezes the UI. This module reads the ``compute``
config block and caps torch's thread pools BEFORE any training runs, and exports
the OMP/MKL thread limits so any spawned worker inherits the same cap. Every
limit is config-driven (CLAUDE.md §4, no hardcode): tune ``compute.num_threads``
to leave cores free on a laptop, or raise it on a dedicated training box.
"""

from __future__ import annotations

import contextlib
import os

import torch


def apply_compute_limits(cfg: dict) -> int:
    """Cap torch (and OMP/MKL) CPU threads from the ``compute`` config block.

    Pins intra-op threads via :func:`torch.set_num_threads` and, once per
    process, inter-op threads via :func:`torch.set_num_interop_threads` (a second
    call after the pool is sized raises ``RuntimeError`` and is suppressed). Also
    exports ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` so spawned workers inherit
    the cap. Call this at every training entry point (SDK/trainer) so a run can
    never grab all cores and freeze the host.

    Args:
        cfg: Loaded config; reads ``compute.num_threads`` (intra-op) and
            ``compute.num_interop_threads`` (inter-op).

    Returns:
        The effective intra-op thread count (:func:`torch.get_num_threads`).
    """
    compute = cfg["compute"]
    n_intra = int(compute["num_threads"])
    n_inter = int(compute["num_interop_threads"])
    os.environ["OMP_NUM_THREADS"] = str(n_intra)
    os.environ["MKL_NUM_THREADS"] = str(n_intra)
    torch.set_num_threads(n_intra)
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(n_inter)  # inter-op pool is sizable only once per process
    return torch.get_num_threads()
