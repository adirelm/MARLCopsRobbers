"""Unit tests for the mixer seam (T4.0a, P4a Stage 2).

Pure-tensor gates for the swappable Mixer ABC and the VDN mixer (eq6):
VDN sums per-agent Q_i and ignores the global state. BaseMixer is abstract.
No training loop. torch is seeded for determinism.
"""

from __future__ import annotations

import pytest
import torch

from src.marl.mixers.base_mixer import BaseMixer
from src.marl.mixers.vdn_mixer import VdnMixer

SEED = 7
STATE_DIM = 77


def test_base_mixer_is_abstract() -> None:
    """BaseMixer cannot be instantiated (abstract forward)."""
    with pytest.raises(TypeError):
        BaseMixer()  # type: ignore[abstract]


def test_vdn_sums_agents_exactly() -> None:
    """VDN output equals q_agents.sum(dim=1, keepdim=True) exactly (N=2)."""
    torch.manual_seed(SEED)
    mixer = VdnMixer()
    q_agents = torch.randn(4, 2)
    state = torch.randn(4, STATE_DIM)
    q_tot = mixer(q_agents, state)
    expected = q_agents.sum(dim=1, keepdim=True)
    assert q_tot.shape == (4, 1)
    assert torch.equal(q_tot, expected)


def test_vdn_ignores_state() -> None:
    """VDN gives the same output for different states (state is ignored)."""
    torch.manual_seed(SEED)
    mixer = VdnMixer()
    q_agents = torch.randn(3, 2)
    state_a = torch.randn(3, STATE_DIM)
    state_b = torch.randn(3, STATE_DIM)
    assert torch.equal(mixer(q_agents, state_a), mixer(q_agents, state_b))
