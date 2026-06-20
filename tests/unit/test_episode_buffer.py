"""Unit tests for src.marl.replay.episode_buffer (T3.1).

Covers add/sample batched shapes (B,T,N,*), filled-mask padding zeroing, seeded
sample reproducibility, capacity ring overwrite, and THE FOUR-SOURCE provenance
test (one episode per SourceTag lands with the right tag + a mixed sample draws
across more than one source).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.marl.data.schemas import SourceTag
from src.marl.replay.episode_buffer import CentralizedReplayBuffer

# Small, buffer-only dims (decoupled from the real grid; (C,W,W)=(5,5,5)).
_CAP = 4
_T_MAX = 6
_N = 2
_C = 5
_W = 5
_SCALARS = 6
_STATE_DIM = 77
_N_ACTIONS = 5


def _make_buffer(capacity: int = _CAP, n_agents: int = _N, seed: int = 0):
    return CentralizedReplayBuffer(
        capacity=capacity,
        t_max=_T_MAX,
        n_agents=n_agents,
        obs_channels=_C,
        w_v=_W,
        obs_scalars=_SCALARS,
        state_dim=_STATE_DIM,
        n_actions=_N_ACTIONS,
        seed=seed,
    )


def _make_episode(length: int, n_agents: int = _N, fill: float = 1.0) -> dict:
    """Build an UNPADDED episode dict of real-step length ``length``."""
    t = length
    rng = np.random.default_rng(int(fill * 100) + length)
    return {
        "obs": np.full((t + 1, n_agents, _C, _W, _W), fill, dtype=np.float32),
        "scalars": np.full((t + 1, n_agents, _SCALARS), fill, dtype=np.float32),
        "global_state": np.full((t + 1, _STATE_DIM), fill, dtype=np.float32),
        "actions": rng.integers(0, _N_ACTIONS, size=(t, n_agents)).astype(np.int64),
        "reward": np.full((t, n_agents), fill, dtype=np.float32),
        "done": np.array([i == t - 1 for i in range(t)], dtype=bool),
        "filled": np.ones(t, dtype=bool),
        "next_legal_mask": np.ones((t, n_agents, _N_ACTIONS), dtype=bool),
        "hidden_seed": np.int64(123),
    }


def test_len_grows_then_caps_at_capacity():
    """__len__ tracks stored episodes and never exceeds capacity."""
    buf = _make_buffer()
    assert len(buf) == 0
    for i in range(_CAP + 3):
        buf.add_episode(_make_episode(_T_MAX, fill=i), SourceTag.RANDOM)
        assert len(buf) == min(i + 1, _CAP)


def test_sample_returns_batched_shapes():
    """sample returns (B,T,N,*) arrays for every per-step episode field."""
    buf = _make_buffer()
    for i in range(_CAP):
        buf.add_episode(_make_episode(_T_MAX, fill=i + 1), SourceTag.EXPERT)
    batch = buf.sample(3)
    # obs/scalars live on the T+1 axis (carry the terminal next-obs for P4 unroll).
    assert batch["obs"].shape == (3, _T_MAX + 1, _N, _C, _W, _W)
    assert batch["scalars"].shape == (3, _T_MAX + 1, _N, _SCALARS)
    assert batch["global_state"].shape == (3, _T_MAX + 1, _STATE_DIM)
    assert batch["actions"].shape == (3, _T_MAX, _N)
    assert batch["reward"].shape == (3, _T_MAX, _N)
    assert batch["done"].shape == (3, _T_MAX)
    assert batch["filled"].shape == (3, _T_MAX)
    assert batch["next_legal_mask"].shape == (3, _T_MAX, _N, _N_ACTIONS)
    assert batch["actions"].dtype == np.int64
    assert batch["reward"].dtype == np.float32
    assert batch["done"].dtype == bool
    assert len(batch["source_tag"]) == 3


def test_filled_mask_zeroes_padded_steps():
    """A short episode pads to t_max with filled=0 and zeroed per-step data."""
    buf = _make_buffer()
    short = 2
    buf.add_episode(_make_episode(short, fill=9.0), SourceTag.SELF_PLAY)
    batch = buf.sample(1)
    filled = batch["filled"][0]
    assert filled[:short].all()
    assert not filled[short:].any()
    # Padded steps carry zeroed reward/actions despite the fill=9 real steps.
    assert np.all(batch["reward"][0, short:] == 0.0)
    assert np.all(batch["actions"][0, short:] == 0)
    # obs has a T+1 axis: frames [0..short] are real (incl. the terminal next-obs
    # at index `short`); only frames AFTER the next-obs are zero-padded.
    assert np.all(batch["obs"][0, : short + 1] == 9.0)
    assert np.all(batch["obs"][0, short + 1 :] == 0.0)


def test_seeded_sampling_is_reproducible():
    """Two buffers with the same seed and inputs draw identical samples."""
    buf_a = _make_buffer(seed=42)
    buf_b = _make_buffer(seed=42)
    for i in range(_CAP):
        ep = _make_episode(_T_MAX, fill=i + 1)
        buf_a.add_episode(ep, SourceTag.LIVE_CTDE)
        buf_b.add_episode(ep, SourceTag.LIVE_CTDE)
    sample_a = buf_a.sample(3)
    sample_b = buf_b.sample(3)
    np.testing.assert_array_equal(sample_a["obs"], sample_b["obs"])
    np.testing.assert_array_equal(sample_a["actions"], sample_b["actions"])
    assert sample_a["source_tag"] == sample_b["source_tag"]


def test_capacity_ring_overwrites_oldest():
    """Adding past capacity overwrites the oldest episode (ring buffer)."""
    buf = _make_buffer(capacity=2)
    buf.add_episode(_make_episode(_T_MAX, fill=1.0), SourceTag.EXPERT)
    buf.add_episode(_make_episode(_T_MAX, fill=2.0), SourceTag.EXPERT)
    buf.add_episode(_make_episode(_T_MAX, fill=3.0), SourceTag.EXPERT)  # evicts fill=1
    assert len(buf) == 2
    # Every stored obs is now a 2-fill or 3-fill episode; the 1-fill is gone.
    big = buf.sample(50)
    seen = {round(float(v)) for v in np.unique(big["obs"])}
    assert 1 not in seen
    assert seen <= {2, 3}


def test_thief_buffer_single_agent_axis():
    """A thief buffer with n_agents=1 yields an N=1 agent axis."""
    buf = _make_buffer(n_agents=1)
    buf.add_episode(_make_episode(_T_MAX, n_agents=1, fill=1.0), SourceTag.RANDOM)
    batch = buf.sample(1)
    assert batch["obs"].shape == (1, _T_MAX + 1, 1, _C, _W, _W)
    assert batch["actions"].shape == (1, _T_MAX, 1)


def test_terminal_next_obs_survives_add_then_sample():
    """The terminal next-obs frame obs[real_t] (T+1 axis) round-trips add->sample (P4 unroll)."""
    buf = _make_buffer(n_agents=1)
    ep = _make_episode(_T_MAX, n_agents=1, fill=1.0)
    sentinel = 7.0
    ep["obs"][_T_MAX, 0] = sentinel  # terminal next-obs frame (index real_t == T)
    ep["scalars"][_T_MAX, 0] = sentinel
    buf.add_episode(ep, SourceTag.LIVE_CTDE)
    batch = buf.sample(1)
    assert np.all(batch["obs"][0, _T_MAX, 0] == sentinel)
    assert np.all(batch["scalars"][0, _T_MAX, 0] == sentinel)


def test_agent_axis_mismatch_raises_no_silent_broadcast():
    """An N=1 episode fed to an N=2 buffer FAILS LOUD (kills the silent size-1 broadcast)."""
    buf = _make_buffer(n_agents=2)
    bad = _make_episode(_T_MAX, n_agents=1, fill=1.0)
    with pytest.raises(ValueError, match="agent"):
        buf.add_episode(bad, SourceTag.RANDOM)


def test_four_source_tags_land_and_mix():
    """One episode per SourceTag lands with the right tag; a mixed sample spans >1."""
    buf = _make_buffer(capacity=4)
    tags = [SourceTag.EXPERT, SourceTag.SELF_PLAY, SourceTag.RANDOM, SourceTag.LIVE_CTDE]
    for i, tag in enumerate(tags):
        buf.add_episode(_make_episode(_T_MAX, fill=i + 1), tag)
    assert len(buf) == 4
    # All four provenance tags are retrievable from a large draw.
    big = buf.sample(64)
    drawn = set(big["source_tag"])
    assert drawn == set(tags)
    # A single sampled batch draws across more than one source.
    mixed = buf.sample(4)
    assert len(set(mixed["source_tag"])) > 1


def test_sample_empty_buffer_raises():
    """Sampling before any episode is added raises (no silent empty batch)."""
    buf = _make_buffer()
    with pytest.raises(ValueError, match="empty"):
        buf.sample(1)
