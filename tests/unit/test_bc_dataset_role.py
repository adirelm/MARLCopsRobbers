"""Role-parameterized BC dataset + episode-level split tests (T4.4, P4c S2).

Extends the P3 cop-only builder to the thief role (records ``obs["thief"]`` with
the greedy thief expert label) and pins a deterministic EPISODE-level train/val
split that never lands two records of the same episode on opposite sides.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.marl.data.bc_dataset import build_bc_dataset, episode_split


def test_thief_role_labels_below_thief_action_count(cfg):
    """role='thief' yields labels in [0, a_thief) (4 dirs, no PLACE_BARRIER)."""
    _obs, _sc, actions, _eids, _m = build_bc_dataset(cfg, (3, 3), 24, seed=1, role="thief")
    assert actions.min() >= 0
    assert actions.max() < cfg["env"]["actions"]["a_thief"]


def test_cop_default_backward_compatible(cfg):
    """The cop default reproduces the same labels as an explicit role='cop'."""
    a = build_bc_dataset(cfg, (3, 3), 20, seed=9)
    b = build_bc_dataset(cfg, (3, 3), 20, seed=9, role="cop")
    np.testing.assert_array_equal(a[2], b[2])
    np.testing.assert_array_equal(a[0], b[0])


def test_thief_differs_from_cop_same_seed(cfg):
    """The thief role records DIFFERENT obs+labels than the cop on the same seed."""
    cop = build_bc_dataset(cfg, (3, 3), 30, seed=3, role="cop")
    thief = build_bc_dataset(cfg, (3, 3), 30, seed=3, role="thief")
    # The thief observation channel-0 (self) differs from the cop self plane.
    assert not np.array_equal(cop[0], thief[0])


def test_episode_ids_returned_and_grouped(cfg):
    """build_bc_dataset returns a (n_pairs,) episode-id array of contiguous runs."""
    _obs, _sc, _a, episode_ids, _m = build_bc_dataset(cfg, (3, 3), 25, seed=2)
    assert episode_ids.shape == (25,)
    assert episode_ids.dtype == np.int64
    # ids are non-decreasing (records appended episode by episode).
    assert np.all(np.diff(episode_ids) >= 0)
    assert episode_ids.min() == 0


def test_episode_split_disjoint_and_by_episode(cfg):
    """episode_split partitions WHOLE episodes: no episode straddles train/val."""
    _obs, _sc, _a, episode_ids, _m = build_bc_dataset(cfg, (3, 3), 60, seed=4)
    train_idx, val_idx = episode_split(episode_ids, cfg)
    assert set(train_idx.tolist()).isdisjoint(val_idx.tolist())
    assert len(train_idx) + len(val_idx) == len(episode_ids)
    train_eps = set(episode_ids[train_idx].tolist())
    val_eps = set(episode_ids[val_idx].tolist())
    assert train_eps.isdisjoint(val_eps)  # never adjacent records of one episode


def test_episode_split_is_deterministic(cfg):
    """The same episode-ids + split_seed reproduce identical train/val indices."""
    _o, _s, _a, episode_ids, _m = build_bc_dataset(cfg, (3, 3), 60, seed=4)
    t1, v1 = episode_split(episode_ids, cfg)
    t2, v2 = episode_split(episode_ids, cfg)
    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_array_equal(v1, v2)


def test_episode_split_respects_val_fraction(cfg):
    """The held-out episode count is ~ bc.val_fraction of the total episodes."""
    _o, _s, _a, episode_ids, _m = build_bc_dataset(cfg, (3, 3), 120, seed=6)
    _train_idx, val_idx = episode_split(episode_ids, cfg)
    n_eps = len(set(episode_ids.tolist()))
    val_eps = len(set(episode_ids[val_idx].tolist()))
    expected = min(n_eps - 1, max(1, round(cfg["bc"]["val_fraction"] * n_eps)))
    assert val_eps == expected


def test_episode_split_raises_on_single_episode(cfg):
    """A single-episode dataset cannot split into a non-empty train+val (ValueError)."""
    with pytest.raises(ValueError, match="episodes"):
        episode_split(np.zeros(8, dtype=np.int64), cfg)


def test_build_bc_dataset_rejects_unknown_role(cfg):
    """An unknown role is rejected (role is public API added in P4c)."""
    with pytest.raises(ValueError, match="unknown role"):
        build_bc_dataset(cfg, (3, 3), 4, seed=7, role="robber")
