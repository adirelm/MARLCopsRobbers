"""Tests for the SDK self-play training entry (T4.6/T4.8).

MarlSDK.train is the single training entry. Runs a TINY qmix + iql stage on 2x2
(shrunk self-play/replay knobs) so the real loop is exercised through the facade
fast, and asserts the algorithm switch + stage resolution + error handling.
"""

from __future__ import annotations

import copy

import pytest

from src.sdk.sdk import MarlSDK


def _tiny(cfg: dict) -> dict:
    """Shrink self-play/replay so a stage trains in well under a second."""
    c = copy.deepcopy(cfg)
    c["selfplay"]["episodes_per_round"] = 2
    c["selfplay"]["update_ratio"] = 1
    c["selfplay"]["rounds"] = 2
    c["algo"]["batch_episodes"] = 2
    c["replay"]["buffer_episodes"] = 16
    return c


def test_train_qmix_returns_per_round_history(cfg):
    """train('qmix') runs selfplay.rounds rounds through the SDK and reports history."""
    history = MarlSDK(_tiny(cfg)).train("qmix", seed=7, stage_idx=0)
    assert len(history) == 2
    assert all({"round", "role", "loss", "capture_rate"} <= set(rec) for rec in history)


def test_train_iql_baseline_runs(cfg):
    """The IQL baseline arm trains without a mixer (no global-state path)."""
    history = MarlSDK(_tiny(cfg)).train("iql", seed=3, stage_idx=0)
    assert len(history) == 2


def test_train_rejects_unknown_algorithm(cfg):
    """An unknown algorithm name is rejected before any training starts."""
    with pytest.raises(ValueError, match="unknown algorithm"):
        MarlSDK(cfg).train("ppo", seed=1)
