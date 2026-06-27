"""Tests for the Minimax-Q runner + the SDK entry (P-bonus, L11 §5).

Small episode counts keep the per-step LP solves fast while still exercising the full
stack (LP -> learner -> tabular env -> runner -> SDK).
"""

from __future__ import annotations

import copy

import numpy as np

from src.marl.baselines.minimax_runner import train_minimax_q
from src.results.plots import plot_minimax_q
from src.sdk.sdk import MarlSDK


def _small(cfg: dict, episodes: int = 60, window: int = 30) -> dict:
    """Return a deep copy of ``cfg`` with a small Minimax-Q episode budget."""
    c = copy.deepcopy(cfg)
    c["minimax_q"]["episodes"] = episodes
    c["minimax_q"]["window"] = window
    return c


def test_history_structure_and_ranges(cfg):
    """One row per window; capture-rate in [0,1]; the reference game value is finite."""
    history = train_minimax_q(_small(cfg), seed=7)
    assert len(history) == 60 // 30
    for row in history:
        assert {"episode", "capture_rate", "ref_value"} <= row.keys()
        assert 0.0 <= row["capture_rate"] <= 1.0
        assert np.isfinite(row["ref_value"])


def test_reproducible_same_seed(cfg):
    """A seeded run is bit-for-bit reproducible (seeded RNG + deterministic LP)."""
    c = _small(cfg, episodes=30, window=30)
    assert train_minimax_q(c, 11) == train_minimax_q(c, 11)


def test_sdk_entry_runs_baseline(cfg):
    """The single SDK seam ``run_minimax_q_baseline`` drives the whole stack."""
    history = MarlSDK(_small(cfg, episodes=30, window=30)).run_minimax_q_baseline(7)
    assert history and 0.0 <= history[-1]["capture_rate"] <= 1.0


def test_plot_minimax_q_writes_png(tmp_path):
    """F7 renders a non-empty PNG headlessly (Agg), with the escape-floor reference line."""
    history = [{"episode": e, "capture_rate": 0.4, "ref_value": -0.01 * e} for e in (100, 200, 300)]
    out = plot_minimax_q(history, tmp_path / "minimax_q.png", escape_floor=-0.277)
    assert out.exists() and out.stat().st_size > 0
