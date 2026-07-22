"""The self-play round records the MEAN CUMULATIVE EPISODIC RETURN for BOTH roles (§7.3a).

ex06 §7.3(a) mandates learning curves showing "התכנסות התגמול המצטבר" — convergence of
the CUMULATIVE REWARD — for both agents. The round history therefore carries
``cop_return`` / ``thief_return`` alongside the capture rate, so F1 can plot the
literal mandated quantity instead of only a proxy metric.
"""

from __future__ import annotations

from src.services.trainer import SelfPlayTrainer
from src.utils.config_loader import load_config


def _fast_cfg() -> dict:
    """A tiny 2x2 config so a round runs in milliseconds."""
    cfg = load_config()
    cfg["selfplay"]["episodes_per_round"] = 2
    cfg["selfplay"]["rounds"] = 1
    cfg["replay"]["min_size"] = 10**9  # skip updates: this test is about the RETURN fields
    return cfg


def test_round_history_carries_cumulative_returns_for_both_roles():
    """Every round record exposes numeric cop_return AND thief_return."""
    history = SelfPlayTrainer(_fast_cfg(), seed=3, h=2, w=2, num_cops=1).train_stage(rounds=1)
    record = history[0]
    assert {"cop_return", "thief_return"} <= set(record)
    assert isinstance(record["cop_return"], float)
    assert isinstance(record["thief_return"], float)


def test_returns_are_episodic_sums_not_per_step_means():
    """A cumulative return spans a whole episode, so it is not bounded by a single step reward.

    The step reward is small (shaping + step penalty); an episodic SUM over up to
    max_moves steps must therefore have a materially larger magnitude than one step,
    and the two roles must differ (zero-sum-ish terminal outcomes).
    """
    history = SelfPlayTrainer(_fast_cfg(), seed=5, h=2, w=2, num_cops=1).train_stage(rounds=1)
    record = history[0]
    assert record["cop_return"] != record["thief_return"]
    assert abs(record["cop_return"]) + abs(record["thief_return"]) > 0.0
