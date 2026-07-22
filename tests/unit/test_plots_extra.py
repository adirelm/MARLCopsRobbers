"""plots_extra tests (V3 §9.3) — the BOX + HEATMAP figures render real PNGs."""

from __future__ import annotations

from pathlib import Path

from src.results.plots_extra import (
    final_values_by_seed,
    plot_capture_heatmap,
    plot_final_distribution,
)


def _records() -> list[dict]:
    """Two algorithms x two seeds x two stages of synthetic per-round records."""
    out = []
    for algorithm, base in (("iql", 0.4), ("qmix", 0.7)):
        for stage, grid in ((0, 4), (1, 6)):
            for seed in (7, 8):
                for rnd in range(6):
                    out.append(
                        {
                            "algorithm": algorithm,
                            "seed": seed,
                            "stage": stage,
                            "grid": grid,
                            "round": rnd,
                            "role": "cop" if rnd % 2 == 0 else "thief",
                            "loss": 0.1,
                            "capture_rate": base + 0.01 * seed + 0.02 * rnd,
                        }
                    )
    return out


def test_final_values_by_seed_returns_one_value_per_seed():
    values = final_values_by_seed(_records(), "capture_rate", "iql", 0)
    assert len(values) == 2  # one aggregated final value per seed
    assert all(isinstance(v, float) for v in values)


def test_plot_final_distribution_writes_figure(tmp_path):
    out = plot_final_distribution(_records(), "capture_rate", 1, tmp_path / "final_distribution.png")
    assert Path(out).exists() and Path(out).stat().st_size > 0


def test_plot_capture_heatmap_writes_figure(tmp_path):
    out = plot_capture_heatmap(_records(), "capture_rate", tmp_path / "capture_heatmap.png")
    assert Path(out).exists() and Path(out).stat().st_size > 0


def test_final_distribution_skips_algorithm_absent_from_stage(tmp_path):
    records = [
        *_records(),
        {
            "algorithm": "ghost",
            "seed": 7,
            "stage": 9,
            "grid": 2,
            "round": 0,
            "role": "cop",
            "loss": 0.1,
            "capture_rate": 0.1,
        },
    ]
    assert final_values_by_seed(records, "capture_rate", "ghost", 1) == []
    out = plot_final_distribution(records, "capture_rate", 1, tmp_path / "d.png")
    assert Path(out).exists() and Path(out).stat().st_size > 0
