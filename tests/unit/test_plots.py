"""plots tests (T10.2) — the curve figure skips an algorithm absent from the focus stage."""

from __future__ import annotations

from pathlib import Path

from src.results.plots import plot_curve_figure, plot_sensitivity


def test_plot_sensitivity_writes_figure(tmp_path):
    out = plot_sensitivity({1: (0.3, 0.05), 2: (0.6, 0.04)}, "view radius", "sensitivity", tmp_path / "s.png")
    assert Path(out).exists() and Path(out).stat().st_size > 0


def test_curve_figure_skips_algo_absent_from_stage(tmp_path):
    # qmix has stage-3 data; "ghost" appears only at stage 0 -> skipped on the stage-3 plot
    records = [
        {
            "algorithm": "qmix",
            "seed": 7,
            "stage": 3,
            "grid": 5,
            "round": rnd,
            "role": "cop",
            "loss": 0.1,
            "capture_rate": 0.5,
        }
        for rnd in range(3)
    ] + [
        {
            "algorithm": "ghost",
            "seed": 7,
            "stage": 0,
            "grid": 2,
            "round": 0,
            "role": "cop",
            "loss": 0.1,
            "capture_rate": 0.1,
        }
    ]
    out = plot_curve_figure(records, "capture_rate", 3, "title", "ylabel", tmp_path / "c.png")
    assert Path(out).exists() and Path(out).stat().st_size > 0
