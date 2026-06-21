"""aggregate tests (T10.1) — load + per-round mean±SE curve + final-by-algo/grid."""

from __future__ import annotations

import json

import pytest

from src.results.aggregate import curve, final_by_algorithm, final_by_grid, load_runs


def _rec(algo, seed, stage, grid, rnd, cap, loss=0.1):  # noqa: PLR0913 — one kwarg per record field
    return {
        "algorithm": algo,
        "seed": seed,
        "stage": stage,
        "grid": grid,
        "round": rnd,
        "role": "cop",
        "loss": loss,
        "capture_rate": cap,
    }


def _records():
    recs = []
    for seed, cap in [(7, 0.4), (17, 0.6)]:  # two seeds -> mean 0.5
        for rnd in range(3):
            recs.append(_rec("qmix", seed, 3, 5, rnd, cap))
            recs.append(_rec("iql", seed, 3, 5, rnd, cap - 0.2))
    for seed in (7, 17):  # a smaller stage for qmix only (F6 scaling point)
        recs.append(_rec("qmix", seed, 0, 2, 0, 0.9))
    return recs


def test_load_runs_reads_jsonl(tmp_path):
    path = tmp_path / "h.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _records()) + "\n", encoding="utf-8")
    assert len(load_runs(path)) == len(_records())
    assert load_runs(tmp_path / "missing.jsonl") == []


def test_curve_is_cross_seed_mean_se():
    rounds, mean, se = curve(_records(), "capture_rate", "qmix", 3)
    assert rounds == [0, 1, 2]
    assert mean == pytest.approx([0.5, 0.5, 0.5])
    assert all(s > 0 for s in se)  # two distinct seeds -> nonzero SE


def test_final_by_algorithm_ranks_qmix_above_iql():
    stats = final_by_algorithm(_records(), "capture_rate", 3, last_k=2)
    assert stats["qmix"][0] > stats["iql"][0]


def test_final_by_grid_has_per_grid_points():
    by_grid = final_by_grid(_records(), "capture_rate", "qmix", last_k=2)
    assert set(by_grid) == {2, 5}
    assert by_grid[2][0] == pytest.approx(0.9)
