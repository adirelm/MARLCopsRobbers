"""aggregate tests (T10.1) — load + per-round mean±SE curve + final-by-algo/grid."""

from __future__ import annotations

import json

import pytest

from src.results.aggregate import curve, final_by_algorithm, final_by_grid, final_values_by_seed, load_runs


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


def test_load_runs_collapses_double_appended_lines(tmp_path):
    """A byte-identical repeated append must NOT count twice (it would narrow every SE band)."""
    path = tmp_path / "h.jsonl"
    body = "\n".join(json.dumps(r) for r in _records()) + "\n"
    path.write_text(body + body, encoding="utf-8")  # the whole log appended twice
    assert len(load_runs(path)) == len(_records())


def test_load_runs_keeps_last_on_conflicting_run_key_and_warns(tmp_path, caplog):
    """codex W2 R1: duplicate (algorithm, seed, stage, round) with DIFFERENT values must
    not double-weight the seed — the LAST record wins and the conflict is logged."""
    path = tmp_path / "h.jsonl"
    first, last = _rec("qmix", 7, 0, 2, 0, 0.5), _rec("qmix", 7, 0, 2, 0, 0.9)
    path.write_text(json.dumps(first) + "\n" + json.dumps(last) + "\n", encoding="utf-8")
    with caplog.at_level("WARNING", logger="src.results.aggregate"):
        records = load_runs(path)
    assert records == [last]  # single-weight; a crashed-then-resumed rerun is authoritative
    assert any("conflicting duplicate" in message for message in caplog.messages)
    _, mean, _ = curve(records, "capture_rate", "qmix", 0)
    assert mean == [0.9]  # curves see ONE record per key, never a 0.5/0.9 double-weight


def test_final_se_unit_is_the_seed_not_the_pooled_round():
    """SE over 2 per-seed means (0.4, 0.6) is 0.1 — NOT the pooled-6-rounds ~0.045."""
    stats = final_by_algorithm(_records(), "capture_rate", 3, last_k=3)
    assert stats["qmix"][0] == pytest.approx(0.5)  # the mean is unchanged by the fix
    assert stats["qmix"][1] == pytest.approx(0.1)  # stdev([0.4, 0.6]) / sqrt(2)
    assert final_values_by_seed(_records(), "capture_rate", "qmix", 3, last_k=3) == pytest.approx([0.4, 0.6])
