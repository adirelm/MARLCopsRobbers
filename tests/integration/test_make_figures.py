"""make_figures integration (T10.2) — synthetic runs -> 6 PNGs + manifest (headless Agg)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.results._figure_stages import comparison_stage, final_stage, focus_stage
from src.results.make_figures import main


def test_focus_stage_prefers_most_covered_then_largest():
    records = [{"algorithm": a, "stage": 2, "seed": 7} for a in ("qmix", "vdn", "iql")]
    records += [{"algorithm": "qmix", "stage": 3, "seed": 7}]  # only qmix reached the slow stage
    assert focus_stage(records) == 2  # stage 2 (3 arms) beats stage 3 (1 arm)


def test_final_stage_is_the_largest_present():
    records = [{"algorithm": "qmix", "stage": s, "seed": 7} for s in (0, 1, 2, 3)]
    assert final_stage(records) == 3  # §5.1 5x5 final test


def test_comparison_stage_is_the_largest_multi_cop_stage():
    cfg = {"env": {"curriculum": {"num_cops_by_stage": [1, 1, 2, 1]}}}
    records = [{"algorithm": "qmix", "stage": s, "seed": 7} for s in (0, 1, 2, 3)]
    assert comparison_stage(cfg, records) == 2  # 4x4 (2 cops), NOT the degenerate 1-cop 5x5


def test_comparison_stage_falls_back_to_focus_when_no_multi_cop_stage():
    cfg = {"env": {"curriculum": {"num_cops_by_stage": [1, 1, 1, 1]}}}
    records = [{"algorithm": a, "stage": 1, "seed": 7} for a in ("qmix", "vdn", "iql")]
    assert comparison_stage(cfg, records) == 1  # no 2-cop stage -> focus_stage


def _write_runs(path: Path, with_returns: bool = False) -> None:
    records = []
    for algo in ("qmix", "vdn", "iql"):
        for seed in (7, 17, 37):
            for stage, grid in [(0, 2), (3, 5)]:
                for rnd in range(5):
                    record = {
                        "algorithm": algo,
                        "seed": seed,
                        "stage": stage,
                        "grid": grid,
                        "round": rnd,
                        "role": "cop" if rnd % 2 == 0 else "thief",
                        "loss": 0.5 / (rnd + 1),
                        "capture_rate": 0.3 + 0.1 * rnd,
                    }
                    if with_returns:  # a FRESH reproduction logs returns in history.jsonl
                        record.update({"cop_return": 0.1 * rnd, "thief_return": -0.05 * rnd})
                    records.append(record)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _cfg_tmp(cfg: dict, tmp_path: Path) -> dict:
    cfg = json.loads(json.dumps(cfg))
    cfg["paths"]["runs_dir"] = str(tmp_path / "runs")
    cfg["paths"]["figures_dir"] = str(tmp_path / "figs")
    cfg["paths"]["experiment_manifest"] = str(tmp_path / "figs" / "experiment_manifest.json")
    return cfg


def test_make_figures_writes_six_pngs_and_manifest(tmp_path, cfg):
    cfg = _cfg_tmp(cfg, tmp_path)
    (tmp_path / "runs").mkdir()
    _write_runs(tmp_path / "runs" / "history.jsonl")
    saved = main(cfg)
    # 4 line/bar figures + the two §9.3 variety figures (BOX + HEATMAP)
    assert len(saved) == 6
    names = {Path(p).name for p in saved}
    assert {"final_distribution.png", "capture_heatmap.png"} <= names
    for path in saved:
        assert Path(path).exists() and Path(path).stat().st_size > 1000  # a real PNG, not empty
    manifest = json.loads((tmp_path / "figs" / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["runs"] == 18 and manifest["algorithms"] == ["iql", "qmix", "vdn"]


def test_make_figures_raises_without_runs(tmp_path, cfg):
    cfg = _cfg_tmp(cfg, tmp_path)
    with pytest.raises(SystemExit):
        main(cfg)


def test_f1b_falls_back_to_history_returns_on_a_fresh_run(tmp_path, cfg):
    """codex W2 R2: with no returns_history.jsonl, a FRESH run's history.jsonl
    (which carries the return columns) must still produce F1b."""
    cfg = _cfg_tmp(cfg, tmp_path)
    (tmp_path / "runs").mkdir()
    _write_runs(tmp_path / "runs" / "history.jsonl", with_returns=True)
    names = {Path(p).name for p in main(cfg)}
    assert "return_curves.png" in names


def test_f1b_skipped_not_faked_without_return_columns(tmp_path, cfg):
    """codex W2 R2: return-less records are EXCLUDED (no flat-zero fabrication) and
    an existing-but-EMPTY returns_history.jsonl must not crash the run."""
    cfg = _cfg_tmp(cfg, tmp_path)
    (tmp_path / "runs").mkdir()
    _write_runs(tmp_path / "runs" / "history.jsonl", with_returns=False)
    (tmp_path / "runs" / "returns_history.jsonl").write_text("", encoding="utf-8")
    names = {Path(p).name for p in main(cfg)}
    assert "return_curves.png" not in names  # skipped, never a fabricated 0.0 curve
