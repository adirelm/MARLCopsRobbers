"""make_figures integration (T10.2) — synthetic runs -> 4 PNGs + manifest (headless Agg)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.results.make_figures import _focus_stage, main


def test_focus_stage_prefers_most_covered_then_largest():
    records = [{"algorithm": a, "stage": 2, "seed": 7} for a in ("qmix", "vdn", "iql")]
    records += [{"algorithm": "qmix", "stage": 3, "seed": 7}]  # only qmix reached the slow stage
    assert _focus_stage(records) == 2  # stage 2 (3 arms) beats stage 3 (1 arm)


def _write_runs(path: Path) -> None:
    records = []
    for algo in ("qmix", "vdn", "iql"):
        for seed in (7, 17, 37):
            for stage, grid in [(0, 2), (3, 5)]:
                for rnd in range(5):
                    records.append(
                        {
                            "algorithm": algo,
                            "seed": seed,
                            "stage": stage,
                            "grid": grid,
                            "round": rnd,
                            "role": "cop" if rnd % 2 == 0 else "thief",
                            "loss": 0.5 / (rnd + 1),
                            "capture_rate": 0.3 + 0.1 * rnd,
                        }
                    )
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _cfg_tmp(cfg: dict, tmp_path: Path) -> dict:
    cfg = json.loads(json.dumps(cfg))
    cfg["paths"]["runs_dir"] = str(tmp_path / "runs")
    cfg["paths"]["figures_dir"] = str(tmp_path / "figs")
    return cfg


def test_make_figures_writes_four_pngs_and_manifest(tmp_path, cfg):
    cfg = _cfg_tmp(cfg, tmp_path)
    (tmp_path / "runs").mkdir()
    _write_runs(tmp_path / "runs" / "history.jsonl")
    saved = main(cfg)
    assert len(saved) == 4
    for path in saved:
        assert Path(path).exists() and Path(path).stat().st_size > 1000  # a real PNG, not empty
    manifest = json.loads((tmp_path / "figs" / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["runs"] == 18 and manifest["algorithms"] == ["iql", "qmix", "vdn"]


def test_make_figures_raises_without_runs(tmp_path, cfg):
    cfg = _cfg_tmp(cfg, tmp_path)
    with pytest.raises(SystemExit):
        main(cfg)
