"""sensitivity tests (T10.6) — make_variant changes ONLY the swept key; sweep + aggregate."""

from __future__ import annotations

import pytest

from src.results.sensitivity import aggregate_sensitivity, make_variant, run_sensitivity


def _diff_paths(left, right, prefix=()):
    """Recursively yield the key-paths where nested dicts ``left`` and ``right`` differ."""
    if isinstance(left, dict) and isinstance(right, dict):
        for key in set(left) | set(right):
            yield from _diff_paths(left.get(key), right.get(key), (*prefix, key))
    elif left != right:
        yield prefix


def test_make_variant_changes_only_the_one_key(cfg):
    diffs = set(_diff_paths(make_variant(cfg, 1), make_variant(cfg, 2)))
    assert diffs == {("env", "view_radius_by_grid", 4)}


def test_make_variant_does_not_mutate_input(cfg):
    before = cfg["env"]["view_radius_by_grid"][4]
    make_variant(cfg, 99)
    assert cfg["env"]["view_radius_by_grid"][4] == before


def test_run_sensitivity_sweeps_values_and_seeds(tmp_path, cfg):
    out = tmp_path / "sens.jsonl"

    class _SDK:
        def __init__(self, variant):
            self.radius = variant["env"]["view_radius_by_grid"][4]

        def train(self, algorithm, seed, stage_idx):
            return [{"round": 0, "role": "cop", "loss": 0.1, "capture_rate": 0.2 * self.radius}]

    records = run_sensitivity(_SDK, cfg, [1, 2], [7, 17], 3, out)
    assert len(records) == 4 and {r["value"] for r in records} == {1, 2}
    stats = aggregate_sensitivity(records)
    assert stats[2][0] > stats[1][0]  # a wider radius -> higher capture (the swept effect)
    assert len(out.read_text(encoding="utf-8").splitlines()) == 4


def test_aggregate_dedupes_repeated_sweeps_keeping_the_last():
    """Re-running the sweep APPENDS, so the aggregate must not mix two code revisions.

    Shipped consequence: a second run of the documented reproduce command left 12 records
    in the arm and pulled the published mean toward a blend of both runs — which is exactly
    the drift the "Reproduce:" line exists to let a reader detect.
    """
    stale = [
        {
            "param": "view_radius_4",
            "value": 1,
            "algorithm": "qmix",
            "seed": s,
            "stage": 2,
            "final_capture_rate": 0.10,
        }
        for s in (7, 17, 37)
    ]
    fresh = [
        {
            "param": "view_radius_4",
            "value": 1,
            "algorithm": "qmix",
            "seed": s,
            "stage": 2,
            "final_capture_rate": 0.90,
        }
        for s in (7, 17, 37)
    ]
    mean, se = aggregate_sensitivity(stale + fresh)[1]
    assert mean == pytest.approx(0.90), "stale records leaked into the aggregate"
    assert se == pytest.approx(0.0)


def test_published_table_matches_the_committed_arm():
    """ANALYSIS §9's table must be re-derivable from the committed sweep records."""
    import json  # noqa: PLC0415
    import re  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from src.utils.config_loader import load_config  # noqa: PLC0415

    cfg = load_config()
    arm = Path(cfg["paths"]["runs_dir"]) / "sensitivity_view_radius.jsonl"
    if not arm.exists():
        pytest.skip("sweep arm not present")
    agg = aggregate_sensitivity([json.loads(x) for x in arm.read_text().splitlines() if x.strip()])

    doc = (Path(__file__).resolve().parents[2] / "docs" / "ANALYSIS.md").read_text(encoding="utf-8")
    for radius in (1, 2):
        row = re.search(rf"^\| {radius} \(.*?\| \**([\d.]+) ± ([\d.]+)\**", doc, re.M)
        assert row, f"§9 table lost its radius-{radius} row"
        mean, se = agg[radius]
        assert float(row.group(1)) == pytest.approx(mean, abs=0.001)
        assert float(row.group(2)) == pytest.approx(se, abs=0.001)
