"""sensitivity tests (T10.6) — make_variant changes ONLY the swept key; sweep + aggregate."""

from __future__ import annotations

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
    assert diffs == {("env", "view_radius_by_grid", 5)}


def test_make_variant_does_not_mutate_input(cfg):
    before = cfg["env"]["view_radius_by_grid"][5]
    make_variant(cfg, 99)
    assert cfg["env"]["view_radius_by_grid"][5] == before


def test_run_sensitivity_sweeps_values_and_seeds(tmp_path, cfg):
    out = tmp_path / "sens.jsonl"

    class _SDK:
        def __init__(self, variant):
            self.radius = variant["env"]["view_radius_by_grid"][5]

        def train(self, algorithm, seed, stage_idx):
            return [{"round": 0, "role": "cop", "loss": 0.1, "capture_rate": 0.2 * self.radius}]

    records = run_sensitivity(_SDK, cfg, [1, 2], [7, 17], 3, out)
    assert len(records) == 4 and {r["value"] for r in records} == {1, 2}
    stats = aggregate_sensitivity(records)
    assert stats[2][0] > stats[1][0]  # a wider radius -> higher capture (the swept effect)
    assert len(out.read_text(encoding="utf-8").splitlines()) == 4
