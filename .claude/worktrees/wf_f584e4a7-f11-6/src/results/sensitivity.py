"""§9 single-parameter sensitivity sweep (T10.6) — vary ``env.view_radius_by_grid[4]`` ONLY.

Sweeps the ONE config key config names as the §9 param (the 4x4 focus-stage view radius)
over a small grid while holding EVERYTHING else fixed (seeds / nets / replay / gamma /
target cadence identical). This is the §9 SENSITIVITY analysis — explicitly distinct from
T4.7 IQL/VDN/QMIX ABLATION (which swaps the algorithm). ``make_variant`` changes exactly
that key (a test deep-diffs to prove it); ``run_sensitivity`` trains each variant and
appends records; ``aggregate_sensitivity`` reduces to capture-rate vs the swept value.
"""

from __future__ import annotations

import copy
import statistics
from collections import defaultdict
from pathlib import Path

from src.results.run_log import append_records
from src.services.finetune import stage_params

_SWEPT_KEY = "view_radius_by_grid"
_SWEPT_GRID = 4  # sweep the 4x4 focus-stage view radius (5x5 training is too slow to sweep)


def make_variant(cfg: dict, radius: int) -> dict:
    """Deep-copy ``cfg`` with ONLY ``env.view_radius_by_grid[4]`` set to ``radius``."""
    variant = copy.deepcopy(cfg)
    variant["env"][_SWEPT_KEY][_SWEPT_GRID] = radius
    return variant


def sensitivity_records(  # noqa: PLR0913 — cfg + swept value + run keys + history are distinct
    cfg: dict, radius: int, algorithm: str, seed: int, stage_idx: int, history: list[dict]
) -> list[dict]:
    """Tag one training history with the swept param/value (final-round capture only)."""
    return [
        {
            "param": f"view_radius_{_SWEPT_GRID}",
            "value": int(radius),
            "algorithm": algorithm,
            "seed": int(seed),
            "stage": int(stage_idx),
            "grid": stage_params(cfg, stage_idx)[0],
            "final_capture_rate": float(history[-1]["capture_rate"]),
        }
    ]


def run_sensitivity(  # noqa: PLR0913 — factory + cfg + swept values + seeds + stage + out + arm
    sdk_factory: object,
    cfg: dict,
    values: list[int],
    seeds: list[int],
    stage_idx: int,
    out_path: str | Path,
    algorithm: str = "qmix",
) -> list[dict]:
    """Train one variant per swept radius x seed (everything else fixed); append + return records."""
    records = []
    for radius in values:
        variant = make_variant(cfg, radius)
        sdk = sdk_factory(variant)
        for seed in seeds:
            recs = sensitivity_records(
                variant, radius, algorithm, seed, stage_idx, sdk.train(algorithm, seed, stage_idx)
            )
            append_records(out_path, recs)
            records += recs
    return records


def aggregate_sensitivity(records: list[dict]) -> dict[int, tuple[float, float]]:
    """Cross-seed mean±SE of the final capture rate per swept value (for the figure)."""
    by_value: dict[int, list[float]] = defaultdict(list)
    for rec in records:
        by_value[rec["value"]].append(rec["final_capture_rate"])
    out = {}
    for value, caps in by_value.items():
        se = statistics.stdev(caps) / len(caps) ** 0.5 if len(caps) > 1 else 0.0
        out[value] = (statistics.fmean(caps), se)
    return out
