"""§9 sensitivity sweep runner (T10.6; slow, governed) — view_radius_5 ∈ {1, 2}.

Sweeps ``env.view_radius_by_grid[5]`` ONLY (everything else pinned) at the 5x5 stage and
writes ``results/runs/sensitivity_view_radius.jsonl`` + the figure. Run AFTER
``run_results.py`` — never two training processes at once (the compute-governance rule):
``uv run python scripts/sensitivity_sweep.py``.
"""

from __future__ import annotations

from pathlib import Path

from src.results.plots import plot_sensitivity
from src.results.sensitivity import aggregate_sensitivity, run_sensitivity
from src.sdk.sdk import MarlSDK
from src.utils.config_loader import load_config

_STAGE_5X5 = 3
_VALUES = [1, 2]


def main(cfg: dict | None = None) -> Path:
    """Sweep the 5x5 view radius across a bounded seed set; write the JSONL + figure."""
    cfg = cfg or load_config()
    seeds = [int(s) for s in cfg["training"]["seeds"][:3]]  # 3 seeds give SE bars; compute-bounded
    out = Path(cfg["paths"]["runs_dir"]) / "sensitivity_view_radius.jsonl"
    records = run_sensitivity(MarlSDK, cfg, _VALUES, seeds, _STAGE_5X5, out)
    figure = Path(cfg["paths"]["figures_dir"]) / "sensitivity_view_radius.png"
    plot_sensitivity(
        aggregate_sensitivity(records),
        "5x5 execution view radius",
        "§9 sensitivity — capture rate vs view radius",
        figure,
    )
    print(f"[sensitivity] {len(records)} runs (radius {_VALUES} x {len(seeds)} seeds) -> {figure}")
    return figure


if __name__ == "__main__":
    main()
