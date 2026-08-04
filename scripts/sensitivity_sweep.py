"""§9 sensitivity sweep runner (T10.6; governed) — the 4x4 view radius ∈ {1, 2}.

Sweeps ``env.view_radius_by_grid[4]`` ONLY (everything else pinned) at the 4x4 focus stage
and writes ``results/runs/sensitivity_view_radius.jsonl`` + the figure. (The 4x4 stage is
used because 5x5 training is too slow to sweep.) Run AFTER ``run_results.py`` — never two
training processes at once (the compute-governance rule):
``uv run python scripts/sensitivity_sweep.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.results.plots import plot_sensitivity
from src.results.sensitivity import aggregate_sensitivity, run_sensitivity
from src.sdk.sdk import MarlSDK
from src.utils.config_loader import load_config

_STAGE_4X4 = 2
_VALUES = [1, 2]


def main(cfg: dict | None = None, argv: list[str] | None = None) -> Path:
    """Sweep the 4x4 view radius across a bounded seed set; write the JSONL + figure."""
    # argparse alone gives --help; argv=None means NOT a CLI call, so an in-process
    # main() never parses the caller's sys.argv. Without the parser this script
    # documented `--help` silently started the real job — one of these ran 10
    # minutes before being killed. Same class as the GUI entrypoint hang.
    argparse.ArgumentParser(
        description="Run the §9 view-radius sensitivity sweep and render its figure"
    ).parse_args(argv or [])
    cfg = cfg or load_config()
    seeds = [int(s) for s in cfg["training"]["seeds"][:3]]  # 3 seeds give SE bars; compute-bounded
    out = Path(cfg["paths"]["runs_dir"]) / "sensitivity_view_radius.jsonl"
    # TRUNCATE first: run_sensitivity appends, so re-running doubled the tracked file every
    # time and left `git status` dirty after a command the README calls idempotent. The
    # figure never drifted because aggregate_sensitivity de-dupes on
    # (param, value, algorithm, seed, stage) keeping the last — which is precisely why this
    # went unnoticed. The docstring always said "writes".
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("", encoding="utf-8")
    records = run_sensitivity(MarlSDK, cfg, _VALUES, seeds, _STAGE_4X4, out)
    figure = Path(cfg["paths"]["figures_dir"]) / "sensitivity_view_radius.png"
    plot_sensitivity(
        aggregate_sensitivity(records),
        "4x4 execution view radius",
        "§9 sensitivity — capture rate vs view radius",
        figure,
    )
    print(f"[sensitivity] {len(records)} runs (radius {_VALUES} x {len(seeds)} seeds) -> {figure}")
    return figure


if __name__ == "__main__":
    main(argv=sys.argv[1:])
