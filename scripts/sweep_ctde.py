"""Thin ablation-sweep entry — qmix/vdn/iql x training.seeds -> runs JSONL (T4.7).

Routes through the SDK + :func:`src.services.sweep.run_sweep` (serial; never N
full-core training processes at once). Appends one reproducible record per run to
``paths.runs_dir/ablation.jsonl`` for the §7.3/§9 analysis. Slow; run manually:
``uv run python scripts/sweep_ctde.py``.
"""

from __future__ import annotations

from pathlib import Path

from src.sdk.sdk import MarlSDK
from src.services.sweep import run_sweep
from src.utils.config_loader import load_config

_ALGORITHMS = ["qmix", "vdn", "iql"]


def main(cfg: dict | None = None, stage_idx: int = 0) -> list[dict]:
    """Sweep the three algorithm arms across all seeds; append run records to JSONL."""
    cfg = cfg or load_config()
    out = Path(cfg["paths"]["runs_dir"]) / "ablation.jsonl"
    seeds = list(cfg["training"]["seeds"])
    records = run_sweep(MarlSDK(cfg), cfg, _ALGORITHMS, seeds, stage_idx, out)
    print(f"[sweep] {len(records)} runs ({_ALGORITHMS} x {len(seeds)} seeds) -> {out}")
    return records


if __name__ == "__main__":
    main()
