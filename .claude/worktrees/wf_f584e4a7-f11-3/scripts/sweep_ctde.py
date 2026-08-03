"""Thin ablation-sweep entry — qmix/vdn/iql x training.seeds -> runs JSONL (T4.7).

Calls ``MarlSDK.run_ablation_sweep`` (ADR-0002: launchers import only the SDK); the sweep
runs SERIALLY in the service layer, never N full-core training processes at once, and
appends one reproducible record per run for the §7.3/§9 analysis. Slow; run manually:
``uv run python scripts/sweep_ctde.py``.
"""

from __future__ import annotations

from src.sdk import MarlSDK
from src.utils.config_loader import load_config

_ALGORITHMS = ["qmix", "vdn", "iql"]


def main(cfg: dict | None = None, stage_idx: int = 0) -> list[dict]:
    """Sweep the three algorithm arms across all seeds; append run records to JSONL."""
    cfg = cfg or load_config()
    return MarlSDK(cfg).run_ablation_sweep(_ALGORITHMS, stage_idx)


if __name__ == "__main__":
    main()
