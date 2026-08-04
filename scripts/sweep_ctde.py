"""Thin ablation-sweep entry — qmix/vdn/iql x training.seeds -> runs JSONL (T4.7).

Calls ``MarlSDK.run_ablation_sweep`` (ADR-0002: launchers import only the SDK); the sweep
runs SERIALLY in the service layer, never N full-core training processes at once, and
appends one reproducible record per run for the §7.3/§9 analysis. Slow; run manually:
``uv run python scripts/sweep_ctde.py``.
"""

from __future__ import annotations

import argparse
import sys

from src.sdk import MarlSDK
from src.utils.config_loader import load_config

_ALGORITHMS = ["qmix", "vdn", "iql"]


def main(cfg: dict | None = None, stage_idx: int = 0, argv: list[str] | None = None) -> list[dict]:
    """Sweep the three algorithm arms across all seeds; append run records to JSONL."""
    # argparse alone gives --help and rejects unknown flags. argv=None means NOT a
    # CLI call, so an in-process main() never parses the caller's sys.argv.
    # Without this the parser read pytest's argv and every such test died.
    # script IGNORED argv, so a documented `--help` started the real job.
    parser = argparse.ArgumentParser(description="Run the IQL/VDN/QMIX ablation sweep (long)")
    parser.parse_args(argv or [])
    cfg = cfg or load_config()
    return MarlSDK(cfg).run_ablation_sweep(_ALGORITHMS, stage_idx)


if __name__ == "__main__":
    main(argv=sys.argv[1:])
