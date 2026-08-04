"""Thin CTDE QMIX training entry — routes through the SDK (T4.8; slow, run manually).

``uv run python scripts/train_ctde.py``. All business logic lives in the SDK /
services; this is a thin wrapper (the §4 single-entry rule). Compute thread caps
are applied inside the trainer, so a full run cannot freeze the host.
"""

from __future__ import annotations

import argparse
import sys

from src.sdk.sdk import MarlSDK
from src.utils.config_loader import load_config


def main(cfg: dict | None = None, stage_idx: int = 0, argv: list[str] | None = None) -> list[dict]:
    """Train the QMIX cop team via self-play at one curriculum stage; print a summary."""
    # argparse alone gives --help and rejects unknown flags. argv=None means NOT a
    # CLI call, so an in-process main() never parses the caller's sys.argv.
    # Without this the parser read pytest's argv and every such test died.
    # script IGNORED argv, so a documented `--help` started the real job.
    parser = argparse.ArgumentParser(description="Train the CTDE learner (long)")
    parser.parse_args(argv or [])
    cfg = cfg or load_config()
    seed = int(cfg["training"]["seeds"][0])
    history = MarlSDK(cfg).train("qmix", seed, stage_idx)
    cap = history[-1]["capture_rate"]
    print(f"[train_ctde] qmix seed={seed} stage={stage_idx} rounds={len(history)} capture={cap:.3f}")
    return history


if __name__ == "__main__":
    main(argv=sys.argv[1:])
