"""Thin IQL-baseline training entry — routes through the SDK (T4.8; slow, run manually).

The §7.2 non-stationarity baseline (independent per-agent Double-DQN, no mixer).
``uv run python scripts/train_iql_baseline.py``. Thin wrapper over the SDK (§4).
"""

from __future__ import annotations

import argparse
import sys

from src.sdk.sdk import MarlSDK
from src.utils.config_loader import load_config


def main(cfg: dict | None = None, stage_idx: int = 0, argv: list[str] | None = None) -> list[dict]:
    """Train the IQL baseline via self-play at one curriculum stage; print a summary."""
    # argparse alone gives --help and rejects unknown flags. argv=None means NOT a
    # CLI call, so an in-process main() never parses the caller's sys.argv.
    # Without this the parser read pytest's argv and every such test died.
    # script IGNORED argv, so a documented `--help` started the real job.
    parser = argparse.ArgumentParser(description="Train the IQL baseline (long)")
    parser.parse_args(argv or [])
    cfg = cfg or load_config()
    seed = int(cfg["training"]["seeds"][0])
    history = MarlSDK(cfg).train("iql", seed, stage_idx)
    cap = history[-1]["capture_rate"]
    print(f"[train_iql] iql seed={seed} stage={stage_idx} rounds={len(history)} capture={cap:.3f}")
    return history


if __name__ == "__main__":
    main(argv=sys.argv[1:])
