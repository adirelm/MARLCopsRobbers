"""Thin Minimax-Q baseline entry — routes through the SDK (L11 §5 bonus; slow, manual).

The tabular ZERO-SUM equilibrium learner for the 1-cop-vs-thief pursuit, an empirical
contrast to the deep self-play arms (README §7.2). ``uv run python scripts/train_minimax_q.py``.
Thin wrapper over the SDK (§4); all tunables come from ``config.yaml`` ``minimax_q.*``.
"""

from __future__ import annotations

import argparse
import sys

from src.sdk.sdk import MarlSDK
from src.utils.config_loader import load_config


def main(cfg: dict | None = None, seed: int | None = None, argv: list[str] | None = None) -> list[dict]:
    """Train tabular Minimax-Q via the SDK and print the final-window summary."""
    # argparse alone gives --help and rejects unknown flags. argv=None means NOT a
    # CLI call, so an in-process main() never parses the caller's sys.argv.
    # Without this the parser read pytest's argv and every such test died.
    # script IGNORED argv, so a documented `--help` started the real job.
    parser = argparse.ArgumentParser(description="Train the tabular Minimax-Q baseline")
    parser.parse_args(argv or [])
    cfg = cfg or load_config()
    seed = int(cfg["training"]["seeds"][0]) if seed is None else seed
    history = MarlSDK(cfg).run_minimax_q_baseline(seed)
    last = history[-1]
    print(
        f"[minimax_q] seed={seed} episodes={last['episode']} "
        f"capture={last['capture_rate']:.3f} game_value={last['ref_value']:.3f}"
    )
    return history


if __name__ == "__main__":
    main(argv=sys.argv[1:])
