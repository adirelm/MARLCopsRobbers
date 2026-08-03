"""Thin CTDE QMIX training entry — routes through the SDK (T4.8; slow, run manually).

``uv run python scripts/train_ctde.py``. All business logic lives in the SDK /
services; this is a thin wrapper (the §4 single-entry rule). Compute thread caps
are applied inside the trainer, so a full run cannot freeze the host.
"""

from __future__ import annotations

from src.sdk.sdk import MarlSDK
from src.utils.config_loader import load_config


def main(cfg: dict | None = None, stage_idx: int = 0) -> list[dict]:
    """Train the QMIX cop team via self-play at one curriculum stage; print a summary."""
    cfg = cfg or load_config()
    seed = int(cfg["training"]["seeds"][0])
    history = MarlSDK(cfg).train("qmix", seed, stage_idx)
    cap = history[-1]["capture_rate"]
    print(f"[train_ctde] qmix seed={seed} stage={stage_idx} rounds={len(history)} capture={cap:.3f}")
    return history


if __name__ == "__main__":
    main()
