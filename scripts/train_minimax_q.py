"""Thin Minimax-Q baseline entry — routes through the SDK (P-bonus, L11 §5; slow, manual).

The tabular ZERO-SUM equilibrium learner for the 1-cop-vs-thief pursuit, an empirical
contrast to the deep self-play arms (README §7.2). ``uv run python scripts/train_minimax_q.py``.
Thin wrapper over the SDK (§4); all tunables come from ``config.yaml`` ``minimax_q.*``.
"""

from __future__ import annotations

from src.sdk.sdk import MarlSDK
from src.utils.config_loader import load_config


def main(cfg: dict | None = None, seed: int | None = None) -> list[dict]:
    """Train tabular Minimax-Q via the SDK and print the final-window summary."""
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
    main()
