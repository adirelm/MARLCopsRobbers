"""Generate the F7 Minimax-Q convergence figure (P-bonus, L11 §5; slow, manual).

Runs the tabular Minimax-Q baseline through the SDK and renders the convergence curves to
``results/figures/minimax_q.png``. ``uv run python scripts/plot_minimax_q.py``. Slow (per-step
maximin LP); run manually like ``scripts/train_iql_baseline.py``. Deterministic for a fixed seed.
"""

from __future__ import annotations

from pathlib import Path

from src.results.plots import plot_minimax_q
from src.sdk.sdk import MarlSDK
from src.utils.config_loader import load_config


def main(cfg: dict | None = None, seed: int | None = None) -> Path:
    """Train Minimax-Q via the SDK, plot the convergence curves, return the figure path."""
    cfg = cfg or load_config()
    seed = int(cfg["training"]["seeds"][0]) if seed is None else seed
    history = MarlSDK(cfg).run_minimax_q_baseline(seed)
    mq = cfg["minimax_q"]
    # Lower bound on the start-state value: a sure escape pays -1 on the H-th transition,
    # discounted by gamma**(H-1) back to step 0 (a lone cop cannot beat this on an open grid).
    floor = -(float(mq["gamma"]) ** (int(mq["max_moves"]) - 1))
    out = Path(cfg["paths"]["figures_dir"]) / "minimax_q.png"
    path = plot_minimax_q(history, out, escape_floor=floor)
    print(f"[plot_minimax_q] seed={seed} -> {path} ({len(history)} windows)")
    return path


if __name__ == "__main__":
    main()
