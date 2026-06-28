"""Matplotlib figure builders for F1/F2/F5/F6 (T10.2; Agg headless).

Each function reads the aggregated run records and writes ONE PNG. Styling literals
(dpi / fontsize / alpha / figsize) are LOCAL to this rendering module (CLAUDE.md §4). F1
and F2 are per-round cross-seed mean±SE curves with a shaded SE band per algorithm; F5 is
a final-capture bar with SE whiskers; F6 is capture rate vs grid size. All regenerate
deterministically from ``results/runs/`` via :mod:`src.results.make_figures`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend — must precede pyplot import

import matplotlib.pyplot as plt

from src.results.aggregate import curve, final_by_algorithm, final_by_grid

_DPI = 150
_FIGSIZE = (7.0, 4.5)
_BAND_ALPHA = 0.2


def _algorithms(records: list[dict]) -> list[str]:
    """Sorted distinct algorithm names present in the records."""
    return sorted({rec["algorithm"] for rec in records})


def _save(fig: object, out_path: str | Path) -> Path:
    """Tight-layout, save at ``_DPI``, close, and return the path (parent dirs created)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    return out_path


def plot_curve_figure(  # noqa: PLR0913 — records + metric + stage + 2 labels + path are distinct
    records: list[dict], metric: str, stage: int, title: str, ylabel: str, out_path: str | Path
) -> Path:
    """F1/F2: per-round cross-seed mean±SE of ``metric``, one line + SE band per algorithm."""
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for algorithm in _algorithms(records):
        rounds, mean, se = curve(records, metric, algorithm, stage)
        if not rounds:
            continue
        lower = [m - s for m, s in zip(mean, se, strict=True)]
        upper = [m + s for m, s in zip(mean, se, strict=True)]
        (line,) = ax.plot(rounds, mean, label=algorithm.upper())
        ax.fill_between(rounds, lower, upper, alpha=_BAND_ALPHA, color=line.get_color())
    ax.set_xlabel("self-play round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return _save(fig, out_path)


def plot_comparison(records: list[dict], metric: str, stage: int, out_path: str | Path) -> Path:
    """F5: final-rounds capture rate per algorithm (bar + SE whisker)."""
    stats = final_by_algorithm(records, metric, stage)
    algos = list(stats)
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.bar(
        [a.upper() for a in algos],
        [stats[a][0] for a in algos],
        yerr=[stats[a][1] for a in algos],
        capsize=6,
    )
    ax.set_ylabel("final capture rate")
    ax.set_title(f"IQL vs VDN vs QMIX — final capture rate (stage {stage})")
    return _save(fig, out_path)


def plot_scaling(records: list[dict], metric: str, out_path: str | Path) -> Path:
    """F6: final capture rate vs grid size, one line per algorithm."""
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for algorithm in _algorithms(records):
        by_grid = final_by_grid(records, metric, algorithm)
        grids = sorted(by_grid)
        ax.errorbar(
            grids,
            [by_grid[g][0] for g in grids],
            yerr=[by_grid[g][1] for g in grids],
            marker="o",
            capsize=4,
            label=algorithm.upper(),
        )
    ax.set_xlabel("grid size (cells per side)")
    ax.set_ylabel("final capture rate")
    ax.set_title("Scale effect — capture rate vs grid size")
    ax.legend()
    return _save(fig, out_path)


def plot_sensitivity(stats: dict, xlabel: str, title: str, out_path: str | Path) -> Path:
    """§9 sensitivity: capture rate vs a swept parameter value (mean±SE), one line."""
    values = sorted(stats)
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.errorbar(
        values,
        [stats[v][0] for v in values],
        yerr=[stats[v][1] for v in values],
        marker="o",
        capsize=5,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("final capture rate")
    ax.set_title(title)
    return _save(fig, out_path)


def plot_minimax_q(history: list[dict], out_path: str | Path, escape_floor: float | None = None) -> Path:
    """F7 (P-bonus, L11 §5): tabular Minimax-Q convergence on the 3x3 zero-sum pursuit.

    Plots the rolling cop capture rate and the certified game value at a fixed reference start
    state vs episode. ``escape_floor`` (= -gamma**(H-1), the discounted value of a guaranteed
    escape at the horizon) is the lower bound the game value converges *down* toward: a lone cop
    cannot corner an equal-speed evader, so the minimax value settles near the escape floor.
    """
    episodes = [row["episode"] for row in history]
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(episodes, [row["capture_rate"] for row in history], marker="o", label="cop capture rate")
    ax.plot(episodes, [row["ref_value"] for row in history], marker="s", label="game value (ref state)")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    if escape_floor is not None:
        ax.axhline(
            escape_floor,
            color="crimson",
            linewidth=1.0,
            linestyle="--",
            label=f"escape floor $-\\gamma^{{H-1}}$ = {escape_floor:.3f}",
        )
    ax.set_xlabel("training episode")
    ax.set_ylabel("capture rate / game value")
    ax.set_title("Minimax-Q convergence — 3x3 cop-vs-thief (zero-sum equilibrium)")
    ax.legend()
    return _save(fig, out_path)
