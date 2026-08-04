"""Distribution + matrix figure builders (V3 §9.3 chart variety; Agg headless).

:mod:`src.results.plots` covers the LINE (F1/F2/F6) and BAR (F5) families; this module adds
the two remaining families the guidelines ask for — a BOX plot of the per-seed spread and a
HEATMAP of the algorithm x curriculum-stage matrix. Both read the SAME aggregated record
shape as ``plots.py`` (one dict per algorithm/seed/stage/round) and share its conventions.
Kept separate purely to respect the 150-LOC file cap — which is exactly why the shared
styling and I/O now live in :mod:`src.results._plot_io` rather than being copied here: a
split made for file size, not for meaning, must not fork the two modules' rendering.
"""

from __future__ import annotations

import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend — must precede pyplot import

import matplotlib.pyplot as plt

from src.results._plot_io import FIGSIZE, algorithms, save_figure
from src.results.aggregate import final_values_by_seed

_CMAP = "viridis"
_ANNOT_FONTSIZE = 9
_SEED_COLOR = "#1f77b4"
_MEAN_PROPS = {"marker": "D", "markerfacecolor": "crimson", "markeredgecolor": "crimson", "markersize": 5}


def plot_final_distribution(
    records: list[dict], metric: str, stage: int, out_path: str | Path, last_k: int
) -> Path:
    """BOX plot of the per-seed final capture rate — one box per algorithm at ``stage``.

    WHY a box plot: the F5 bar chart collapses the whole seed population into mean±SE, which
    hides whether an algorithm is reliably good or merely lucky on one seed. A box shows the
    median, the inter-quartile range and the outliers, so seed-to-seed VARIANCE — the thing
    that separates a stable CTDE learner from a high-variance independent one — is visible.
    Individual seed points are overlaid on each box so small-N spread is not misread.
    """
    algos = [a for a in algorithms(records) if final_values_by_seed(records, metric, a, stage, last_k)]
    data = [final_values_by_seed(records, metric, a, stage, last_k) for a in algos]
    grid = next(rec["grid"] for rec in records if rec["stage"] == stage)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.boxplot(data, tick_labels=[a.upper() for a in algos], showmeans=True, meanprops=_MEAN_PROPS)
    for position, values in enumerate(data, start=1):
        ax.scatter([position] * len(values), values, color=_SEED_COLOR, alpha=0.6, zorder=3)
    ax.scatter([], [], color=_SEED_COLOR, alpha=0.6, label="individual seed")
    ax.plot([], [], linestyle="none", label="mean across seeds", **_MEAN_PROPS)
    ax.set_xlabel("algorithm")
    ax.set_ylabel("final capture rate (per-seed mean of last rounds)")
    ax.set_title(f"Seed-to-seed spread of final capture rate ({grid}x{grid} board)")
    ax.legend(loc="best")
    return save_figure(fig, out_path)


def _heatmap_matrix(
    records: list[dict], metric: str, algos: list[str], stages: list[int], last_k: int
) -> list[list[float]]:
    """Mean final ``metric`` per (algorithm, stage); missing combinations become ``nan``."""
    return [
        [
            statistics.fmean(values)
            if (values := final_values_by_seed(records, metric, a, s, last_k))
            else float("nan")
            for s in stages
        ]
        for a in algos
    ]


def plot_capture_heatmap(records: list[dict], metric: str, out_path: str | Path, last_k: int) -> Path:
    """HEATMAP of mean final capture rate: algorithms (rows) x curriculum stage / grid (columns).

    WHY a heatmap: the result is a two-factor matrix (which algorithm x how big the board),
    and a line or bar chart can only foreground one factor at a time. A colour matrix shows
    the whole grid at once, so the reader sees both the per-algorithm ranking and the decay
    with board size in one glance. Each cell is annotated with its numeric value so the
    figure stays readable in greyscale, and a colorbar gives the value scale.
    """
    algos = algorithms(records)
    stages = sorted({rec["stage"] for rec in records})
    grids = {s: next(rec["grid"] for rec in records if rec["stage"] == s) for s in stages}
    matrix = _heatmap_matrix(records, metric, algos, stages, last_k)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    image = ax.imshow(matrix, cmap=_CMAP, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(stages)), [f"stage {s}\n{grids[s]}x{grids[s]}" for s in stages])
    ax.set_yticks(range(len(algos)), [a.upper() for a in algos])
    for row, values in enumerate(matrix):
        for col, value in enumerate(values):
            label = "n/a" if math.isnan(value) else f"{value:.2f}"
            ax.text(col, row, label, ha="center", va="center", color="w", fontsize=_ANNOT_FONTSIZE)
    ax.set_xlabel("curriculum stage (board size)")
    ax.set_ylabel("algorithm")
    ax.set_title("Mean final capture rate — algorithm x curriculum stage")
    fig.colorbar(image, ax=ax, label="final capture rate")
    return save_figure(fig, out_path)
