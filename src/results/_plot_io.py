"""Shared figure I/O + styling for the plot builders (the ONE definition of both).

:mod:`src.results.plots` and :mod:`src.results.plots_extra` were split purely to respect the
150-LOC file cap, not along a semantic boundary, so both carried byte-identical copies of
``save_figure``, ``algorithms``, ``DPI`` and ``FIGSIZE``. Duplicated STYLE is worse than
duplicated logic here: the two copies of ``DPI`` silently disagreed once already — F7 stayed
at 150 dpi while every sibling moved to 300, and regenerating it turned the test suite red.
One definition means that cannot recur.

Styling literals still live in the rendering layer rather than config (CLAUDE.md §4): they
are visual design, not tunable parameters. This module is where "the rendering layer" is.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend — must precede pyplot import

import matplotlib.pyplot as plt

DPI = 300  # V3 §9.3 "high resolution" — print-quality raster for EVERY figure
FIGSIZE = (7.0, 4.5)  # the single-axes default
WIDE_FIGSIZE = (11.0, 4.2)  # the two-panel (per-agent / per-net) layout


def algorithms(records: list[dict]) -> list[str]:
    """Sorted distinct algorithm names present in the records."""
    return sorted({rec["algorithm"] for rec in records})


def save_figure(fig: object, out_path: str | Path) -> Path:
    """Tight-layout, save at :data:`DPI`, close, and return the path (parent dirs created)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path
