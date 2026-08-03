"""Stage-selection helpers for :mod:`src.results.make_figures` (kept separate for the LOC cap).

Three focus notions the figure set needs: the largest well-covered stage (``focus``, F1b/
heatmap), the largest stage present (``final`` = the §5.1 5x5 final test, F1/F2 learning +
loss curves), and the largest MULTI-cop stage (``comparison`` = the 4x4 2-cop rung where the
algorithms actually separate, F5/F8 — at 1-cop stages VDN's sum-decomposition reduces to IQL).
"""

from __future__ import annotations

from collections import defaultdict


def focus_stage(records: list[dict]) -> int:
    """The LARGEST stage with the most algorithm coverage (robust to a partial run)."""
    algos_by_stage: dict[int, set] = defaultdict(set)
    for rec in records:
        algos_by_stage[rec["stage"]].add(rec["algorithm"])
    most = max(len(algos) for algos in algos_by_stage.values())
    return max(stage for stage, algos in algos_by_stage.items() if len(algos) == most)


def final_stage(records: list[dict]) -> int:
    """The LARGEST curriculum stage present — the brief §5.1 5x5 'final test' for F1/F2."""
    return max(rec["stage"] for rec in records)


def comparison_stage(cfg: dict, records: list[dict]) -> int:
    """The 3-way F5/F8 comparison stage: the largest MULTI-cop stage present, else focus.

    At 1-cop stages VDN's sum-decomposition reduces exactly to IQL, so a 3-way comparison
    there is degenerate; the genuinely multi-agent stage (``num_cops>1``, the 4x4 rung) is
    where the algorithms separate.
    """
    by_stage = cfg["env"]["curriculum"]["num_cops_by_stage"]
    present = {rec["stage"] for rec in records}
    multi = [s for s in present if s < len(by_stage) and int(by_stage[s]) > 1]
    return max(multi) if multi else focus_stage(records)
