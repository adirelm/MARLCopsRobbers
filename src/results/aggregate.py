"""Aggregate ``results/runs/*.jsonl`` → per-method mean±SE curves + final stats (T10.1).

Pure reduction over the append-only run log: group the per-round records and compute the
cross-seed mean±SE of a metric per round (F1 capture curve, F2 loss curve), the
final-rounds mean±SE per method at a stage (F5 comparison), and per grid size (F6
scaling). The round schedule is deterministic, so round ``r`` is the SAME role across
every seed — the cross-seed mean at a round is clean. Stdlib ``statistics`` only.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_runs(path: str | Path) -> list[dict]:
    """Load every JSON line from the run log (missing/empty → ``[]``)."""
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _mean_se(values: list[float]) -> tuple[float, float]:
    """Return ``(mean, standard error)``; SE is 0 for a single sample."""
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
    return mean, se


def curve(
    records: list[dict], metric: str, algorithm: str, stage: int
) -> tuple[list[int], list[float], list[float]]:
    """Per-round cross-seed mean±SE of ``metric`` for one ``(algorithm, stage)``."""
    by_round: dict[int, list[float]] = defaultdict(list)
    for rec in records:
        if rec["algorithm"] == algorithm and rec["stage"] == stage:
            by_round[rec["round"]].append(rec[metric])
    rounds = sorted(by_round)
    stats = [_mean_se(by_round[rd]) for rd in rounds]
    return rounds, [m for m, _ in stats], [s for _, s in stats]


def _final_values(records: list[dict], metric: str, algorithm: str, stage: int, last_k: int) -> list[float]:
    """The ``metric`` over the last ``last_k`` rounds of each seed for ``(algorithm, stage)``."""
    by_seed: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for rec in records:
        if rec["algorithm"] == algorithm and rec["stage"] == stage:
            by_seed[rec["seed"]].append((rec["round"], rec[metric]))
    values: list[float] = []
    for rounds in by_seed.values():
        rounds.sort()
        values += [v for _, v in rounds[-last_k:]]
    return values


def final_by_algorithm(records: list[dict], metric: str, stage: int, last_k: int = 5) -> dict:
    """Final-rounds cross-seed mean±SE per algorithm at one stage (F5 comparison)."""
    algos = sorted({r["algorithm"] for r in records if r["stage"] == stage})
    out = {a: _mean_se(vals) for a in algos if (vals := _final_values(records, metric, a, stage, last_k))}
    return out


def final_by_grid(records: list[dict], metric: str, algorithm: str, last_k: int = 5) -> dict:
    """Final-rounds mean±SE per grid size for one algorithm (F6 scaling)."""
    stages = sorted({r["stage"] for r in records if r["algorithm"] == algorithm})
    out = {}
    for stage in stages:
        grid = next(r["grid"] for r in records if r["algorithm"] == algorithm and r["stage"] == stage)
        # a stage in `stages` came from this algorithm's records, so _final_values is non-empty
        out[grid] = _mean_se(_final_values(records, metric, algorithm, stage, last_k))
    return out
