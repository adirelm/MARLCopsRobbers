"""Aggregate ``results/runs/*.jsonl`` → per-method mean±SE curves + final stats (T10.1).

Pure reduction over the append-only run log: group the per-round records and compute the
cross-seed mean±SE of a metric per round (F1 capture curve, F2 loss curve), the
final-rounds mean±SE per method at a stage (F5 comparison), and per curriculum stage
(F6 — the stages vary board size AND team size together, not board size alone).
The round schedule is deterministic, so round ``r`` is the SAME role across
every seed — the cross-seed mean at a round is clean. The independent unit for EVERY
final statistic is the SEED: each seed is first reduced to one number (its last-``k``-round
mean) and the SE is taken across seeds — pooling the raw rounds would treat autocorrelated
within-seed values as independent and understate the SE. Stdlib ``statistics`` only.
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path

_LOG = logging.getLogger(__name__)
# The semantic identity of one logged round (role is round-determined by the schedule).
_RUN_KEY = ("algorithm", "seed", "stage", "round")


def load_runs(path: str | Path) -> list[dict]:
    """Load every JSON line from the run log (missing/empty → ``[]``), de-duplicated.

    Records are de-duplicated by the SEMANTIC run key ``(algorithm, seed, stage,
    round)`` keeping the LAST record (a crashed-then-resumed combo re-appends its
    full history, so the last append is authoritative) and WARNING when the dropped
    duplicate carried different values — a silent double append would double-weight
    that seed in every curve and narrow every SE band (codex W2 R1). Records without
    the full run key fall back to byte-identity de-dup (order-preserving).
    """
    path = Path(path)
    if not path.exists():
        return []
    records: dict[object, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        key: object = tuple(rec[k] for k in _RUN_KEY) if all(k in rec for k in _RUN_KEY) else line
        previous = records.get(key)
        if previous is not None and previous != rec:
            _LOG.warning("%s: conflicting duplicate for run key %s — keeping the LAST record", path.name, key)
        records[key] = rec
    return list(records.values())


def _mean_se(values: list[float]) -> tuple[float, float]:
    """Return ``(mean, standard error)``; SE is 0 for a single sample."""
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
    return mean, se


def curve(
    records: list[dict], metric: str, algorithm: str, stage: int, role: str | None = None
) -> tuple[list[int], list[float], list[float]]:
    """Per-round cross-seed mean±SE of ``metric`` for one ``(algorithm, stage)``.

    ``role`` filters to one agent's training rounds (self-play alternates cop/thief per
    round); ``None`` pools both — pooling ``loss`` would interleave two different nets'
    losses, so the per-agent figures (F1/F2) pass an explicit role.
    """
    by_round: dict[int, list[float]] = defaultdict(list)
    for rec in records:
        if rec["algorithm"] == algorithm and rec["stage"] == stage and role in (None, rec["role"]):
            by_round[rec["round"]].append(rec[metric])
    rounds = sorted(by_round)
    stats = [_mean_se(by_round[rd]) for rd in rounds]
    return rounds, [m for m, _ in stats], [s for _, s in stats]


def final_values_by_seed(
    records: list[dict], metric: str, algorithm: str, stage: int, last_k: int
) -> list[float]:
    """ONE final value per seed: that seed's mean of ``metric`` over its last ``last_k`` rounds.

    The seed is the independent replication unit — successive rounds of one seed are
    autocorrelated, so every final mean±SE is computed OVER these per-seed values,
    never over the pooled rounds (which would shrink the SE by ~``sqrt(last_k)``).

    ``last_k`` is REQUIRED (``results.final_window_rounds``). It used to default to 5, and
    since no caller ever passed it, that literal silently set the averaging window behind
    every published mean±SE — an analysis choice, not plot styling, so it belongs in config.
    """
    by_seed: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for rec in records:
        if rec["algorithm"] == algorithm and rec["stage"] == stage:
            by_seed[rec["seed"]].append((rec["round"], rec[metric]))
    values: list[float] = []
    for rounds in by_seed.values():
        rounds.sort()
        values.append(statistics.fmean(v for _, v in rounds[-last_k:]))
    return values


def final_by_algorithm(records: list[dict], metric: str, stage: int, last_k: int) -> dict:
    """Final cross-SEED mean±SE per algorithm at one stage (F5 comparison)."""
    algos = sorted({r["algorithm"] for r in records if r["stage"] == stage})
    out = {
        a: _mean_se(vals) for a in algos if (vals := final_values_by_seed(records, metric, a, stage, last_k))
    }
    return out


def final_by_grid(records: list[dict], metric: str, algorithm: str, last_k: int) -> dict:
    """Final cross-SEED mean±SE per CURRICULUM-STAGE grid for one algorithm (F6).

    Keys are the stage grid sizes, but the curriculum stages vary board size AND
    team size together (the 4x4 stage trains a 2-cop team, codex W2 R3) — this is
    a curriculum-stage curve, NOT an isolated board-size scaling experiment.
    """
    stages = sorted({r["stage"] for r in records if r["algorithm"] == algorithm})
    out = {}
    for stage in stages:
        grid = next(r["grid"] for r in records if r["algorithm"] == algorithm and r["stage"] == stage)
        # a stage in `stages` came from this algorithm's records, so final_values_by_seed is non-empty
        out[grid] = _mean_se(final_values_by_seed(records, metric, algorithm, stage, last_k))
    return out
