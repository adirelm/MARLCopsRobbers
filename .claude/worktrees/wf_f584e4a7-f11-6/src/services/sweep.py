"""Seeded ablation sweep -> append-only JSONL (T4.7, §7.3 / §9 evidence).

Runs ``sdk.train`` for each ``(algorithm, seed)`` combination SERIALLY (never N
full-core training processes at once — the compute-governance contract) and
appends ONE JSON record per run to ``results/runs/*.jsonl``, the append-only log
the §9 analysis notebook consumes. Each record is fully reproducible from its
``(algorithm, seed, stage)``. The sweep routes through the SDK (it calls only
``sdk.train``), so no business logic leaks into the harness.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.services.finetune import stage_params


def sweep_record(cfg: dict, algorithm: str, seed: int, stage_idx: int, history: list[dict]) -> dict:
    """Build one JSONL run record from a self-play training history."""
    h, w, num_cops = stage_params(cfg, stage_idx)
    last = history[-1]
    return {
        "algorithm": algorithm,
        "seed": int(seed),
        "stage": int(stage_idx),
        "grid": [h, w],
        "num_cops": num_cops,
        "rounds": len(history),
        "final_loss": last["loss"],
        "final_capture_rate": last["capture_rate"],
    }


def append_jsonl(path: str | Path, record: dict) -> None:
    """Append ``record`` as one JSON line (append-only; parent dirs created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def run_sweep(  # noqa: PLR0913 — sdk + cfg + the 2 swept axes + stage + out path are distinct
    sdk: object,
    cfg: dict,
    algorithms: list[str],
    seeds: list[int],
    stage_idx: int,
    out_path: str | Path,
) -> list[dict]:
    """Run ``sdk.train`` over every ``(algorithm, seed)`` SERIALLY; log + return records.

    Args:
        sdk: An object exposing ``train(algorithm, seed, stage_idx) -> history``.
        cfg: Loaded config (drives the per-record stage grid).
        algorithms: The ablation arms to sweep (e.g. ``["qmix", "vdn", "iql"]``).
        seeds: The seeds to sweep per arm (``training.seeds``).
        stage_idx: Curriculum stage index trained for each run.
        out_path: The append-only JSONL log path.

    Returns:
        The list of run records (also appended, one JSON line each).
    """
    records = []
    for algorithm in algorithms:
        for seed in seeds:
            history = sdk.train(algorithm, seed, stage_idx)
            record = sweep_record(cfg, algorithm, seed, stage_idx, history)
            append_jsonl(out_path, record)
            records.append(record)
    return records
