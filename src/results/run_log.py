"""Per-round training logger → ``results/runs/*.jsonl`` (T10.1 — the figure data source).

Runs ``sdk.train(algorithm, seed, stage)`` and appends ONE JSON line PER ROUND (the run
keys ``algorithm`` / ``seed`` / ``stage`` / ``grid`` + the round metrics ``round`` /
``role`` / ``loss`` / ``capture_rate``) to the append-only log ``make_figures`` reads for
F1 (learning curve), F2 (loss), F5 (algo comparison), and F6 (scaling). ``done_runs``
makes the matrix RESUMABLE — an already-logged ``(algorithm, seed, stage)`` is skipped.
Routes through the SDK only (serial + thread-capped, so a full run cannot freeze the host).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.services.finetune import stage_params


def history_records(cfg: dict, algorithm: str, seed: int, stage_idx: int, history: list[dict]) -> list[dict]:
    """Expand one self-play history into per-round JSONL records (run keys + metrics)."""
    grid = stage_params(cfg, stage_idx)[0]
    return [
        {
            "algorithm": algorithm,
            "seed": int(seed),
            "stage": int(stage_idx),
            "grid": grid,
            "round": int(entry["round"]),
            "role": entry["role"],
            "loss": float(entry["loss"]),
            "capture_rate": float(entry["capture_rate"]),
        }
        for entry in history
    ]


def append_records(path: str | Path, records: list[dict]) -> None:
    """Append each record as one JSON line (append-only; parent dirs created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def done_runs(path: str | Path) -> set[tuple[str, int, int]]:
    """Return the ``(algorithm, seed, stage)`` combos already in the log (resume support)."""
    path = Path(path)
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            done.add((rec["algorithm"], int(rec["seed"]), int(rec["stage"])))
    return done


def run_and_log(  # noqa: PLR0913 — sdk + cfg + the 3 matrix axes + out path are all distinct
    sdk: object, cfg: dict, algorithm: str, seed: int, stage_idx: int, out_path: str | Path
) -> list[dict]:
    """Train one ``(algorithm, seed, stage)`` and append its per-round records; return them."""
    history = sdk.train(algorithm, seed, stage_idx)
    records = history_records(cfg, algorithm, seed, stage_idx, history)
    append_records(out_path, records)
    return records
