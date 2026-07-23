"""Per-round training logger → ``results/runs/*.jsonl`` (T10.1 — the figure data source).

Runs ``sdk.train(algorithm, seed, stage)`` and appends ONE JSON line PER ROUND (the run
keys ``algorithm`` / ``seed`` / ``stage`` / ``grid`` + the round metrics ``round`` /
``role`` / ``loss`` / ``capture_rate``) to the append-only log ``make_figures`` reads for
F1 (learning curve), F2 (loss), F5 (algo comparison), and F6 (curriculum stages). ``done_runs``
makes the matrix RESUMABLE — a ``(algorithm, seed, stage)`` with ALL its rounds logged is skipped.
Routes through the SDK only (serial + thread-capped, so a full run cannot freeze the host).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.services.finetune import stage_params


def history_records(cfg: dict, algorithm: str, seed: int, stage_idx: int, history: list[dict]) -> list[dict]:
    """Expand one self-play history into per-round JSONL records (run keys + metrics).

    The §7.3(a) return columns (``cop_return`` / ``thief_return``) are OPTIONAL-ABSENT:
    a legacy history without them logs records WITHOUT the keys so the return curves
    EXCLUDE those rounds — never a fabricated flat ``0.0`` (codex W2 R2).
    """
    grid = stage_params(cfg, stage_idx)[0]
    records = []
    for entry in history:
        record = {
            "algorithm": algorithm,
            "seed": int(seed),
            "stage": int(stage_idx),
            "grid": grid,
            "round": int(entry["round"]),
            "role": entry["role"],
            "loss": float(entry["loss"]),
            "capture_rate": float(entry["capture_rate"]),
        }
        record.update({key: float(entry[key]) for key in ("cop_return", "thief_return") if key in entry})
        records.append(record)
    return records


def append_records(path: str | Path, records: list[dict]) -> None:
    """Append each record as one JSON line (append-only; parent dirs created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def done_runs(path: str | Path, required_rounds: int) -> set[tuple[str, int, int]]:
    """Return the ``(algorithm, seed, stage)`` combos COMPLETE in the log (resume support).

    A combo counts as done only when it has at least ``required_rounds`` DISTINCT
    logged rounds (pass ``selfplay.rounds``): a run that crashed mid-append would
    otherwise be declared complete after ONE round and its remaining rounds skipped
    forever (codex W2 R1). Re-running a partial combo re-appends the full history;
    ``aggregate.load_runs`` keeps the LAST record per round key.
    """
    path = Path(path)
    if not path.exists():
        return set()
    rounds_seen: dict[tuple[str, int, int], set[int]] = defaultdict(set)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            rounds_seen[(rec["algorithm"], int(rec["seed"]), int(rec["stage"]))].add(int(rec["round"]))
    return {combo for combo, rounds in rounds_seen.items() if len(rounds) >= int(required_rounds)}


def run_and_log(  # noqa: PLR0913 — sdk + cfg + the 3 matrix axes + out path are all distinct
    sdk: object, cfg: dict, algorithm: str, seed: int, stage_idx: int, out_path: str | Path
) -> list[dict]:
    """Train one ``(algorithm, seed, stage)`` and append its per-round records; return them."""
    history = sdk.train(algorithm, seed, stage_idx)
    records = history_records(cfg, algorithm, seed, stage_idx, history)
    append_records(out_path, records)
    return records
