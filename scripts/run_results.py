"""Full results matrix → ``results/runs/history.jsonl`` (T10.2; slow, governed, resumable).

Runs ``sdk.train`` for every ``(algorithm, seed, stage)`` in the config matrix SERIALLY
(thread-capped via the SDK — never N full-core processes, so the host never freezes) and
appends per-round records. RESUMABLE: a combo already in the log is skipped, so a run
interrupted by a Drive .git hiccup just resumes. Run: ``uv run python scripts/run_results.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.results.run_log import done_runs, run_and_log
from src.sdk.sdk import MarlSDK
from src.utils.config_loader import load_config

_ALGORITHMS = ["qmix", "vdn", "iql"]


def main(
    cfg: dict | None = None,
    algorithms: list[str] | None = None,
    stages: list[int] | None = None,
) -> Path:
    """Run the full (algorithm x seed x stage) matrix; append per-round records; return the path."""
    cfg = cfg or load_config()
    algorithms = algorithms or _ALGORITHMS
    stages = stages if stages is not None else list(range(len(cfg["env"]["curriculum"]["stages"])))
    seeds = [int(s) for s in cfg["training"]["seeds"]]
    out = Path(cfg["paths"]["runs_dir"]) / "history.jsonl"
    done = done_runs(out)
    sdk = MarlSDK(cfg)
    # STAGE-OUTER: a stage completes across all algorithms before the next (slower) stage,
    # so the F5 3-way comparison + F6 scaling are plottable from partial runs (5x5 trickles last).
    for stage in stages:
        for algorithm in algorithms:
            for seed in seeds:
                if (algorithm, seed, stage) in done:
                    print(f"[skip] {algorithm} seed={seed} stage={stage} (already logged)")
                    continue
                records = run_and_log(sdk, cfg, algorithm, seed, stage, out)
                cap = records[-1]["capture_rate"]
                print(f"[run] {algorithm} seed={seed} stage={stage} rounds={len(records)} capture={cap:.3f}")
    return out


if __name__ == "__main__":
    selected = [int(s) for s in sys.argv[1:]] or None  # CLI: `run_results.py 0 1 2`
    main(stages=selected)
