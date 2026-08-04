"""Full results matrix → ``results/runs/history.jsonl`` (T10.2; slow, governed, resumable).

Runs ``sdk.train`` for every ``(algorithm, seed, stage)`` in the config matrix SERIALLY
(thread-capped via the SDK — never N full-core processes, so the host never freezes) and
appends per-round records. RESUMABLE: a combo already in the log is skipped, so a run
interrupted by a Drive .git hiccup just resumes. Run: ``uv run python scripts/run_results.py``.
"""

from __future__ import annotations

import argparse
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
    argv: list[str] | None = None,
) -> Path:
    """Run the full (algorithm x seed x stage) matrix; append per-round records; return the path."""
    # argparse alone gives --help. Without it this script IGNORED argv and a documented
    # `--help` silently started the real job — one ran 10 minutes before being killed.
    # Same class as the GUI entrypoint hang, which this half of the sweep had missed.
    argparse.ArgumentParser(
        description="Run the full training matrix behind F1/F2/F5/F6 (takes HOURS)"
    ).parse_args(argv or [])
    cfg = cfg or load_config()
    algorithms = algorithms or _ALGORITHMS
    stages = stages if stages is not None else list(range(len(cfg["env"]["curriculum"]["stages"])))
    seeds = [int(s) for s in cfg["training"]["seeds"]]
    out = Path(cfg["paths"]["runs_dir"]) / "history.jsonl"
    # A combo is resumable-DONE only with its FULL round count logged (codex W2 R1):
    # a crash mid-append re-runs the combo; load_runs keeps the LAST record per round.
    done = done_runs(out, required_rounds=int(cfg["selfplay"]["rounds"]))
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
    main(argv=sys.argv[1:])  # stages come from argv via main's own parser (a bare int() crashed on --help)
