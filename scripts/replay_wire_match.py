"""Replay the §9 wire-match log and save the §9.3 screenshot evidence (README bonus PNGs).

Thin CLI over :mod:`src.mcp.wire_replay`: re-runs every logged sub-game from exactly
(per-request JSONL log + §9.4 sub-game records + the config P7 seed schedule), verifies
spawns and moves/winner/scores against the records — raising loudly on ANY divergence —
then renders the t00/mid/final god-view PNG per sub-game into
``gui.bonus_screenshot_dir`` and prints the verification table + the files written.
Defaults are resolved as a MATCHING PAIR (see ``select_log_and_records``) — the log and the
records must describe the same match, which choosing them independently did not guarantee.
Run:
``uv run python scripts/replay_wire_match.py [--log ...] [--records ...] [--out ...]``.
"""

from __future__ import annotations

import argparse

from src.mcp._replay_log import select_log_and_records
from src.mcp.wire_replay import replay_match, save_screens
from src.utils.config_loader import load_config


def main(argv: list[str] | None = None) -> dict:
    """Replay + verify the match, save the PNGs, print the evidence table; return the paths."""
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Replay the §9 wire-match log into §9.3 screenshots")
    parser.add_argument("--log", default=None, help="referee JSONL log (default: newest in log_dir)")
    parser.add_argument("--records", default=None, help="§9.4 sub-game records JSON (default: from config)")
    parser.add_argument("--out", default=None, help="output dir (default: gui.bonus_screenshot_dir)")
    args = parser.parse_args(argv)
    # Resolved as a PAIR: picking the newest log and the fallback records independently
    # crashed on every fresh clone (real match log vs rehearsal records).
    pair_log, pair_records = (None, None) if (args.log and args.records) else select_log_and_records(cfg)
    log = args.log or str(pair_log)
    records = args.records or str(pair_records)
    replays = replay_match(cfg, log, records)  # raises ReplayMismatchError on ANY divergence
    print(f"[wire-replay] log:     {log}")
    print(f"[wire-replay] P7 seed schedule from config: {cfg['wire_match']['seeds']}")
    print(f"[wire-replay] records: {records}")
    print("sub-game  seed  spawn-verify  moves  winner  scores(cop/thief)")
    for game in replays:
        scores = f"{game['scores']['cop']}/{game['scores']['thief']}"
        print(
            f"  sg{game['gid']}     {game['seed']:>5}  OK            {game['moves']:>5}  "
            f"{game['winner']:<6}  {scores}"
        )
    saved = save_screens(cfg, replays, args.out)
    print(f"[wire-replay] wrote {len(saved)} PNGs:")
    for path in saved:
        print(f"  {path}")
    return {"log": log, "records": records, "screens": [str(p) for p in saved]}


if __name__ == "__main__":
    main()
