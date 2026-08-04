"""Pick a log and records that describe the SAME match (split from _replay_log at the cap).

Separated because it answers a different question from parsing: not "what does this log say"
but "which two artifacts belong together". Choosing them independently was a real bug we
shipped — see :func:`select_log_and_records`.
"""

from __future__ import annotations

import json
from pathlib import Path


def select_log_and_records(cfg: dict) -> tuple[Path, Path]:
    """Return a log and records that describe the SAME match — never a mismatched pair.

    Choosing them independently is a real bug we shipped: the log default took the newest
    timestamped file while the records default fell back to the committed REHEARSAL records
    whenever the git-ignored real draft was absent. On any fresh clone that pairs the real
    §9 match log with rehearsal records, and the README's documented replay command dies
    with ReplayMismatchError before printing anything.

    Preference order: the git-ignored real draft (local only), then the TRACKED redacted
    §9.4 body, then the rehearsal records. The redacted copy is what makes a fresh clone
    work at all — it masks both student blocks and both repo URLs but keeps ``sub_games``
    intact, so a grader who clones the public repo replays the REAL graded match rather
    than being handed rehearsal records the config's seed list can no longer verify.

    Matching on (id, winner, moves) rather than on filenames means adding more logs later
    cannot silently re-pair them.

    Raises:
        SystemExit: When no committed log matches any available records file.
    """
    log_dir = Path(cfg["wire_match"]["log_dir"])
    logs = sorted(log_dir.glob("wire_log_[0-9]*.jsonl"), reverse=True)  # newest first
    candidates = [
        Path(cfg["wire_match"]["draft_report"]),  # local real draft (git-ignored: carries PII)
        Path(cfg["wire_match"]["redacted_records"]),  # TRACKED — what a fresh clone gets
        Path(cfg["wire_match"]["rehearsal"]["records"]),
    ]
    for records in candidates:
        if not records.exists():
            continue
        want = [
            (g["id"], g["winner"], g["moves"])
            for g in json.loads(records.read_text(encoding="utf-8"))["sub_games"]
        ]
        for log in logs:
            results = [
                json.loads(line)["sub_game"]
                for line in log.read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line).get("direction") == "result"
            ]
            if [(r["id"], r["winner"], r["moves"]) for r in results] == want:
                return log, records
    raise SystemExit(
        f"no log under {log_dir} matches any available records — pass --log and --records explicitly"
    )
