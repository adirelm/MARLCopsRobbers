"""Send the §3.5 report for the RECORDED CLOUD match (the §5.3 Stage-2 game).

``scripts/run_match.py`` replays a fresh in-memory match; this script instead reports the
match the referee actually drove against the two Render-deployed MCP servers over the public
internet, so the emailed body carries the real distributed run's timestamps. Both produce the
identical result (cop 30 - thief 60) — the cloud one is the graded §5.3 Stage-2 evidence.

Input: ``results/subgames/cloud_match_5x5.redacted.json`` (the tracked, identity-redacted
record) + ``players.local.yaml`` (git-ignored) for the student/repo block the redaction removed.
Output: the assembled + validated §3.5 body; with ``--send`` it performs the REAL Gmail egress.
Setup: needs ``GMAIL_SENDER`` / ``GMAIL_APP_PASSWORD`` in the env for ``--send``; the send is
idempotent (sha256 sentinel) and HARD-GATED behind an explicit human go.

Run: ``uv run python scripts/send_cloud_report.py [--send]``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import zoneinfo
from pathlib import Path

from src.sdk.sdk import MarlSDK
from src.utils.config_loader import load_config

_RECORD = Path("results/subgames/cloud_match_5x5.redacted.json")


def build_cloud_report(cfg: dict, record_path: Path = _RECORD) -> dict:
    """Assemble the §3.5 body for the recorded cloud match; raise if it fails validation.

    The per-sub-game records are the referee's REAL cloud timings/scores; only the identity
    block is restored (that is exactly what the tracked copy's redaction stripped).
    """
    from src.reporting.players import load_players  # noqa: PLC0415 — lazy: reads a local file
    from src.reporting.schema import Student, build_report, validate  # noqa: PLC0415

    players = load_players()
    recorded = json.loads(record_path.read_text(encoding="utf-8"))
    report = build_report(
        players["group_name"],
        [Student(**student) for student in players["students"]],
        players["github_repo"],
        cfg["project"]["timezone"],
        recorded["sub_games"],
    ).to_dict()
    validate(report, expected_games=int(cfg["game"]["num_games"]), scoring=cfg["game"]["scoring"])
    return report


def main(cfg: dict | None = None, send: bool = False, argv: list[str] | None = None) -> dict:
    """Print the cloud match's §3.5 totals; ``send`` performs the real (idempotent) egress."""
    # argparse alone gives --help and rejects unknown flags. argv=None means NOT a
    # CLI call, so an in-process main() never parses the caller's sys.argv.
    # Without this the parser read pytest's argv and every such test died.
    # script IGNORED argv, so a documented `--help` started the real job.
    parser = argparse.ArgumentParser(description="Print or send the 3.5 cloud match report")
    parser.add_argument("--send", action="store_true")
    parser.parse_args(argv or [])
    cfg = cfg or load_config()
    report = build_cloud_report(cfg)
    games = report["sub_games"]
    span = datetime.datetime.fromisoformat(games[-1]["end"]) - datetime.datetime.fromisoformat(
        games[0]["start"]
    )
    print(f"[cloud-report] {len(games)} sub-games | totals={report['totals']} | match span {span}")
    if send:
        date_str = datetime.datetime.now(zoneinfo.ZoneInfo(report["timezone"])).strftime("%Y-%m-%d")
        # date_str MUST be keyword — the 2nd positional of send_final_report is `sender`.
        result = MarlSDK(cfg).send_final_report(report, date_str=date_str)
        # Echo recipient + subject: this send is irreversible, so the operator must be able to
        # confirm WHERE it landed without digging through the sentinel.
        print(f"[email] sent={result['sent']} reason={result.get('reason', 'ok')}")
        print(f"[email] to={result.get('to')}\n[email] subject={result.get('subject')}")
    return report


if __name__ == "__main__":
    main(send="--send" in sys.argv, argv=sys.argv[1:])
