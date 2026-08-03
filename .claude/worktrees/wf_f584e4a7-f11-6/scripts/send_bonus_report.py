"""Send the §9.4 bonus report for the PLAYED inter-group match (adrl-001 vs biu-azri).

Reads the draft the referee wrote (``wire_match.draft_report``, git-ignored — it carries
both groups' real PII), re-derives the §5 byte-compare digest so the operator can confirm
it still matches the partner's, and sends exactly once.

``--agreed`` is what flips ``mutual_agreement`` to True. It is a separate, explicit flag
because §9.3 makes that flip the whole gate: emailing before both groups have compared
identical drafts scores BOTH groups zero. The flag asserts a fact about the other group,
so it must be a deliberate act, never a default.

Run: ``uv run python scripts/send_bonus_report.py [--agreed] [--send]``
  (no flags)          -> print the report + compare digest, change nothing
  --agreed            -> also show what WOULD be sent, still no egress
  --agreed --send     -> the real, idempotent, one-per-match send
Setup: needs ``GMAIL_SENDER`` / ``GMAIL_APP_PASSWORD`` for ``--send``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from src.utils.config_loader import load_config

_COMPARE_KEYS = ("sub_games", "totals_by_group", "bonus_claim")


def compare_digest(report: dict) -> tuple[int, str]:
    """Return ``(byte length, sha256)`` of the §5 compare subject, canonicalised.

    Sorted keys + compact separators so neither side's formatting can fake a mismatch;
    these three sections carry no identity data, so the digest is safe to exchange.
    """
    canon = json.dumps(
        {k: report[k] for k in _COMPARE_KEYS}, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return len(canon), hashlib.sha256(canon.encode("utf-8")).hexdigest()


def load_draft(cfg: dict, agreed: bool = False) -> dict:
    """Load the referee's draft; ``agreed`` flips ``mutual_agreement`` to True."""
    report = json.loads(Path(cfg["wire_match"]["draft_report"]).read_text(encoding="utf-8"))
    if agreed:
        report["mutual_agreement"] = True
    return report


def main(cfg: dict | None = None, agreed: bool = False, send: bool = False) -> dict:
    """Print the report's decisive facts; with ``send`` perform the real gated egress."""
    cfg = cfg or load_config()
    report = load_draft(cfg, agreed=agreed)
    length, digest = compare_digest(report)
    print(f"[bonus] totals={report['totals_by_group']} claim={report['bonus_claim']}")
    print(f"[bonus] compare subject: {length} bytes  sha256 {digest}")
    print(f"[bonus] mutual_agreement={report['mutual_agreement']}")
    if send:
        from src.reporting.bonus_send import send_bonus_report  # noqa: PLC0415 - lazy: mail extra
        from src.reporting.mailer import GmailMailer  # noqa: PLC0415

        result = send_bonus_report(cfg, report, GmailMailer(cfg))
        print(f"[email] sent={result['sent']} reason={result.get('reason', 'ok')}")
        print(f"[email] to={result.get('to')}\n[email] subject={result.get('subject')}")
    return report


if __name__ == "__main__":
    main(agreed="--agreed" in sys.argv, send="--send" in sys.argv)
