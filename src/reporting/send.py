"""Report send pipeline — subject, redaction, idempotency, gatekeeper egress (T9.3/9.5).

The single send path for the §3.5 report: validate the body, format the subject from
``gmail.subject_template``, write a role-only REDACTED copy (the tracked §7.3 evidence),
then send EXACTLY once — the canonical report hash is recorded in ``gmail.sentinel`` only
AFTER a successful send, so a retry / double-trigger never emails the lecturer twice.
Egress is routed through the §5 ApiGatekeeper (``gmail`` channel). The recipient is
unconditionally ``gmail.to`` (never inlined, never an env override).
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.api.gatekeeper import DEFERRED, ApiGatekeeper
from src.reporting.schema import validate


def format_subject(cfg: dict, report: dict, date_str: str) -> str:
    """Render ``gmail.subject_template`` from the report totals + build date (no PII)."""
    return cfg["gmail"]["subject_template"].format(
        group_name=report["group_name"],
        num_games=len(report["sub_games"]),
        cop_total=report["totals"]["cop"],
        thief_total=report["totals"]["thief"],
        date=date_str,
    )


def redact_report(report: dict) -> dict:
    """Return a copy with ALL PII masked: students to role labels + the ``github_repo`` owner slug.

    Drops student ``full_name``/``id`` to ``role`` only AND masks ``github_repo`` (its owner slug is
    PII per the project deny-list). The real URL survives only in the git-ignored ``*.real.json`` copy
    + the lecturer email; the tracked ``*.redacted.json`` stays PII-free (the QUALITY.md boundary).
    """
    redacted = copy.deepcopy(report)
    redacted["students"] = [{"role": s["role"]} for s in report["students"]]
    redacted["github_repo"] = "<redacted — real URL only in the git-ignored *.real.json + the email>"
    return redacted


def report_hash(report: dict) -> str:
    """Return a stable sha256 over the canonical (sorted-key) report JSON."""
    canonical = json.dumps(report, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def already_sent(sentinel: str | Path, digest: str) -> bool:
    """Return whether ``digest`` is already recorded in the sentinel file."""
    path = Path(sentinel)
    return path.exists() and digest in path.read_text(encoding="utf-8").split()


def mark_sent(sentinel: str | Path, digest: str) -> None:
    """Append ``digest`` to the sentinel (called ONLY after a successful send)."""
    path = Path(sentinel)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(digest + "\n")


def write_redacted(cfg: dict, report: dict) -> Path:
    """Write the role-only redacted copy under ``gmail.output_dir``; return its path."""
    out_dir = Path(cfg["gmail"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "match_report.redacted.json"
    body = json.dumps(redact_report(report), indent=2, ensure_ascii=False)
    path.write_text(body, encoding="utf-8")
    return path


def build_date(cfg: dict) -> str:
    """Return today's date (ISO ``YYYY-MM-DD``) in the configured project timezone."""
    return datetime.now(ZoneInfo(cfg["project"]["timezone"])).date().isoformat()


def send_report(cfg: dict, report: dict, sender: object, date_str: str, gatekeeper: object = None) -> dict:
    """Validate -> subject -> redact-copy -> idempotent gatekeeper send; return a result dict.

    Sends EXACTLY once: a report whose hash is already in ``gmail.sentinel`` is a no-op
    (``sent=False, reason=already_sent``). The sentinel is recorded INSIDE the egress
    thunk, so it is written iff/when the email actually goes out — immediately, OR when a
    deferred call later drains (the §5 gatekeeper queues, it does not cancel). That makes
    a deferred send impossible to turn into an untracked duplicate.
    """
    validate(report, expected_games=int(cfg["game"]["num_games"]))  # §3.5: exactly N at SEND
    digest = report_hash(report)
    redacted_path = str(write_redacted(cfg, report))
    sentinel = cfg["gmail"]["sentinel"]
    if already_sent(sentinel, digest):
        return {"sent": False, "reason": "already_sent", "redacted_path": redacted_path}
    subject = format_subject(cfg, report, date_str)
    body = json.dumps(report, indent=2, ensure_ascii=False)
    recipient = cfg["gmail"]["to"]

    def _send() -> str:
        # Recheck INSIDE the thunk: if a previously-queued send for this same report
        # already drained (gatekeeper drains queued calls before admitting new ones), the
        # sentinel is now set — skip, so a retry-during-deferral can never double-email.
        if already_sent(sentinel, digest):
            return "already_sent"
        sender.send(subject, body, recipient)
        mark_sent(sentinel, digest)  # email + sentinel committed together
        return "sent"

    gate = gatekeeper or ApiGatekeeper()
    outcome = gate.execute("gmail", _send)
    if outcome is DEFERRED:
        return {"sent": False, "reason": "deferred", "redacted_path": redacted_path}
    if outcome == "already_sent":  # the inner recheck skipped — a prior drain already delivered it
        return {"sent": False, "reason": "already_sent", "redacted_path": redacted_path}
    return {"sent": True, "subject": subject, "to": recipient, "redacted_path": redacted_path}
