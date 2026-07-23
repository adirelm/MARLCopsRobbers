"""Report send pipeline — subject, redaction, idempotency, gatekeeper egress (T9.3/9.5).

The single send path for the §3.5 report: validate the body (schema + §3.4 Table-1
scores), format the subject from ``gmail.subject_template``, write a role-only REDACTED
copy (the tracked §7.3 evidence), then send guarded three ways via ``sentinel.py``:
an ``intent`` line BEFORE dialing SMTP (a crash mid-send blocks retries until the operator
verifies the inbox), ``sent`` recorded only after the mail is committed (idempotent
retries), and a one-email-per-match refusal of any DIFFERENT digest (``RESEND_APPROVED=1``
overrides). Egress is routed through the §5 ApiGatekeeper (``gmail`` channel). The
recipient is unconditionally ``gmail.to`` (never inlined, never an env override).

Input: the loaded config (``_validate_config`` -> ValueError), the §3.5 report body, the
INJECTED ``sender`` (real ``GmailMailer`` or a ``FakeEmailSender``), the build date and an
optional ``gatekeeper`` — the caller-supplied trio is guarded by ``_validate_input``
(TypeError).
Output: a result dict (``sent`` / ``reason`` / ``subject`` / ``to`` / ``redacted_path``),
the redacted JSON copy on disk, and the sentinel line once the mail is committed.
Setup: ``gmail.*`` config + the mailer's env credentials; inject a fake sender + a
tmp sentinel/output_dir for a dry run that touches no network.
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
from src.reporting.send_lock import send_lock
from src.reporting.sentinel import LOCK, check_clear_to_send, mark_intent, mark_sent

_REQUIRED_GMAIL_KEYS = ("to", "subject_template", "output_dir", "sentinel")


def _validate_config(cfg: dict) -> dict:
    """Return ``cfg`` after checking the ``gmail`` / ``game`` knobs the send path reads.

    Raises:
        ValueError: If a required ``gmail.*`` key is missing/empty, or
            ``game.num_games`` is not a positive int.
    """
    gmail = cfg.get("gmail")
    if not isinstance(gmail, dict):
        raise ValueError("config is missing required section 'gmail'")
    for key in _REQUIRED_GMAIL_KEYS:
        if not gmail.get(key):
            raise ValueError(f"config gmail.{key} must be set (non-empty)")
    num_games = cfg.get("game", {}).get("num_games")
    if not isinstance(num_games, int) or isinstance(num_games, bool) or num_games < 1:
        raise ValueError(f"config game.num_games must be a positive int, got {num_games!r}")
    return cfg


def _validate_input(report: object, sender: object, date_str: object) -> None:
    """Type-check the caller-supplied send arguments.

    Raises:
        TypeError: If ``report`` is not a dict, ``sender`` exposes no callable
            ``send()``, or ``date_str`` is not a str.
    """
    if not isinstance(report, dict):
        raise TypeError(f"report must be a dict, got {type(report).__name__}")
    if not callable(getattr(sender, "send", None)):
        raise TypeError("sender must expose a callable send(subject, body, recipient)")
    if not isinstance(date_str, str):
        raise TypeError(f"date_str must be a str, got {type(date_str).__name__}")


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
    PII per the project deny-list). The real URL is injected at send time and reaches ONLY the
    lecturer's email; the tracked ``*.redacted.json`` stays PII-free (the QUALITY.md boundary).
    """
    redacted = copy.deepcopy(report)
    redacted["students"] = [{"role": s["role"]} for s in report["students"]]
    redacted["github_repo"] = "<redacted — the real URL is sent in the email only>"
    return redacted


def report_hash(report: dict) -> str:
    """Return a stable sha256 over the canonical (sorted-key) report JSON."""
    canonical = json.dumps(report, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
    """Validate -> subject -> redact-copy -> guarded gatekeeper send; return a result dict.

    A report whose hash is already in ``gmail.sentinel`` is a no-op (``sent=False,
    reason=already_sent``); a CHANGED report or a dangling mid-send ``intent`` refuses
    (see :func:`gated_idempotent_send`). The sentinel is recorded INSIDE the egress
    thunk, so it is written iff/when the email actually goes out — immediately, OR when a
    deferred call later drains (the §5 gatekeeper queues, it does not cancel). That makes
    a deferred send impossible to turn into an untracked duplicate.
    """
    _validate_input(report, sender, date_str)
    _validate_config(cfg)
    # §3.5: exactly N sub-games at SEND, each scored per the §3.4 Table-1 mapping
    validate(report, expected_games=int(cfg["game"]["num_games"]), scoring=cfg["game"]["scoring"])
    subject = format_subject(cfg, report, date_str)
    redacted_path = str(write_redacted(cfg, report))
    return gated_idempotent_send(cfg, report, sender, subject, redacted_path, gatekeeper)


def gated_idempotent_send(  # noqa: PLR0913 — cfg + body + sender + subject + path + gate + scope are distinct
    cfg: dict,
    report: dict,
    sender: object,
    subject: str,
    redacted_path: str,
    gatekeeper: object = None,
    sentinel: str | None = None,
) -> dict:
    """Send ``report`` at most once through the §5 gatekeeper; return the result dict.

    The one place the send guards live — shared by the §3.5 report and the §9 bonus
    report (which passes its OWN ``sentinel`` scope). Per attempt: ``check_clear_to_send``
    no-ops an already-sent digest, BLOCKS on a dangling ``intent`` (crashed mid-send —
    verify the inbox first) and refuses a DIFFERENT digest (one email per match;
    ``RESEND_APPROVED=1`` overrides); then ``intent`` is recorded BEFORE dialing SMTP and
    ``sent`` after, with the check->intent window under ``send_lock`` (thread lock +
    ``<sentinel>.lock`` file — two PROCESSES can never both pass the gate, immediately
    or when a deferred call later drains). Callers do their own validation first.

    Raises:
        RuntimeError: Dangling ``intent`` in the sentinel (delivery state unknown).
        ValueError: A different report digest is already recorded (no resend approval).
    """
    digest = report_hash(report)
    sentinel = sentinel or cfg["gmail"]["sentinel"]
    with LOCK:
        if check_clear_to_send(sentinel, digest):
            return {"sent": False, "reason": "already_sent", "redacted_path": redacted_path}
    body = json.dumps(report, indent=2, ensure_ascii=False)
    recipient = cfg["gmail"]["to"]

    def _send() -> str:
        # Recheck INSIDE the thunk (under the send lock — thread lock + <sentinel>.lock
        # file — atomically with the intent write): a previously-queued drain, a
        # concurrent caller, or a SECOND PROCESS can never double-email (wave-2 SENT).
        with send_lock(sentinel):
            if check_clear_to_send(sentinel, digest):
                return "already_sent"
            mark_intent(sentinel, digest)  # written BEFORE dialing SMTP (crash guard)
        sender.send(subject, body, recipient)
        mark_sent(sentinel, digest)  # the email is now committed
        return "sent"

    outcome = (gatekeeper or ApiGatekeeper()).execute("gmail", _send)
    if outcome is DEFERRED:
        return {"sent": False, "reason": "deferred", "redacted_path": redacted_path}
    if outcome == "already_sent":  # the inner recheck skipped — a prior drain already delivered it
        return {"sent": False, "reason": "already_sent", "redacted_path": redacted_path}
    return {"sent": True, "subject": subject, "to": recipient, "redacted_path": redacted_path}
