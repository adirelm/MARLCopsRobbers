"""Send-guard tests: one-email-per-match refusal + the SMTP intent line (crash/concurrency).

Pins the two guards layered over content idempotency: (1) a CHANGED report digest next
to an already-sent one REFUSES unless RESEND_APPROVED=1 (a corrected resend is an explicit
operator decision, never an accident); (2) an `intent <digest>` line is written BEFORE
dialing SMTP, so a crash after SMTP-accept but before the sent-mark — or a concurrent
caller racing the SMTP window — can never silently double-email: retries BLOCK until the
operator verifies the inbox and clears the line manually.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from src.reporting import send as send_mod
from src.reporting.mailer import FakeEmailSender
from src.reporting.sentinel import already_sent, mark_intent
from tests.unit._send_fixtures import cfg_tmp as _cfg_tmp
from tests.unit._send_fixtures import make_report as _report


def test_changed_report_is_refused_without_resend_approval(cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("RESEND_APPROVED", raising=False)
    cfg = _cfg_tmp(cfg, tmp_path)
    sender = FakeEmailSender()
    send_mod.send_report(cfg, _report(), sender, "2026-06-21")
    changed = _report()
    changed["sub_games"][0]["moves"] = 5  # a "corrected" report -> different digest
    with pytest.raises(ValueError, match="RESEND_APPROVED"):
        send_mod.send_report(cfg, changed, sender, "2026-06-21")
    assert len(sender.sent) == 1  # the changed report NEVER left the machine


def test_resend_approved_env_allows_a_conscious_corrected_resend(cfg, tmp_path, monkeypatch):
    cfg = _cfg_tmp(cfg, tmp_path)
    sender = FakeEmailSender()
    send_mod.send_report(cfg, _report(), sender, "2026-06-21")
    changed = _report()
    changed["sub_games"][0]["moves"] = 5
    monkeypatch.setenv("RESEND_APPROVED", "1")
    out = send_mod.send_report(cfg, changed, sender, "2026-06-21")
    assert out["sent"] is True and len(sender.sent) == 2
    monkeypatch.delenv("RESEND_APPROVED")
    again = send_mod.send_report(cfg, changed, sender, "2026-06-21")  # identical -> plain no-op
    assert again["reason"] == "already_sent" and len(sender.sent) == 2


class _CrashAfterSmtp(FakeEmailSender):
    """Delivers the mail, then crashes BEFORE the caller can record the sent-mark."""

    def send(self, subject: str, body: str, recipient: str) -> None:
        super().send(subject, body, recipient)
        raise ConnectionResetError("simulated crash AFTER the SMTP server accepted the mail")


def test_crash_between_smtp_and_mark_blocks_the_retry(cfg, tmp_path):
    cfg = _cfg_tmp(cfg, tmp_path)
    sender = _CrashAfterSmtp()
    with pytest.raises(ConnectionResetError):
        send_mod.send_report(cfg, _report(), sender, "2026-06-21")
    assert len(sender.sent) == 1
    sentinel = Path(cfg["gmail"]["sentinel"])
    assert "intent" in sentinel.read_text(encoding="utf-8")  # the pre-SMTP intent line survives
    with pytest.raises(RuntimeError, match="inbox"):  # retry BLOCKS: delivery state is unknown
        send_mod.send_report(cfg, _report(), sender, "2026-06-21")
    assert len(sender.sent) == 1  # no silent duplicate
    sentinel.write_text("", encoding="utf-8")  # operator verified inbox: mail lost -> clear the line
    out = send_mod.send_report(cfg, _report(), FakeEmailSender(), "2026-06-21")
    assert out["sent"] is True  # after the manual clear the send proceeds


def test_dangling_intent_blocks_before_any_smtp_dial(cfg, tmp_path):
    cfg = _cfg_tmp(cfg, tmp_path)
    sender = FakeEmailSender()
    report = _report()
    mark_intent(cfg["gmail"]["sentinel"], send_mod.report_hash(report))
    with pytest.raises(RuntimeError, match="intent"):
        send_mod.send_report(cfg, report, sender, "2026-06-21")
    assert sender.sent == []  # refused before dialing


class _HoldingSender(FakeEmailSender):
    """Barrier-style double: holds the first send open so a second caller overlaps it."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def send(self, subject: str, body: str, recipient: str) -> None:
        self.entered.set()
        assert self.release.wait(timeout=5)
        super().send(subject, body, recipient)


def test_concurrent_caller_during_the_smtp_window_cannot_double_send(cfg, tmp_path):
    cfg = _cfg_tmp(cfg, tmp_path)
    sender = _HoldingSender()
    report = _report()
    thread = threading.Thread(target=send_mod.send_report, args=(cfg, report, sender, "2026-06-21"))
    thread.start()
    assert sender.entered.wait(timeout=5)  # first caller is now INSIDE the SMTP dial
    with pytest.raises(RuntimeError, match="intent"):  # racer sees the intent line -> blocked
        send_mod.send_report(cfg, report, sender, "2026-06-21")
    sender.release.set()
    thread.join(timeout=5)
    assert len(sender.sent) == 1  # exactly one email left the machine
    assert already_sent(cfg["gmail"]["sentinel"], send_mod.report_hash(report))


def test_legacy_bare_digest_sentinel_lines_still_count_as_sent(cfg, tmp_path):
    cfg = _cfg_tmp(cfg, tmp_path)
    sender = FakeEmailSender()
    report = _report()
    sentinel = Path(cfg["gmail"]["sentinel"])
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(send_mod.report_hash(report) + "\n", encoding="utf-8")  # pre-guard format
    out = send_mod.send_report(cfg, report, sender, "2026-06-21")
    assert out["sent"] is False and out["reason"] == "already_sent"
    assert sender.sent == []
