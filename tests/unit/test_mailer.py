"""Mailer tests (T9.4) — GmailMailer STARTTLS order + FakeEmailSender, zero live sends."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.reporting.mailer import FakeEmailSender, GmailMailer


def test_fake_sender_records_sends_without_network():
    sender = FakeEmailSender()
    sender.send("subj", "body", "to@example.com")
    assert sender.sent == [{"subject": "subj", "body": "body", "to": "to@example.com"}]


def test_gmail_mailer_starttls_login_send_order(monkeypatch, cfg):
    monkeypatch.setenv("GMAIL_SENDER", "me@gmail.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "app-pw")
    smtp = MagicMock()
    smtp.__enter__.return_value = smtp
    factory = MagicMock(return_value=smtp)
    GmailMailer(cfg, smtp_factory=factory).send("S", "B — שלום", "rcpt@example.com")
    factory.assert_called_once_with(cfg["gmail"]["smtp_host"], int(cfg["gmail"]["smtp_port"]))
    smtp.starttls.assert_called_once()
    smtp.login.assert_called_once_with("me@gmail.com", "app-pw")
    smtp.send_message.assert_called_once()
    message = smtp.send_message.call_args[0][0]
    assert message["To"] == "rcpt@example.com"
    assert "שלום" in message.get_content()  # Hebrew survives the UTF-8 body


def test_gmail_mailer_requires_env_creds(monkeypatch, cfg):
    monkeypatch.delenv("GMAIL_SENDER", raising=False)
    monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
    with pytest.raises(ValueError, match="GMAIL_SENDER"):
        GmailMailer(cfg).send("S", "B", "to@example.com")
