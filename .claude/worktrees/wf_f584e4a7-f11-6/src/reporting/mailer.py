"""Email senders — EmailSender Protocol + Gmail App-Password + a fake (T9.4).

``GmailMailer`` sends the §3.5 report via ``smtplib`` STARTTLS + a Gmail App
Password (creds from the env ``GMAIL_SENDER`` / ``GMAIL_APP_PASSWORD``; the SMTP
class is injectable for tests). ``FakeEmailSender`` records sends in-memory with NO
network — the default in tests + dry runs. The recipient is always ``gmail.to``
(never inlined). UTF-8 bodies preserve Hebrew names (``ensure_ascii=False`` upstream).
"""

from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from typing import Protocol


class EmailSender(Protocol):
    """Anything that can send one report email (Protocol — structural typing)."""

    def send(self, subject: str, body: str, to: str) -> None:
        """Send ``body`` with ``subject`` to ``to``."""
        ...


class FakeEmailSender:
    """Records sends in-memory (no network) — the test / dry-run default."""

    def __init__(self) -> None:
        """Start with an empty send log."""
        self.sent: list[dict] = []

    def send(self, subject: str, body: str, to: str) -> None:
        """Append the (subject, body, to) to the in-memory log."""
        self.sent.append({"subject": subject, "body": body, "to": to})


class GmailMailer:
    """Send the report via Gmail ``smtplib`` STARTTLS + an App Password."""

    def __init__(self, cfg: dict, smtp_factory: object = None) -> None:
        """Read SMTP host/port from config + sender creds from the env."""
        gmail = cfg["gmail"]
        self._host = gmail["smtp_host"]
        self._port = int(gmail["smtp_port"])
        self._sender = os.environ.get("GMAIL_SENDER")
        self._password = os.environ.get("GMAIL_APP_PASSWORD")
        self._smtp_factory = smtp_factory or smtplib.SMTP

    def send(self, subject: str, body: str, to: str) -> None:
        """STARTTLS -> login -> send one UTF-8 message; require env creds.

        Raises:
            ValueError: If ``GMAIL_SENDER`` / ``GMAIL_APP_PASSWORD`` are unset.
        """
        if not (self._sender and self._password):
            raise ValueError("GMAIL_SENDER / GMAIL_APP_PASSWORD must be set to send")
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self._sender
        message["To"] = to
        message.set_content(body)
        with self._smtp_factory(self._host, self._port) as smtp:
            smtp.starttls()
            smtp.login(self._sender, self._password)
            smtp.send_message(message)
