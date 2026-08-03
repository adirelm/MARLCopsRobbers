"""SMTP App-Password smoke test (T9.5; run days before the demo — REQUIRES creds).

Sends a tiny UTF-8 test email to ``GMAIL_SENDER`` itself to confirm the Gmail App
Password + STARTTLS path works BEFORE the real lecturer send (and that a Hebrew body
survives). Run with ``GMAIL_SENDER`` + ``GMAIL_APP_PASSWORD`` set in the env:
``uv run python scripts/smtp_smoke.py``. Never sends to the lecturer.
"""

from __future__ import annotations

import os

from src.reporting.mailer import GmailMailer
from src.utils.config_loader import load_config


def main(cfg: dict | None = None) -> None:  # pragma: no cover - requires creds + network
    """Send one self-addressed test email to validate the App Password path."""
    cfg = cfg or load_config()
    sender_addr = os.environ["GMAIL_SENDER"]
    GmailMailer(cfg).send(
        "[MARL] SMTP smoke test — שלום",
        "If you received this, the Gmail App Password + STARTTLS egress path works.",
        sender_addr,
    )
    print(f"[smtp-smoke] sent a self-test email to {sender_addr} — check the inbox")


if __name__ == "__main__":
    main()
