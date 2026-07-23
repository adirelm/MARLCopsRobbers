"""Sentinel-file bookkeeping for the one-email-per-match send guards (§3.5 / §9.3).

Line format: ``intent <sha256>`` is appended immediately BEFORE dialing SMTP and
``sent <sha256>`` after the mail is committed; legacy bare-digest lines still count
as sent (backward compatible). Two guards live here beyond plain idempotency:
a DANGLING intent (no matching ``sent``) blocks every further send on that sentinel
until the operator verifies the recipient inbox and deletes the line manually, and a
NEW digest next to an already-recorded one is refused — one email per match — unless
the operator consciously sets ``RESEND_APPROVED=1`` for a corrected resend.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

# Serializes the check->intent critical section across threads in this process; the
# intent line itself is the (pragmatic, single-operator) cross-process/crash guard.
LOCK = threading.Lock()

_TAGGED = 2  # a tagged record line is exactly "sent <digest>" / "intent <digest>"


def _entries(sentinel: str | Path) -> tuple[set[str], set[str]]:
    """Return the ``(sent, intent)`` digest sets recorded in the sentinel file."""
    sent: set[str] = set()
    intents: set[str] = set()
    path = Path(sentinel)
    if not path.exists():
        return sent, intents
    for line in path.read_text(encoding="utf-8").splitlines():
        words = line.split()
        if len(words) == 1:  # legacy bare-digest line == sent
            sent.add(words[0])
        elif len(words) == _TAGGED and words[0] == "sent":
            sent.add(words[1])
        elif len(words) == _TAGGED and words[0] == "intent":
            intents.add(words[1])
    return sent, intents


def already_sent(sentinel: str | Path, digest: str) -> bool:
    """Return whether ``digest`` is recorded as SENT in the sentinel file."""
    sent, _ = _entries(sentinel)
    return digest in sent


def _append(sentinel: str | Path, line: str) -> None:
    """Append one record line to the sentinel file (parent dirs created)."""
    path = Path(sentinel)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def mark_intent(sentinel: str | Path, digest: str) -> None:
    """Record ``intent <digest>`` — called immediately BEFORE dialing SMTP."""
    _append(sentinel, f"intent {digest}")


def mark_sent(sentinel: str | Path, digest: str) -> None:
    """Record ``sent <digest>`` — called ONLY after a successful send."""
    _append(sentinel, f"sent {digest}")


def check_clear_to_send(sentinel: str | Path, digest: str) -> bool:
    """Gate a send attempt against the recorded sentinel state.

    Returns:
        True when ``digest`` is already recorded as sent (the caller must no-op),
        False when the send may proceed.

    Raises:
        RuntimeError: A dangling ``intent`` line exists — a previous attempt dialed
            SMTP without confirming, so delivery state is UNKNOWN. Verify the
            recipient inbox; if the mail did NOT arrive, delete that line from the
            sentinel file manually and retry.
        ValueError: The sentinel already records a DIFFERENT report digest — the
            brief allows one email per match. Set env ``RESEND_APPROVED=1`` to
            consciously send a corrected report anyway.
    """
    sent, intents = _entries(sentinel)
    if digest in sent:
        return True
    dangling = sorted(intents - sent)
    if dangling:
        raise RuntimeError(
            f"sentinel {sentinel} has a dangling 'intent {dangling[0][:12]}…' line: a previous send "
            "dialed SMTP without confirming, so delivery state is UNKNOWN. Verify the recipient "
            "inbox; if (and only if) the mail did NOT arrive, delete that line manually and retry."
        )
    if sent and os.environ.get("RESEND_APPROVED") != "1":
        raise ValueError(
            f"sentinel {sentinel} already records a sent report with a DIFFERENT digest — the brief "
            "allows one email per match, so a changed report is refused. If this is a consciously "
            "corrected resend, set the env var RESEND_APPROVED=1 and retry."
        )
    return False
