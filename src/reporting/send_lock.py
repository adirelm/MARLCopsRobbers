"""Inter-process send lock for the sentinel check->intent critical section (§3.5/§9.3).

The wave-2 review showed the thread ``LOCK`` alone lets two PROCESSES both pass
``check_clear_to_send`` before either records ``intent`` — and both email. This module
adds the missing mutex: a ``<sentinel>.lock`` file created with ``O_CREAT|O_EXCL``
(atomic on POSIX), unlinked on release, with the thread lock kept underneath.
Fail-CLOSED policy (it guards an email): contention refuses after a bounded wait. A
stale lockfile is reclaimed ONLY when the sentinel records NO dangling ``intent`` — a
pre-send crash provably never dialed SMTP, so the MANDATORY email must not be blocked
forever; a stale lock WITH a dangling intent (delivery UNKNOWN) still refuses and is
NEVER auto-deleted.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path

from src.reporting.sentinel import LOCK, has_open_intent


def _env_seconds(name: str, default: float) -> float:
    """Read a seconds override from the env (tests tune these; defaults stay safe)."""
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _refuse_or_reclaim(lockfile: Path, sentinel: str | Path, deadline: float) -> None:
    """Reclaim a stale lock the send never got past; else raise for a live/held one.

    A STALE lockfile with NO dangling ``intent`` in the sentinel is a pre-send crash:
    ``intent`` is written before dialing SMTP and the lock is dropped before the dial,
    so the email was PROVABLY never attempted — the lockfile is unlinked and the caller
    retries (otherwise the MANDATORY send is blocked forever by a recoverable crash). A
    stale lock WITH a dangling intent keeps the fail-CLOSED stance (delivery UNKNOWN):
    it refuses with operator instructions and is NEVER auto-deleted.
    """
    with contextlib.suppress(FileNotFoundError):  # the holder may release between checks
        age = time.time() - lockfile.stat().st_mtime
        if age > _env_seconds("SENTINEL_LOCK_STALE_S", 300.0):
            if not has_open_intent(sentinel):
                with contextlib.suppress(FileNotFoundError):
                    lockfile.unlink()  # crash before intent -> SMTP never dialed -> reclaim
                return
            raise RuntimeError(
                f"sentinel lockfile {lockfile} looks STALE (age {age:.0f}s) WITH a dangling "
                "'intent' line — a previous send crashed after dialing SMTP, so delivery state "
                "is UNKNOWN. Verify the recipient inbox, then delete the lockfile manually and "
                "clear the intent line and retry (never auto-removed)."
            )
    if time.monotonic() > deadline:
        raise RuntimeError(
            f"sentinel lockfile {lockfile} is held by another send process — refusing after "
            "SENTINEL_LOCK_TIMEOUT_S to keep at most one email. Let that send finish and retry."
        )


@contextlib.contextmanager
def send_lock(sentinel: str | Path):
    """Serialize the check->intent critical section across THREADS and PROCESSES.

    Creates ``<sentinel>.lock`` with ``O_CREAT|O_EXCL`` (the inter-process mutex; the
    thread ``LOCK`` is kept underneath) and unlinks it on release. Contention polls up
    to ``SENTINEL_LOCK_TIMEOUT_S`` (default 10s) then refuses. A lockfile older than
    ``SENTINEL_LOCK_STALE_S`` (default 300s) is reclaimed when the sentinel has NO
    dangling ``intent`` (a pre-send crash never dialed SMTP), else refused with operator
    instructions — this guards an email send, so the dangerous case fails CLOSED.
    """
    lockfile = Path(f"{sentinel}.lock")
    lockfile.parent.mkdir(parents=True, exist_ok=True)
    with LOCK:
        deadline = time.monotonic() + _env_seconds("SENTINEL_LOCK_TIMEOUT_S", 10.0)
        while True:
            try:
                handle = os.open(lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                _refuse_or_reclaim(lockfile, sentinel, deadline)
                time.sleep(0.05)
        try:
            yield
        finally:
            os.close(handle)
            with contextlib.suppress(FileNotFoundError):
                os.unlink(lockfile)
