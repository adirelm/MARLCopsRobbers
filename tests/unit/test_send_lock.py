"""In-process pins for ``send_lock`` (wave-2 SENT): acquire/release, contention, staleness.

The lock guards an email send, so contention fails CLOSED: a held fresh lock refuses
after ``SENTINEL_LOCK_TIMEOUT_S``. Staleness is intent-aware (S1): a stale lock with a
dangling ``intent`` (delivery UNKNOWN) refuses and is NEVER auto-deleted, but a stale
lock with NO intent — a crash BEFORE mark_intent, SMTP provably never dialed — is
RECLAIMED so a recoverable pre-send crash cannot block the mandatory email forever. The
two-process end-to-end race lives in ``tests/integration/test_sentinel_process_lock.py``.
"""

from __future__ import annotations

import os
import time

import pytest

from src.reporting.send_lock import _env_seconds, send_lock


def test_send_lock_creates_then_removes_the_lockfile(tmp_path):
    sentinel = tmp_path / "s.sha256"
    lockfile = tmp_path / "s.sha256.lock"
    with send_lock(sentinel):
        assert lockfile.exists()  # held: O_CREAT|O_EXCL claimed the mutex
    assert not lockfile.exists()  # released: unlinked on the way out


def test_held_fresh_lock_refuses_after_the_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_LOCK_TIMEOUT_S", "0.2")
    lockfile = tmp_path / "s.sha256.lock"
    lockfile.touch()  # another process holds the lock RIGHT NOW
    with pytest.raises(RuntimeError, match="held by another send process"), send_lock(tmp_path / "s.sha256"):
        pytest.fail("must not enter the critical section")  # pragma: no cover
    assert lockfile.exists()  # fail-closed: the other holder's lock is untouched


def test_stale_lock_with_dangling_intent_refuses_and_is_never_deleted(tmp_path, monkeypatch):
    # S1: a stale lock whose sentinel HAS a dangling `intent` means the crashed send
    # may have dialed SMTP — delivery is UNKNOWN, so it stays fail-CLOSED.
    monkeypatch.setenv("SENTINEL_LOCK_STALE_S", "5")
    sentinel = tmp_path / "s.sha256"
    sentinel.write_text("intent deadbeef\n", encoding="utf-8")  # recorded, no matching sent
    lockfile = tmp_path / "s.sha256.lock"
    lockfile.touch()
    crashed_at = time.time() - 60  # older than the stale threshold -> presumed crashed
    os.utime(lockfile, (crashed_at, crashed_at))
    with pytest.raises(RuntimeError, match="STALE"), send_lock(sentinel):
        pytest.fail("must not enter the critical section")  # pragma: no cover
    assert lockfile.exists()  # NEVER auto-deleted — the operator removes it consciously


def test_stale_lock_with_no_intent_is_reclaimed(tmp_path, monkeypatch):
    # S1: a crash AFTER O_EXCL but BEFORE mark_intent leaves a lockfile with NO intent
    # in the sentinel — SMTP was provably never dialed, so the mandatory email must not
    # be blocked forever. The stale lock is reclaimed and the send proceeds.
    monkeypatch.setenv("SENTINEL_LOCK_STALE_S", "5")
    monkeypatch.setenv("SENTINEL_LOCK_TIMEOUT_S", "0.3")
    sentinel = tmp_path / "s.sha256"  # no sentinel file at all -> no dangling intent
    lockfile = tmp_path / "s.sha256.lock"
    lockfile.touch()
    crashed_at = time.time() - 60  # pre-send crash left this behind
    os.utime(lockfile, (crashed_at, crashed_at))
    entered = False
    with send_lock(sentinel):
        entered = True
        assert lockfile.exists()  # reclaimed: we now hold a fresh lock at the same path
    assert entered  # the critical section was reached, not refused forever
    assert not lockfile.exists()  # released on the way out


def test_env_seconds_ignores_garbage_and_keeps_the_default(monkeypatch):
    monkeypatch.setenv("SENTINEL_LOCK_TIMEOUT_S", "not-a-number")
    assert _env_seconds("SENTINEL_LOCK_TIMEOUT_S", 10.0) == 10.0
    monkeypatch.setenv("SENTINEL_LOCK_TIMEOUT_S", "2.5")
    assert _env_seconds("SENTINEL_LOCK_TIMEOUT_S", 10.0) == 2.5
