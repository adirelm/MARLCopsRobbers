"""In-process pins for ``send_lock`` (wave-2 SENT): acquire/release, contention, staleness.

The lock guards an email send, so every failure mode must fail CLOSED: a held fresh
lock refuses after ``SENTINEL_LOCK_TIMEOUT_S``; a stale lock refuses with operator
instructions and is NEVER auto-deleted. The two-process end-to-end race lives in
``tests/integration/test_sentinel_process_lock.py``.
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


def test_stale_lock_refuses_with_operator_instructions(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_LOCK_STALE_S", "5")
    lockfile = tmp_path / "s.sha256.lock"
    lockfile.touch()
    crashed_at = time.time() - 60  # older than the stale threshold -> presumed crashed
    os.utime(lockfile, (crashed_at, crashed_at))
    with pytest.raises(RuntimeError, match="STALE"), send_lock(tmp_path / "s.sha256"):
        pytest.fail("must not enter the critical section")  # pragma: no cover
    assert lockfile.exists()  # NEVER auto-deleted — the operator removes it consciously


def test_env_seconds_ignores_garbage_and_keeps_the_default(monkeypatch):
    monkeypatch.setenv("SENTINEL_LOCK_TIMEOUT_S", "not-a-number")
    assert _env_seconds("SENTINEL_LOCK_TIMEOUT_S", 10.0) == 10.0
    monkeypatch.setenv("SENTINEL_LOCK_TIMEOUT_S", "2.5")
    assert _env_seconds("SENTINEL_LOCK_TIMEOUT_S", 10.0) == 2.5
