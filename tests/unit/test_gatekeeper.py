"""Tests for the ApiGatekeeper egress governor (§5, T5.9).

Burst admits immediately; the FIFO overflow queue absorbs the excess WITHOUT a
crash and drains FIFO as tokens refill; a full queue rejects with an explicit
error; get_queue_status reports per-channel depth. The clock is injected.
"""

from __future__ import annotations

import json

import pytest

from src.api.gatekeeper import DEFERRED, ApiGatekeeper


def test_burst_admits_immediately_then_overflow_enqueues():
    """The first `burst` calls run now; the next is queued (no crash), not run."""
    clock = [0.0]
    gk = ApiGatekeeper(clock=lambda: clock[0])  # peer_mcp burst=10
    ran = []
    for i in range(10):
        gk.execute("peer_mcp", lambda i=i: ran.append(i))
    assert len(ran) == 10
    assert gk.get_queue_status()["peer_mcp"] == 0
    deferred = gk.execute("peer_mcp", lambda: ran.append(99))
    assert deferred is DEFERRED  # explicit deferral marker, not a silent None result
    assert gk.get_queue_status()["peer_mcp"] == 1
    assert 99 not in ran


def test_queue_drains_fifo_as_tokens_refill():
    """A queued call drains (FIFO, before fresh calls) once tokens refill."""
    clock = [0.0]
    gk = ApiGatekeeper(clock=lambda: clock[0])
    ran = []
    for _ in range(10):
        gk.execute("peer_mcp", lambda: None)
    gk.execute("peer_mcp", lambda: ran.append("queued"))
    assert gk.get_queue_status()["peer_mcp"] == 1
    clock[0] = 1.0  # +1s -> peer_mcp refills 120/60 = 2 tokens
    gk.execute("peer_mcp", lambda: ran.append("fresh"))
    assert ran == ["queued", "fresh"]
    assert gk.get_queue_status()["peer_mcp"] == 0


def test_full_queue_rejects_with_error(tmp_path):
    """Beyond max_queue the gatekeeper rejects with an explicit error (never crashes silently)."""
    spec = {"version": "1", "limits": {"gmail": {"per_minute": 5, "burst": 1}}, "max_queue": 2}
    path = tmp_path / "rl.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    gk = ApiGatekeeper(path=path, clock=lambda: 0.0)
    gk.execute("gmail", lambda: None)  # admitted (burst 1)
    gk.execute("gmail", lambda: None)  # queued (depth 1)
    gk.execute("gmail", lambda: None)  # queued (depth 2 == max_queue)
    assert gk.get_queue_status()["gmail"] == 2
    with pytest.raises(RuntimeError, match="queue full"):
        gk.execute("gmail", lambda: None)


def test_unknown_channel_raises():
    """An unconfigured channel is rejected."""
    gk = ApiGatekeeper(clock=lambda: 0.0)
    with pytest.raises(KeyError, match="channel"):
        gk.execute("nope", lambda: None)


def test_get_queue_status_reports_all_channels():
    """Queue status enumerates every configured channel at zero depth on start."""
    status = ApiGatekeeper(clock=lambda: 0.0).get_queue_status()
    assert set(status) == {"peer_mcp", "gmail", "prefect_deploy"}
    assert all(depth == 0 for depth in status.values())
