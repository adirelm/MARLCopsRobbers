"""RED->GREEN tests for src.mcp.wire_client — P8 fault policy + idempotent re-POST.

The egress callables are injected fakes (the sanctioned ``src.api.http_client`` wrappers
are the defaults and carry their own tests) — no sockets, no monkeypatching.
"""

from __future__ import annotations

import time

import pytest

from src.mcp.wire_client import VoidSubGame, WireClient
from tests.unit._wire_fixtures import FakeResp


def _client(post_fn=None, get_fn=None, **kw) -> WireClient:
    kw.setdefault("timeout_s", 10.0)
    kw.setdefault("retries", 1)
    fns = {}
    if post_fn is not None:
        fns["post_fn"] = post_fn
    if get_fn is not None:
        fns["get_fn"] = get_fn
    kw.setdefault("max_inflight", 8)  # explicit: WireClient no longer carries a default cap
    return WireClient("http://partner.test", "tok-abc", **fns, **kw)


def test_bearer_token_and_urls_reach_the_egress_wrapper():
    seen = []

    def post_fn(url, token, payload, timeout):
        seen.append((url, token, payload, timeout))
        return FakeResp(200, {"ok": True})

    def get_fn(url, token, timeout):
        seen.append((url, token, None, timeout))
        return FakeResp(200, {"status": "ok"})

    client = _client(post_fn, get_fn)
    assert client.health() is True
    assert client.new_sub_game({"session_id": "sg-0"}) == {"ok": True}
    assert seen[0] == ("http://partner.test/health", "tok-abc", None, 10.0)
    assert seen[1] == ("http://partner.test/new_sub_game", "tok-abc", {"session_id": "sg-0"}, 10.0)


def test_timeout_retries_once_with_the_same_body():
    bodies = []

    def post_fn(url, token, payload, timeout):
        bodies.append(dict(payload))
        if len(bodies) == 1:
            raise TimeoutError("slow")  # first attempt burns the P8 budget
        return FakeResp(200, {"action": "up"})

    assert _client(post_fn).request_move({"session_id": "sg-0", "tick": 3}) == {"action": "up"}
    assert len(bodies) == 2 and bodies[0] == bodies[1]  # idempotent re-POST, identical body


def test_two_transport_faults_raise_void_sub_game():
    calls = {"n": 0}

    def post_fn(url, token, payload, timeout):
        calls["n"] += 1
        raise ConnectionError("down")

    with pytest.raises(VoidSubGame):
        _client(post_fn).request_move({"tick": 0})
    assert calls["n"] == 2  # exactly ONE retry (wire_match.retries), then void


def test_http_error_status_counts_as_fault():
    def post_fn(url, token, payload, timeout):
        return FakeResp(401, {})

    with pytest.raises(VoidSubGame):
        _client(post_fn).new_sub_game({"session_id": "sg-1"})


def test_non_json_reply_counts_as_fault():
    def post_fn(url, token, payload, timeout):
        return FakeResp(200, None)  # body=None -> json() raises

    with pytest.raises(VoidSubGame):
        _client(post_fn).request_move({"tick": 0})


def test_reply_landing_past_the_wall_clock_budget_is_a_timeout_fault():
    # F5: httpx's scalar timeout is only PER-PHASE — a dribbled reply can land "successfully"
    # long after the promised 10 s window. The client must enforce the WALL CLOCK itself.
    calls = {"n": 0}

    def slow_then_fast(url, token, payload, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            time.sleep(0.08)  # a valid 200 that lands past the budget
        return FakeResp(200, {"action": "up"})

    assert _client(slow_then_fast, timeout_s=0.05).request_move({"tick": 0}) == {"action": "up"}
    assert calls["n"] == 2  # the late reply was discarded as a P8 timeout fault; the retry answered


def test_every_reply_past_the_wall_clock_voids_the_sub_game():
    def always_slow(url, token, payload, timeout):
        time.sleep(0.03)
        return FakeResp(200, {"action": "up"})

    with pytest.raises(VoidSubGame, match="wall clock"):
        _client(always_slow, timeout_s=0.01).request_move({"tick": 0})


def test_on_event_hook_sees_request_then_response_with_latency():
    events = []

    def post_fn(url, token, payload, timeout):
        return FakeResp(200, {"ok": True})

    client = _client(post_fn, label="group_2-cop", on_event=events.append)
    client.new_sub_game({"session_id": "sg-0"})
    assert [e["direction"] for e in events] == ["request", "response"]
    assert events[0]["label"] == "group_2-cop" and "/new_sub_game" in events[0]["url"]
    assert events[0]["payload"] == {"session_id": "sg-0"}
    assert events[1]["response"] == {"ok": True} and events[1]["latency_ms"] >= 0


def test_error_events_are_emitted_for_every_faulted_attempt():
    events = []

    def post_fn(url, token, payload, timeout):
        raise ConnectionError("down")

    with pytest.raises(VoidSubGame):
        _client(post_fn, on_event=events.append).request_move({"tick": 0})
    assert [e["direction"] for e in events] == ["request", "error", "request", "error"]
    assert all(e["attempt"] == i // 2 for i, e in enumerate(events))


def test_health_false_on_transport_error_and_on_bad_status():
    def down(url, token, timeout):
        raise ConnectionError("down")

    def bad(url, token, timeout):
        return FakeResp(500, {})

    assert _client(get_fn=down).health() is False
    assert _client(get_fn=bad).health() is False
