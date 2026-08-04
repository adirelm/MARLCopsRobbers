"""RED->GREEN tests for P8 protocol faults, P3 barrier semantics, and the wire E2E path."""

from __future__ import annotations

import json

import pytest

from src.mcp.wire_client import VoidSubGame, WireClient
from src.mcp.wire_referee import WireReferee
from tests.unit._wire_fixtures import SEEDS, FakeResp, StubAgent, stub_clients, wire_cfg


def _referee(tmp_path) -> WireReferee:
    return WireReferee(wire_cfg(), tmp_path / "log.jsonl")


def test_malformed_action_voids_after_one_protocol_retry(tmp_path):
    g1 = StubAgent(actions={"cop": "fly", "thief": "down"})  # unknown action string
    g2 = StubAgent()
    with pytest.raises(VoidSubGame):
        _referee(tmp_path)._play_sub_game(stub_clients(g1, g2), 1, SEEDS[0])
    assert len(g1.move_calls) == 2  # P8: exactly ONE re-ask of the same tick, then void


def test_thief_place_barrier_is_a_fault(tmp_path):
    g1 = StubAgent()
    g2 = StubAgent(actions={"cop": "up", "thief": "place_barrier"})  # cop-only action
    with pytest.raises(VoidSubGame):
        _referee(tmp_path)._play_sub_game(stub_clients(g1, g2), 1, SEEDS[0])  # game 1: g2 = thief
    assert len(g2.move_calls) == 2


def test_new_sub_game_without_ok_true_is_a_fault(tmp_path):
    class NoAck(StubAgent):
        def new_sub_game(self, payload):
            super().new_sub_game(payload)
            return {"ok": False}

    bad = NoAck()
    with pytest.raises(VoidSubGame):
        _referee(tmp_path)._play_sub_game(stub_clients(bad, StubAgent()), 1, SEEDS[0])
    assert len(bad.new_calls) == 2


def test_cop_place_barrier_budget_and_masked_barrier_lists(tmp_path):
    def cop_policy(payload):
        return "place_barrier" if payload["tick"] < 2 else "up"

    g1 = StubAgent(actions={"cop": cop_policy, "thief": "down"})
    g2 = StubAgent()
    record = _referee(tmp_path)._play_sub_game(stub_clients(g1, g2), 1, SEEDS[0])
    cop_calls = [c for c in g1.move_calls if c["session_id"] == "sg-0"]
    # P3: the tick-1 re-place on the cop's own (already-barrier) cell consumes NO budget.
    assert [c["barriers_left"] for c in cop_calls[:3]] == [5, 4, 4]
    assert cop_calls[1]["barriers"] == [cop_calls[0]["your_pos"]]  # own cell is within radius
    for call in g1.move_calls + g2.move_calls:  # P5: every listed barrier is within radius 2
        for cell in call["barriers"]:
            dist = abs(cell[0] - call["your_pos"][0]) + abs(cell[1] - call["your_pos"][1])
            assert dist <= 2
    assert record["winner"] == "thief" and record["moves"] == 25


def test_end_to_end_one_sub_game_through_wire_clients(tmp_path):
    referee = _referee(tmp_path)

    def make_post_fn():
        roles: dict[str, str] = {}

        def post_fn(url, token, payload, timeout):
            assert token == "tok" and timeout == 10.0
            if url.endswith("/new_sub_game"):
                roles[payload["session_id"]] = payload["your_role"]
                return FakeResp(200, {"ok": True})
            action = "up" if roles[payload["session_id"]] == "cop" else "down"
            return FakeResp(200, {"action": action})

        return post_fn

    def client(label: str) -> WireClient:
        return WireClient(
            "http://partner.test",
            "tok",
            timeout_s=10.0,
            retries=1,
            max_inflight=8,
            label=label,
            post_fn=make_post_fn(),
            on_event=referee.log_event,
        )

    clients = {k: {r: client(f"{k}-{r}") for r in ("cop", "thief")} for k in ("group_1", "group_2")}
    record = referee._play_sub_game(clients, 1, SEEDS[0])
    assert record["winner"] == "thief" and record["moves"] == 25 and record["session_id"] == "sg-0"
    entries = [json.loads(x) for x in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    requests = [e for e in entries if e["direction"] == "request"]
    responses = [e for e in entries if e["direction"] == "response"]
    assert len(requests) == len(responses) == 52  # 2 new_sub_game + 2 roles x 25 ticks
    assert all("latency_ms" in e and "url" in e and "ts" in e for e in responses)
    assert {e["label"] for e in requests} == {"group_1-cop", "group_2-thief"}  # game 1 roles
