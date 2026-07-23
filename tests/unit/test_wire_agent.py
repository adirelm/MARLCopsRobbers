"""HTTP behaviour of the our-side §9 wire agent (partner brief §2 endpoints).

Real stdlib ThreadingHTTPServer on an ephemeral localhost port, driven with httpx:
unauthenticated /health, 401 on wrong/missing bearer, exact action-string mapping,
(session_id, tick) idempotency without re-advancing the policy, full per-session
reset on a re-POSTed new_sub_game (void replay), tick-sequence + stale-session
guards, and a real RecurrentPolicy smoke over the wire.
"""

from __future__ import annotations

import httpx
import pytest

from src.marl.env.actions import Action
from src.mcp.wire_agent import make_wire_agent
from src.sdk.sdk import MarlSDK

_TOKEN = "wire-test-bearer"  # test-injection only; real tokens live in .env
_AUTH = {"Authorization": f"Bearer {_TOKEN}"}


class StubPolicy:
    """act()+reset() double: scripted actions + call recording (no torch)."""

    def __init__(self, script):
        self.script, self.calls, self.resets = list(script), [], 0

    def reset(self):
        self.resets += 1

    def act(self, obs_list, legal_masks, epsilon, rng, state=None):
        self.calls.append({"obs": obs_list[0], "mask": legal_masks[0], "epsilon": epsilon})
        return [self.script[(len(self.calls) - 1) % len(self.script)]]


def _new(role="cop", sid="sg-0"):
    return {"session_id": sid, "grid": [5, 5], "your_role": role, "your_pos": [2, 0], "max_moves": 25}


def _move(tick, sid="sg-0", opponent=None, barriers=(), left=5, pos=(2, 2)):  # noqa: PLR0913 — payload fields
    return {
        "session_id": sid,
        "tick": tick,
        "your_pos": list(pos),
        "opponent_pos": opponent,
        "barriers": [list(b) for b in barriers],
        "barriers_left": left,
    }


@pytest.fixture
def served(cfg):
    """A running cop wire agent + its stub policy + an httpx client."""
    stub = StubPolicy([Action.UP, Action.PLACE_BARRIER, Action.LEFT, Action.DOWN, Action.RIGHT])
    agent = make_wire_agent(cfg, "cop", stub, _TOKEN, port=0)
    with httpx.Client(base_url=f"http://127.0.0.1:{agent.port}") as client:
        yield client, stub
    agent.close()


def test_health_is_unauthenticated_and_unknown_paths_404(served):
    client, _ = served
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/nope").status_code == 404
    assert client.post("/nope", json={}, headers=_AUTH).status_code == 404


def test_posts_reject_wrong_or_missing_token(served):
    client, stub = served
    for headers in ({}, {"Authorization": "Bearer wrong"}):
        assert client.post("/new_sub_game", json=_new(), headers=headers).status_code == 401
        assert client.post("/request_move", json=_move(0), headers=headers).status_code == 401
    assert stub.calls == [] and stub.resets == 0


def test_new_sub_game_rejects_other_role_and_bad_json(served):
    client, _ = served
    assert client.post("/new_sub_game", json=_new(role="thief"), headers=_AUTH).status_code == 400
    assert client.post("/request_move", content=b"not json", headers=_AUTH).status_code == 400


def test_action_ints_map_to_brief_strings(served):
    client, stub = served
    assert client.post("/new_sub_game", json=_new(), headers=_AUTH).json() == {"ok": True}
    got = [client.post("/request_move", json=_move(t), headers=_AUTH).json()["action"] for t in range(5)]
    assert got == ["up", "place_barrier", "left", "down", "right"]
    first = stub.calls[0]
    assert first["obs"]["image"].shape == (5, 5, 5) and first["obs"]["scalars"].shape == (6,)
    assert len(first["mask"]) == 5 and first["epsilon"] == 0.0


def test_idempotent_re_post_returns_cached_action_without_re_acting(served):
    client, stub = served
    client.post("/new_sub_game", json=_new(), headers=_AUTH)
    a0 = client.post("/request_move", json=_move(0), headers=_AUTH).json()
    assert client.post("/request_move", json=_move(0), headers=_AUTH).json() == a0
    assert len(stub.calls) == 1  # the retry never re-advanced the policy


def test_tick_gap_and_unknown_session_are_rejected(served):
    client, _ = served
    client.post("/new_sub_game", json=_new(), headers=_AUTH)
    assert client.post("/request_move", json=_move(0), headers=_AUTH).status_code == 200
    assert client.post("/request_move", json=_move(2), headers=_AUTH).status_code == 400
    assert client.post("/request_move", json=_move(0, sid="sg-9"), headers=_AUTH).status_code == 404


def test_void_replay_resets_session_cache_and_policy(served):
    client, stub = served
    client.post("/new_sub_game", json=_new(), headers=_AUTH)
    assert client.post("/request_move", json=_move(0), headers=_AUTH).json()["action"] == "up"
    client.post("/new_sub_game", json=_new(), headers=_AUTH)  # SAME session_id: void replay
    replay = client.post("/request_move", json=_move(0), headers=_AUTH).json()["action"]
    assert replay == "place_barrier" and len(stub.calls) == 2  # recomputed, not cached
    assert stub.resets == 2  # fresh hidden state per new_sub_game


def test_stale_session_replays_cache_but_rejects_fresh_ticks(served):
    client, _ = served
    client.post("/new_sub_game", json=_new(sid="sg-0"), headers=_AUTH)
    a0 = client.post("/request_move", json=_move(0, sid="sg-0"), headers=_AUTH).json()
    client.post("/new_sub_game", json=_new(sid="sg-1"), headers=_AUTH)
    assert client.post("/request_move", json=_move(0, sid="sg-0"), headers=_AUTH).json() == a0
    assert client.post("/request_move", json=_move(1, sid="sg-0"), headers=_AUTH).status_code == 409


def test_thief_server_serves_thief_role(cfg):
    stub = StubPolicy([Action.RIGHT])
    agent = make_wire_agent(cfg, "thief", stub, _TOKEN, port=0)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{agent.port}") as client:
            assert client.post("/new_sub_game", json=_new(role="thief"), headers=_AUTH).json() == {"ok": True}
            body = client.post("/request_move", json=_move(0, opponent=[2, 4], left=3), headers=_AUTH)
            assert body.json() == {"action": "right"}
            assert bool(stub.calls[0]["mask"][int(Action.PLACE_BARRIER)]) is False  # thief: no barrier slot
    finally:
        agent.close()


def test_real_recurrent_policy_over_the_wire(cfg):
    sdk = MarlSDK(cfg)
    agent = make_wire_agent(cfg, "cop", sdk.build_policy("cop", sdk.fresh_net("cop", 1)), _TOKEN, port=0)
    legal = {"up", "down", "left", "right", "place_barrier"}
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{agent.port}") as client:
            assert client.post("/new_sub_game", json=_new(), headers=_AUTH).json() == {"ok": True}
            acts = [client.post("/request_move", json=_move(t), headers=_AUTH).json() for t in range(3)]
            assert all(a["action"] in legal for a in acts)  # hidden state advanced 3 ticks, no error
            assert client.post("/request_move", json=_move(2), headers=_AUTH).json() == acts[2]
    finally:
        agent.close()
