"""Wire-agent hardening: the cross-generation stale-request race + per-sub-game rng.

Race (was CRITICAL): a ThreadingHTTPServer thread can finish a STALE timed-out request
AFTER ``/new_sub_game`` reset the same session_id. The stale body lands first and
advances the freshly reset policy once, so the server snapshots (policy + rng) before
EVERY fresh advance and, on a differing body for the NEWEST tick, RESTORES the snapshot
before recomputing — the genuine request gets exactly the post-hello state, not a
once-poisoned one. A novel body for any OLDER (sealed) tick is a 409, so out-of-order
stale bodies can never bypass the sequential-tick rule. The race is reproduced
DETERMINISTICALLY (no sleeps): the stale request is parked mid-body on a raw socket, the
void re-hello lands, then the body's tail is released.
Rng: the acting rng is reseeded on every hello, so stochastic policies (the flee thief
pre-contact, the conformance uniform-random) redraw the SAME stream — identical
hello+tick sequences must produce identical actions across resets (void-replay fidelity).
"""

from __future__ import annotations

import json
import socket

import httpx

from src.marl.env.actions import Action
from src.mcp.wire_agent import make_wire_agent
from src.sdk.sdk import MarlSDK
from src.services.bonus_policies import AdaptiveThiefPolicy
from tests.unit.test_wire_agent import _AUTH, _TOKEN, StubPolicy, _move, _new


class DrawingPolicy:
    """Stochastic acting double: draws rng.choice exactly like flee-pre-contact/uniform."""

    def reset(self):
        pass

    def act(self, obs_list, legal_masks, epsilon, rng, state=None):
        return [rng.choice([i for i, ok in enumerate(legal_masks[0]) if ok])]


def test_stale_request_finishing_after_the_void_reset_cannot_poison_the_new_run(cfg):
    stub = StubPolicy([Action.UP, Action.LEFT, Action.DOWN])
    agent = make_wire_agent(cfg, "cop", stub, _TOKEN, port=0)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{agent.port}") as client:
            client.post("/new_sub_game", json=_new(), headers=_AUTH)
            stale = json.dumps(_move(0, pos=(0, 0))).encode()  # the FIRST run's tick-0 body
            head = (
                f"POST /request_move HTTP/1.1\r\nHost: t\r\nAuthorization: Bearer {_TOKEN}\r\n"
                f"Content-Type: application/json\r\nContent-Length: {len(stale)}\r\n\r\n"
            ).encode()
            with socket.create_connection(("127.0.0.1", agent.port)) as sock:
                sock.sendall(head + stale[:5])  # park the stale request mid-body...
                client.post("/new_sub_game", json=_new(), headers=_AUTH)  # ...the void reset lands
                sock.sendall(stale[5:])  # ...then the stale request completes POST-reset
                assert sock.recv(1)  # any reply byte => its serve_move fully finished
            fresh = client.post("/request_move", json=_move(0, pos=(4, 4)), headers=_AUTH).json()
            again = client.post("/request_move", json=_move(0, pos=(4, 4)), headers=_AUTH).json()
            nxt = client.post("/request_move", json=_move(1, pos=(4, 3)), headers=_AUTH).json()
    finally:
        agent.close()
    # The stale advance was REWOUND: the genuine tick 0 sees the post-hello state and gets
    # the FIRST scripted action (not the poisoned second), and tick 1 continues from there.
    assert fresh == {"action": "up"}
    assert again == fresh  # identical re-POST replays the cache without re-advancing
    assert nxt == {"action": "left"}  # tick 1 = SECOND action: exactly one effective advance before it
    assert len(stub.calls) == 1  # the original object saw only the stale advance; its restored
    #   pre-advance deepcopy served the genuine recompute (state continuity proven by `nxt`)


def test_novel_body_for_an_older_sealed_tick_is_rejected_409(cfg):
    stub = StubPolicy([Action.UP, Action.LEFT, Action.DOWN, Action.RIGHT])
    agent = make_wire_agent(cfg, "cop", stub, _TOKEN, port=0)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{agent.port}") as client:
            client.post("/new_sub_game", json=_new(), headers=_AUTH)
            a0 = client.post("/request_move", json=_move(0), headers=_AUTH).json()
            client.post("/request_move", json=_move(1), headers=_AUTH)
            novel_old = client.post("/request_move", json=_move(0, pos=(3, 3)), headers=_AUTH)
            assert novel_old.status_code == 409  # sealed tick: novel bodies recompute ONLY the newest
            assert client.post("/request_move", json=_move(0), headers=_AUTH).json() == a0  # cache intact
            assert client.post("/request_move", json=_move(2), headers=_AUTH).status_code == 200
    finally:
        agent.close()
    assert len(stub.calls) == 3  # ticks 0/1/2 advanced once each; the 409 advanced NOTHING


def test_identical_hello_and_ticks_redraw_identical_actions_for_a_stochastic_policy(cfg):
    agent = make_wire_agent(cfg, "cop", DrawingPolicy(), _TOKEN, port=0)
    runs = []
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{agent.port}") as client:
            for _ in range(3):
                client.post("/new_sub_game", json=_new(), headers=_AUTH)
                runs.append(
                    [
                        client.post("/request_move", json=_move(t), headers=_AUTH).json()["action"]
                        for t in range(4)
                    ]
                )
    finally:
        agent.close()
    assert runs[0] == runs[1] == runs[2]  # a void replay reproduces the first attempt exactly


def _steps_since_seen(agent) -> int:
    """Read the internal wire visibility counter for sub-game sg-0."""
    return agent._server._sessions["sg-0"]["wire"]["steps_since_seen"]


def test_a_newest_tick_recompute_does_not_double_advance_the_visibility_counter(cfg):
    """W1: the snapshot restore rewinds the wire ``steps_since_seen`` too, not just policy+rng.

    A novel body for the already-answered newest tick recomputes; without rewinding the wire
    state the visibility counter advances TWICE (once per build_observation) instead of once,
    diverging from a clean single request for the same tick.
    """
    agent = make_wire_agent(cfg, "cop", StubPolicy([Action.UP, Action.LEFT]), _TOKEN, port=0)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{agent.port}") as client:
            client.post("/new_sub_game", json=_new(), headers=_AUTH)
            client.post("/request_move", json=_move(0, pos=(0, 0)), headers=_AUTH)  # first body
            client.post("/request_move", json=_move(0, pos=(4, 4)), headers=_AUTH)  # novel newest-tick body
            recompute = _steps_since_seen(agent)
    finally:
        agent.close()
    clean = make_wire_agent(cfg, "cop", StubPolicy([Action.UP]), _TOKEN, port=0)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{clean.port}") as client:
            client.post("/new_sub_game", json=_new(), headers=_AUTH)
            client.post("/request_move", json=_move(0, pos=(4, 4)), headers=_AUTH)  # one clean request
            single = _steps_since_seen(clean)
    finally:
        clean.close()
    assert recompute == single == 1  # the stale advance was rewound, not compounded onto the recompute


def test_a_recompute_preserves_a_barrier_triggered_sticky_switch(cfg):
    """W2: a fired AdaptiveThiefPolicy switch is match-sticky and MUST survive the snapshot rewind.

    The pre-advance deepcopy captured ``switched=False``; restoring it verbatim would erase a
    barrier sighting, contradicting the deliberate match-level stickiness (bonus_policies docstring
    + ANALYSIS §12). The restore re-applies the monotonic switch instead.
    """
    net = MarlSDK(cfg).fresh_net("thief")
    agent = make_wire_agent(cfg, "thief", AdaptiveThiefPolicy(cfg, net), _TOKEN, port=0)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{agent.port}") as client:
            client.post("/new_sub_game", json=_new(role="thief"), headers=_AUTH)
            client.post("/request_move", json=_move(0, barriers=[(2, 1)], left=4), headers=_AUTH)  # trips it
            assert agent._server.policy.switched is True
            client.post("/request_move", json=_move(0, barriers=(), left=5, pos=(2, 3)), headers=_AUTH)
            assert agent._server.policy.switched is True  # the sticky switch survived the rewind
    finally:
        agent.close()


def test_sessions_are_lru_capped_so_authenticated_hellos_cannot_grow_memory(cfg):
    """W3: past ``wire_agent.max_sessions`` hellos evict the oldest session (bounded snapshots)."""
    cap = int(cfg["wire_agent"]["max_sessions"])
    agent = make_wire_agent(cfg, "cop", StubPolicy([Action.UP]), _TOKEN, port=0)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{agent.port}") as client:
            for i in range(cap + 5):
                client.post("/new_sub_game", json=_new(sid=f"sg-{i}"), headers=_AUTH)
            live = set(agent._server._sessions)
    finally:
        agent.close()
    assert len(live) == cap  # never grows past the cap
    assert "sg-0" not in live and f"sg-{cap + 4}" in live  # oldest evicted, newest kept
