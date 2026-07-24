"""Tests for the AgentController per-session hidden state + the auth verifier (T5.4/T5.5).

The controller holds one recurrent hidden stream per session_id (reset on
new_session), acts greedily, and never leaks policy internals; build_verifier maps
a role bearer token to its scoped claims. torch seeded.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
import torch
from fastmcp.server.auth import StaticTokenVerifier

from src.marl.nets.agent_net import RecurrentQNet
from src.mcp.agent_runtime import AgentController
from src.mcp.auth import build_verifier
from src.sdk.sdk import MarlSDK

SEED = 7


def _controller(cfg) -> AgentController:
    """Build a cop AgentController over a fresh seeded cop net."""
    torch.manual_seed(SEED)
    return AgentController(MarlSDK(cfg), "cop", RecurrentQNet(cfg, "cop", 2), n_agents=1)


def _obs(cfg):
    """Return a zero (image, scalars, legal_mask) move payload for the cop."""
    c, w = cfg["env"]["obs_channels"], 2 * cfg["env"]["view_radius_max"] + 1
    image = np.zeros((c, w, w), np.float32).tolist()
    scalars = np.zeros(cfg["env"]["obs_scalars"], np.float32).tolist()
    legal = [True] * cfg["env"]["actions"]["a_cop"]
    return image, scalars, legal


def test_act_requires_a_started_session(cfg):
    """act on an unknown session raises (new_sub_game must precede request_move)."""
    image, scalars, legal = _obs(cfg)
    with pytest.raises(KeyError, match="session"):
        _controller(cfg).act("nope", 0, image, scalars, legal)


def test_act_returns_a_legal_action_int(cfg):
    """A started session returns a plain legal action int (no internals leaked)."""
    ctrl = _controller(cfg)
    ctrl.new_session("s1")
    action = ctrl.act("s1", 0, *_obs(cfg))
    assert isinstance(action, int)
    assert 0 <= action < cfg["env"]["actions"]["a_cop"]


def test_act_is_idempotent_on_retried_tick(cfg):
    """A retried (session, tick) returns the cached action and does NOT re-advance z_t."""
    ctrl = _controller(cfg)
    ctrl.new_session("s1")
    image, scalars, legal = _obs(cfg)
    first = ctrl.act("s1", 0, image, scalars, legal)
    hidden_after_first = ctrl._sessions["s1"]["policy"]._hidden.clone()
    retried = ctrl.act("s1", 0, image, scalars, legal)  # same tick -> cached, no re-advance
    assert retried == first
    assert torch.equal(ctrl._sessions["s1"]["policy"]._hidden, hidden_after_first)


def test_act_rejects_a_non_sequential_tick(cfg):
    """An uncached tick that is not exactly last+1 (a gap OR a regress) is rejected."""
    ctrl = _controller(cfg)
    ctrl.new_session("s1")
    image, scalars, legal = _obs(cfg)
    with pytest.raises(ValueError, match="breaks the sequence"):
        ctrl.act("s1", 5, image, scalars, legal)  # gap from a fresh session (last=-1, expected 0)
    ctrl.act("s1", 0, image, scalars, legal)
    ctrl.act("s1", 1, image, scalars, legal)
    with pytest.raises(ValueError, match="breaks the sequence"):
        ctrl.act("s1", 3, image, scalars, legal)  # gap forward (last=1, expected 2)


def test_new_session_resets_hidden_stream(cfg):
    """new_session replaces the session's policy -> a fresh z_0 (new_sub_game reset)."""
    ctrl = _controller(cfg)
    ctrl.new_session("s1")
    ctrl.act("s1", 0, *_obs(cfg))  # advance z_t
    before = ctrl._sessions["s1"]
    ctrl.new_session("s1")
    assert ctrl._sessions["s1"] is not before


class _CountingSDK:
    """Build a policy whose act() counts invocations and returns a distinct action each call."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def build_policy(self, role, net, n):
        sdk = self

        class _Policy:
            def act(self, obs_list, mask_list, eps, rng):
                sdk.calls.append(1)
                return [len(sdk.calls) - 1]  # distinct per real advance -> reveals a double-advance

        return _Policy()


def test_concurrent_retry_of_same_tick_advances_gru_exactly_once():
    """Two threads racing the SAME (session, tick) advance z_t once and share one cached action.

    No sleeps: a barrier releases both threads into act() together. The per-session lock
    serializes the check->advance->commit transaction, so the second thread hits the cache
    instead of re-running the net (the unlocked code double-advanced — see probe_c1).
    """
    sdk = _CountingSDK()
    ctrl = AgentController(sdk, "cop", object(), n_agents=1)
    ctrl.new_session("s1")
    args = ("s1", 0, [[[0.0]]], [0.0], [True])
    barrier = threading.Barrier(2, timeout=5)
    out: dict[int, int] = {}

    def worker(i: int) -> None:
        barrier.wait()
        out[i] = ctrl.act(*args)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert len(sdk.calls) == 1  # the GRU advanced exactly once for the retried tick
    assert out[0] == out[1] == 0  # both callers observe the same committed action


def test_build_verifier_returns_static_verifier(cfg):
    """build_verifier with an explicit token yields a StaticTokenVerifier."""
    assert isinstance(build_verifier(cfg, "cop", token="dev-cop-token"), StaticTokenVerifier)


def test_build_verifier_requires_a_token(cfg, monkeypatch):
    """No explicit token and no env var -> a clear ValueError."""
    monkeypatch.delenv("COP_MCP_TOKEN", raising=False)
    with pytest.raises(ValueError, match="token"):
        build_verifier(cfg, "cop", token=None)
