"""Tests for the AgentController per-session hidden state + the auth verifier (T5.4/T5.5).

The controller holds one recurrent hidden stream per session_id (reset on
new_session), acts greedily, and never leaks policy internals; build_verifier maps
a role bearer token to its scoped claims. torch seeded.
"""

from __future__ import annotations

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
        _controller(cfg).act("nope", image, scalars, legal)


def test_act_returns_a_legal_action_int(cfg):
    """A started session returns a plain legal action int (no internals leaked)."""
    ctrl = _controller(cfg)
    ctrl.new_session("s1")
    action = ctrl.act("s1", *_obs(cfg))
    assert isinstance(action, int)
    assert 0 <= action < cfg["env"]["actions"]["a_cop"]


def test_new_session_resets_hidden_stream(cfg):
    """new_session replaces the session's policy -> a fresh z_0 (new_sub_game reset)."""
    ctrl = _controller(cfg)
    ctrl.new_session("s1")
    ctrl.act("s1", *_obs(cfg))  # advance z_t
    before = ctrl._sessions["s1"]
    ctrl.new_session("s1")
    assert ctrl._sessions["s1"] is not before


def test_end_session_drops_hidden(cfg):
    """end_session removes the session's hidden state (end_sub_game)."""
    ctrl = _controller(cfg)
    ctrl.new_session("s1")
    assert ctrl.has_session("s1")
    ctrl.end_session("s1")
    assert not ctrl.has_session("s1")


def test_build_verifier_returns_static_verifier(cfg):
    """build_verifier with an explicit token yields a StaticTokenVerifier."""
    assert isinstance(build_verifier(cfg, "cop", token="dev-cop-token"), StaticTokenVerifier)


def test_build_verifier_requires_a_token(cfg, monkeypatch):
    """No explicit token and no env var -> a clear ValueError."""
    monkeypatch.delenv("COP_MCP_TOKEN", raising=False)
    with pytest.raises(ValueError, match="token"):
        build_verifier(cfg, "cop", token=None)
