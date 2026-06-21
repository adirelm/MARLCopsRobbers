"""Dual MCP server contract tests (T5.6) via the FastMCP in-memory Client.

Drives the canonical tools end to end against an in-memory server: health handshake,
new_sub_game -> request_move returns a legal action (no value/logit/hidden), a
global_state leak is rejected at the protocol edge, reveal_location is radius-gated,
and neither server module imports the other's policy. torch seeded.
"""

from __future__ import annotations

import asyncio
import inspect

import numpy as np
import pytest
import torch
from fastmcp import Client

from src.marl.nets.agent_net import RecurrentQNet
from src.mcp import cop_server, thief_server
from src.mcp.cop_server import make_cop_server
from src.mcp.server_builder import _require
from src.mcp.thief_server import make_thief_server

SEED = 7


def _call(server, tool: str, args: dict):
    """Run one in-memory tool call synchronously and return result.data."""

    async def _run():
        async with Client(server) as client:
            return (await client.call_tool(tool, args)).data

    return asyncio.run(_run())


def _cop_server(cfg):
    """Build a cop server over a fresh seeded cop net + a dev token."""
    torch.manual_seed(SEED)
    return make_cop_server(cfg, RecurrentQNet(cfg, "cop", 2), token="dev-cop")


def _move_args(cfg, session: str = "s1") -> dict:
    """A zero-obs request_move payload for the cop."""
    c, w = cfg["env"]["obs_channels"], 2 * cfg["env"]["view_radius_max"] + 1
    return {
        "req": {
            "session_id": session,
            "tick": 0,
            "image": np.zeros((c, w, w), np.float32).tolist(),
            "scalars": np.zeros(cfg["env"]["obs_scalars"], np.float32).tolist(),
            "legal_mask": [True] * cfg["env"]["actions"]["a_cop"],
        }
    }


def test_health_returns_protocol_version(cfg):
    """health() reports ok + the configured cross-server protocol version."""
    data = _call(_cop_server(cfg), "health", {})
    assert data["status"] == "ok"
    assert data["protocol_version"] == cfg["mcp"]["protocol_version"]


def test_request_move_after_new_sub_game_returns_only_action(cfg):
    """new_sub_game then request_move yields ONLY a legal action (no internals leak)."""
    server = _cop_server(cfg)
    _call(server, "new_sub_game", {"req": {"session_id": "s1", "grid": [2, 2]}})
    data = _call(server, "request_move", _move_args(cfg))
    assert set(data) == {"action"}
    assert 0 <= data["action"] < cfg["env"]["actions"]["a_cop"]


def test_request_move_rejects_global_state_leak(cfg):
    """A global_state field on request_move is rejected at the protocol edge."""
    server = _cop_server(cfg)
    _call(server, "new_sub_game", {"req": {"session_id": "s1", "grid": [2, 2]}})
    args = _move_args(cfg)
    args["req"]["global_state"] = [1, 2, 3]
    with pytest.raises(Exception, match=r"(?i)forbid|extra|global_state"):
        _call(server, "request_move", args)


def test_reveal_location_is_radius_gated(cfg):
    """reveal_location returns the position only when the requester is within radius."""
    server = _cop_server(cfg)
    _call(server, "new_sub_game", {"req": {"session_id": "s1", "grid": [5, 5], "position": [0, 0]}})
    near = _call(
        server,
        "reveal_location",
        {"req": {"session_id": "s1", "requester": "thief", "requester_pos": [0, 1]}},
    )
    assert near["visible"] is True
    assert tuple(near["position"]) == (0, 0)
    far = _call(
        server,
        "reveal_location",
        {"req": {"session_id": "s1", "requester": "thief", "requester_pos": [4, 4]}},
    )
    assert far["visible"] is False
    assert far["position"] is None


def test_thief_server_pings(cfg):
    """The thief server builds independently and answers ping."""
    torch.manual_seed(SEED)
    server = make_thief_server(cfg, RecurrentQNet(cfg, "thief", 1), token="dev-thief")
    assert _call(server, "ping", {}) == "pong"


def test_servers_do_not_import_each_others_policy():
    """Neither server module references the other's module (decentralized boundary)."""
    assert "thief_server" not in inspect.getsource(cop_server)
    assert "cop_server" not in inspect.getsource(thief_server)


def test_require_input_guard_raises_type_error():
    """The §16 _require guard raises TypeError on a falsy field and passes otherwise."""
    with pytest.raises(TypeError, match="session_id"):
        _require("", "session_id must be non-empty")
    _require("ok", "should not raise")
