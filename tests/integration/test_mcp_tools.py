"""Canonical tool-set completion tests (T5.6/T6.1) — query_opponent + report.

query_opponent is evidence-only (no policy internals) and structurally CANNOT feed
request_move; send_final_report is cop-only (FR-MCP-8); the full seven canonical
tools are registered on the cop server. torch seeded.
"""

from __future__ import annotations

import asyncio
import inspect

import torch
from fastmcp import Client

from src.marl.nets.agent_net import RecurrentQNet
from src.mcp.agent_runtime import AgentController
from src.mcp.cop_server import make_cop_server
from src.mcp.thief_server import make_thief_server

SEED = 7
_CANONICAL = {
    "health",
    "ping",
    "new_sub_game",
    "request_move",
    "reveal_location",
    "query_opponent",
    "send_final_report",
}


def _call(server, tool, args):
    """One in-memory tool call -> result.data."""

    async def _run():
        async with Client(server) as client:
            return (await client.call_tool(tool, args)).data

    return asyncio.run(_run())


def _tool_names(server) -> set:
    """The registered tool names on a server (in-memory list_tools)."""

    async def _run():
        async with Client(server) as client:
            return {t.name for t in await client.list_tools()}

    return asyncio.run(_run())


def _cop(cfg, peer_query=None):
    torch.manual_seed(SEED)
    return make_cop_server(cfg, RecurrentQNet(cfg, "cop", 2), token="dev-cop", peer_query=peer_query)


def test_query_opponent_returns_evidence_only(cfg):
    """query_opponent returns visible/position evidence only — no Q-values/weights/z_t."""

    def peer(_role, _tick):
        return {"visible": True, "position": (1, 2)}

    data = _call(
        _cop(cfg, peer), "query_opponent", {"req": {"session_id": "s", "requester_role": "cop", "tick": 0}}
    )
    assert set(data) == {"visible", "position"}
    assert data["visible"] is True
    assert tuple(data["position"]) == (1, 2)


def test_query_opponent_without_peer_yields_no_evidence(cfg):
    """With no peer wired, query_opponent yields no evidence (visible False)."""
    data = _call(
        _cop(cfg), "query_opponent", {"req": {"session_id": "s", "requester_role": "cop", "tick": 0}}
    )
    assert data["visible"] is False


def test_send_final_report_is_cop_only_and_dry_runs(cfg):
    """send_final_report is on the cop only (FR-MCP-8) and dry-runs at P6."""
    torch.manual_seed(SEED)
    thief = make_thief_server(cfg, RecurrentQNet(cfg, "thief", 1), token="dev-thief")
    assert "send_final_report" not in _tool_names(thief)
    ack = _call(_cop(cfg), "send_final_report", {"report": {"sub_games": [], "totals": {}}})
    assert ack["sent"] is False
    assert ack["dry_run"] is True


def test_all_seven_canonical_tools_registered(cfg):
    """The cop server registers the full canonical tool contract."""
    assert _tool_names(_cop(cfg)) >= _CANONICAL


def test_query_evidence_cannot_enter_request_move():
    """Structural decoupling: AgentController.act takes only obs — no query evidence param."""
    params = set(inspect.signature(AgentController.act).parameters)
    assert params == {"self", "session_id", "image", "scalars", "legal_mask"}
