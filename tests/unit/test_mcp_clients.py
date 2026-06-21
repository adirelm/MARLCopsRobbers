"""Tests for the typed peer MCP client (T5.7): factory, retry, reveal.

Covers make_client (bearer HTTP), the bounded retry-then-reraise path on a failing
client, and a radius-gated reveal_location round-trip over an in-memory server.
"""

from __future__ import annotations

import asyncio

import pytest
import torch
from fastmcp import Client

from src.marl.nets.agent_net import RecurrentQNet
from src.mcp.clients import AgentClient, make_client
from src.mcp.cop_server import make_cop_server


class _Boom:
    """A fake client whose every call_tool raises — drives the retry/re-raise path."""

    def __init__(self) -> None:
        self.calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def call_tool(self, tool, args):
        self.calls += 1
        raise RuntimeError("boom")


def test_make_client_builds_a_bearer_http_client():
    """make_client returns a FastMCP Client for a remote URL + token."""
    assert isinstance(make_client("http://localhost:8001/mcp", "tok"), Client)


def test_agent_client_retries_then_reraises():
    """A persistently failing call is retried max_retries times, then re-raised."""
    boom = _Boom()

    async def _run():
        async with AgentClient(boom, max_retries=3) as client:
            await client.request_move("s", 0, [[[0.0]]], [0.0], [True])

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(_run())
    assert boom.calls == 3


def test_reveal_location_round_trips_radius_gated(cfg):
    """reveal_location via the client returns a radius-gated visibility result."""
    torch.manual_seed(7)
    server = make_cop_server(cfg, RecurrentQNet(cfg, "cop", 2), token="dev-cop")

    async def _run():
        async with AgentClient(Client(server)) as client:
            await client.new_sub_game("s", (5, 5), (0, 0))
            return await client.reveal_location("s", "thief", (0, 1))

    result = asyncio.run(_run())
    assert result["visible"] is True
    assert tuple(result["position"]) == (0, 0)
