"""Tests for the typed peer MCP client (T5.7): factory, retry, reveal.

Covers make_client (bearer HTTP), the bounded retry-then-reraise path on a failing
client, and a radius-gated reveal_location round-trip over an in-memory server.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

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


def test_agent_client_retries_with_backoff_then_reraises():
    """A persistently failing call is retried max_retries times (with backoff), then re-raised."""
    boom = _Boom()

    async def _run():
        # backoff_s>0 exercises the configured inter-retry sleep (tiny so the test stays fast)
        async with AgentClient(boom, max_retries=3, backoff_s=0.001) as client:
            await client.request_move("s", 0, [[[0.0]]], [0.0], [True])

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(_run())
    assert boom.calls == 3


def test_agent_client_times_out_a_hung_call():
    """A call exceeding timeout_s is aborted (then re-raised after retries), never blocks forever."""

    class _Hang(_Boom):
        async def call_tool(self, tool, args):
            self.calls += 1
            await asyncio.sleep(5)  # far longer than the timeout below -> wait_for fires

    hung = _Hang()

    async def _run():
        async with AgentClient(hung, max_retries=2, timeout_s=0.01) as client:
            await client.request_move("s", 0, [[[0.0]]], [0.0], [True])

    with pytest.raises(TimeoutError):
        asyncio.run(_run())
    assert hung.calls >= 1  # the hung call was attempted + timed out (not blocked)


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


def test_prewarm_absorbs_a_cold_start_that_would_exhaust_the_move_retries():
    """A sleeping free-tier container wakes LONG after timeout_s*max_retries (measured ~90 s).

    The warm-up must poll to its own deadline instead of raising — before this, a cold
    start aborted the whole cloud match before sub-game 1 while the runbook claimed the
    prewarm "absorbed" it.
    """

    class _WakesOnAttempt:
        """Fails more times than max_retries allows, then answers."""

        def __init__(self, wake_on: int) -> None:
            self.calls, self._wake_on = 0, wake_on

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def call_tool(self, tool, args):
            self.calls += 1
            if self.calls < self._wake_on:
                raise ConnectionError("container asleep")
            return SimpleNamespace(data={"status": "ok"})

    stub = _WakesOnAttempt(wake_on=12)  # 12 > max_retries(3): health() alone would give up

    async def _run():
        async with AgentClient(stub, max_retries=3, backoff_s=0.0) as client:
            return await client.prewarm(deadline_s=30.0)

    assert asyncio.run(_run()) is True
    assert stub.calls >= 12  # it kept polling past the per-move retry budget


def test_prewarm_returns_false_at_its_deadline_instead_of_raising():
    """An unreachable peer must surface as a health verdict, not a warm-up traceback."""

    class _NeverWakes:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def call_tool(self, tool, args):
            raise ConnectionError("down")

    async def _run():
        async with AgentClient(_NeverWakes(), max_retries=1, backoff_s=0.0) as client:
            return await client.prewarm(deadline_s=0.05)

    assert asyncio.run(_run()) is False
