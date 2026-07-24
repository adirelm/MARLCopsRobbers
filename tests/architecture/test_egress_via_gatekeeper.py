"""Egress boundary: httpx is imported ONLY inside src/api (§5, T5.9 DoD).

ALL raw HTTP egress must go through src/api/http_client.py (routed via the
ApiGatekeeper). This test fails if any module outside src/api (and, once it exists,
src/reporting/mailer.py) imports httpx — preventing an ungoverned egress path.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import ClassVar

import pytest

from src.api import http_client
from src.api.gatekeeper import ApiGatekeeper
from src.mcp.clients import AgentClient

_SRC = Path(__file__).resolve().parents[2] / "src"
_ALLOWED_PARTS = {"api"}  # + reporting/mailer.py when the Gmail sender lands (P9)


def test_httpx_imported_only_inside_api():
    """No src module outside src/api imports httpx (the single sanctioned wrapper)."""
    offenders = []
    for py in _SRC.rglob("*.py"):
        if set(py.parts) & _ALLOWED_PARTS:
            continue
        if "import httpx" in py.read_text(encoding="utf-8"):
            offenders.append(str(py.relative_to(_SRC)))
    assert offenders == [], f"httpx imported outside src/api: {offenders}"


def test_bearer_get_sets_authorization_header(monkeypatch):
    """bearer_get attaches the bearer token (the only httpx GET wrapper)."""
    captured = {}

    def fake_get(url, headers, timeout):
        captured.update(url=url, headers=headers, timeout=timeout)
        return "resp"

    monkeypatch.setattr(http_client.httpx, "get", fake_get)
    assert http_client.bearer_get("http://x/mcp", "tok") == "resp"
    assert captured["headers"]["Authorization"] == "Bearer tok"


def test_bearer_post_sends_json_with_bearer(monkeypatch):
    """bearer_post attaches the bearer token + JSON body (the only httpx POST wrapper)."""
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured.update(headers=headers, json=json)
        return "resp"

    monkeypatch.setattr(http_client.httpx, "post", fake_post)
    assert http_client.bearer_post("http://x", "tok", {"a": 1}) == "resp"
    assert captured["headers"]["Authorization"] == "Bearer tok"
    assert captured["json"] == {"a": 1}


def test_gatekeeper_is_importable_as_single_entry():
    """The §5 gatekeeper exposes execute + get_queue_status (the single egress entry)."""
    assert hasattr(ApiGatekeeper, "execute")
    assert hasattr(ApiGatekeeper, "get_queue_status")


@pytest.mark.parametrize("symbol", ["bearer_get", "bearer_post"])
def test_http_client_exposes_bearer_wrappers(symbol):
    """The http_client exposes both bearer wrappers."""
    assert hasattr(http_client, symbol)


def test_peer_side_channel_is_gated_but_the_game_loop_is_not():
    """The §5 peer_mcp rate limit governs the spammable cop<->thief SIDE-CHANNEL only.

    Asserts ROUTING, not import location: a recording gatekeeper double must see a
    `peer_mcp` admission for `reveal_location` (the position-probe vector), and NONE for
    the bounded referee game loop (`health` / `new_sub_game` / `request_move`, ~300 calls
    per match — throttling those would spuriously void sub-games). This is the gate that
    would have caught the pre-2026-07 bypass, without breaking a fast local match.
    """

    class _RecordingGate:
        def __init__(self) -> None:
            self.channels: list[str] = []

        def execute(self, channel, call):
            self.channels.append(channel)
            return call()

    class _StubResult:
        data: ClassVar[dict] = {"status": "ok", "action": 0}

    class _StubClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def call_tool(self, tool, args):
            return _StubResult()

    gate = _RecordingGate()

    async def _run():
        async with AgentClient(_StubClient(), label="cop", gatekeeper=gate) as client:
            await client.health()  # game loop — NOT throttled
            await client.new_sub_game("sg-0", (5, 5), (0, 0))  # game loop — NOT throttled
            await client.request_move("sg-0", 0, [[0.0]], [0.0], [True])  # game loop — NOT throttled
            await client.reveal_location("sg-0", "thief", (1, 1))  # PEER probe — gated

    asyncio.run(_run())
    assert gate.channels == ["peer_mcp"]  # exactly one admission: the reveal_location probe


def test_peer_mcp_deferred_admission_hard_faults_without_egress():
    """A DEFERRED admission (backpressure) MUST hard-fault, never fire the real call (V3 §5).

    The pre-fix bug: the gate admitted a no-op thunk while the actual outbound call fired
    regardless — decorative under backpressure. Now a deferred peer-MCP admission raises and
    the underlying client's `call_tool` is never reached (zero ungoverned egress).
    """
    from src.api.gatekeeper import DEFERRED  # noqa: PLC0415 — local to the deferred-path assertion

    class _DeferGate:
        def execute(self, channel, call):
            return DEFERRED  # simulate a full/empty bucket: queue it, do NOT run the thunk

    class _CountingClient:
        def __init__(self) -> None:
            self.calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def call_tool(self, tool, args):
            self.calls += 1  # unreachable on the deferred path — asserted below

    client = _CountingClient()

    async def _run():
        async with AgentClient(client, max_retries=1, gatekeeper=_DeferGate()) as agent:
            await agent.reveal_location("sg-0", "thief", (1, 1))  # a PEER probe — the gated path

    with pytest.raises(RuntimeError, match="deferred by rate limiter"):
        asyncio.run(_run())
    assert client.calls == 0  # the outbound call NEVER fired despite the gate being invoked
