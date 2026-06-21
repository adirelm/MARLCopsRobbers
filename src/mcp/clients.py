"""Typed peer MCP clients — bearer auth + retry + structured logging (T5.7).

``make_client`` builds a FastMCP HTTP client with bearer auth for a remote agent
server; ``AgentClient`` is the async typed wrapper the referee drives — one method
per canonical tool, each call bounded by ``mcp.client.max_retries`` and emitting a
structured log line (the §7.3d F4 evidence). For in-memory tests the wrapper takes
a ``Client(server)`` directly (no HTTP). It is an async context manager so a whole
sub-game reuses ONE connection.
"""

from __future__ import annotations

import logging

from fastmcp import Client
from fastmcp.client.auth import BearerAuth

_log = logging.getLogger("marl.mcp.client")


def make_client(url: str, token: str) -> Client:
    """Build a bearer-authed FastMCP HTTP client for a remote agent server."""
    return Client(url, auth=BearerAuth(token))


class AgentClient:
    """Async typed wrapper over a FastMCP ``Client`` for one agent server."""

    def __init__(self, client: Client, max_retries: int = 3, label: str = "peer") -> None:
        """Wrap a (HTTP or in-memory) client with a bounded retry budget + a log label."""
        self._client = client
        self._max_retries = max(1, int(max_retries))
        self._label = label

    async def __aenter__(self) -> AgentClient:
        """Open the underlying client connection (reused across the sub-game)."""
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Close the underlying client connection."""
        await self._client.__aexit__(*exc)

    async def _call(self, tool: str, args: dict) -> object:
        """Call ``tool`` with bounded retries + structured logging; return result data.

        The log line carries the client ``label`` (which server) + the ``trace`` (the
        session_id): a sub-game's cop AND thief calls share one trace — the §7.3d F4 proof.
        """
        req = args.get("req")
        trace = req.get("session_id", "-") if isinstance(req, dict) else "-"
        last: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                data = (await self._client.call_tool(tool, args)).data
                _log.info(
                    "mcp_call client=%s tool=%s trace=%s attempt=%d status=ok",
                    self._label,
                    tool,
                    trace,
                    attempt,
                )
                return data
            except Exception as exc:  # retried up to max_retries, then re-raised on exhaustion
                last = exc
                _log.warning(
                    "mcp_call client=%s tool=%s trace=%s attempt=%d status=err err=%s",
                    self._label,
                    tool,
                    trace,
                    attempt,
                    type(exc).__name__,
                )
        raise last  # type: ignore[misc]

    async def health(self) -> dict:
        """Liveness handshake (used by the orchestrator to absorb cold-start)."""
        return await self._call("health", {})

    async def new_sub_game(self, session_id: str, grid: tuple, position: tuple | None = None) -> dict:
        """Start/reset a sub-game session on the server (fresh hidden ``z_0``)."""
        pos = list(position) if position is not None else None
        req = {"session_id": session_id, "grid": list(grid), "position": pos}
        return await self._call("new_sub_game", {"req": req})

    async def request_move(
        self, session_id: str, tick: int, image: list, scalars: list, legal_mask: list
    ) -> int:
        """Send LOCAL obs for one tick; return the agent's chosen action int."""
        req = {"session_id": session_id, "tick": tick, "image": image}
        req.update(scalars=scalars, legal_mask=legal_mask)
        return int((await self._call("request_move", {"req": req}))["action"])

    async def reveal_location(self, session_id: str, requester: str, requester_pos: tuple) -> dict:
        """Radius-gated location query (evidence-only; never fed to request_move)."""
        req = {"session_id": session_id, "requester": requester, "requester_pos": list(requester_pos)}
        return await self._call("reveal_location", {"req": req})

    async def send_final_report(self, report: dict) -> dict:
        """Cop-only: send the §3.5 report (dry-run at P6); returns the ReportAck."""
        return await self._call("send_final_report", {"report": report})
