"""Thief FastMCP server entrypoint — the canonical tools over the thief policy (T5.6).

``make_thief_server`` builds the thief :class:`AgentController` (over a trained thief
net, the ``n_agents=1`` ThiefNet) + the Stage-1 bearer verifier + the shared
canonical-tool server. Run as a SEPARATE OS process on ``mcp.thief_port``; this
module imports NO cop policy (decentralized execution boundary). The cloud
module-level ``mcp`` entrypoint is added in P8.
"""

from __future__ import annotations

from fastmcp import FastMCP

from src.mcp.agent_runtime import AgentController
from src.mcp.auth import build_verifier
from src.mcp.server_builder import build_server
from src.sdk.sdk import MarlSDK


def make_thief_server(cfg: dict, net: object, token: str | None = None) -> FastMCP:
    """Build the thief FastMCP server over a trained thief net + its bearer verifier.

    Args:
        cfg: Loaded config.
        net: The trained thief agent net (``n_agents=1``).
        token: Stage-1 bearer token (defaults to ``THIEF_MCP_TOKEN`` in the env).

    Returns:
        The thief :class:`FastMCP` server (run on ``mcp.thief_port`` or test in-memory).
    """
    controller = AgentController(MarlSDK(cfg), "thief", net, n_agents=1)
    return build_server(cfg, "thief", controller, build_verifier(cfg, "thief", token=token))
