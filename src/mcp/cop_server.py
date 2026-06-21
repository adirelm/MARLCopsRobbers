"""Cop FastMCP server entrypoint — the canonical tools over the cop policy (T5.6).

``make_cop_server`` builds the cop :class:`AgentController` (over a trained cop net,
the ``n_agents=2`` CopNet — each session drives ONE cop's hidden stream) + the
Stage-1 bearer verifier + the shared canonical-tool server. Run as a SEPARATE OS
process on ``mcp.cop_port``; this module imports NO thief policy (decentralized
execution boundary). The cloud module-level ``mcp`` entrypoint is added in P8.
"""

from __future__ import annotations

from fastmcp import FastMCP

from src.mcp.agent_runtime import AgentController
from src.mcp.auth import build_verifier
from src.mcp.server_builder import build_server
from src.sdk.sdk import MarlSDK


def make_cop_server(cfg: dict, net: object, token: str | None = None, peer_query: object = None) -> FastMCP:
    """Build the cop FastMCP server over a trained cop net + its bearer verifier.

    Args:
        cfg: Loaded config.
        net: The trained cop agent net (``n_agents=2``).
        token: Stage-1 bearer token (defaults to ``COP_MCP_TOKEN`` in the env).
        peer_query: Optional ``query_opponent`` seam to the thief's reveal.

    Returns:
        The cop :class:`FastMCP` server (run on ``mcp.cop_port`` or test in-memory).
    """
    controller = AgentController(MarlSDK(cfg), "cop", net, n_agents=1)
    return build_server(
        cfg, "cop", controller, build_verifier(cfg, "cop", token=token), peer_query=peer_query
    )
