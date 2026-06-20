"""Bearer-auth verifier seam — Stage-1 static tokens (T5.4; cloud JWT in P8).

``build_verifier`` is the SINGLE auth seam both servers share. Stage-1 (localhost)
returns a FastMCP :class:`StaticTokenVerifier` keyed on the role's bearer token
(value from ``{ROLE}_MCP_TOKEN`` in the env — NEVER tracked) and the configured
``required_scopes``/audience. Stage-2 (P8 cloud) swaps the same seam for a
``JWTVerifier`` (RS256 + ``exp`` + a jti deny-list) without touching the servers.
"""

from __future__ import annotations

import os

from fastmcp.server.auth import StaticTokenVerifier


def build_verifier(cfg: dict, role: str, token: str | None = None) -> StaticTokenVerifier:
    """Return the Stage-1 static-token verifier for ``role`` (localhost bearer auth).

    Args:
        cfg: Loaded config (reads ``mcp.auth`` scopes + per-role audience).
        role: ``"cop"`` or ``"thief"`` (selects the audience + ``{ROLE}_MCP_TOKEN``).
        token: Explicit bearer token (tests/dev); defaults to the env var.

    Returns:
        A :class:`StaticTokenVerifier` mapping the role token to its claims.

    Raises:
        ValueError: If no token is supplied and ``{ROLE}_MCP_TOKEN`` is unset.
    """
    auth = cfg["mcp"]["auth"]
    token = token or os.environ.get(f"{role.upper()}_MCP_TOKEN")
    if not token:
        raise ValueError(f"no bearer token for role {role!r}: set {role.upper()}_MCP_TOKEN")
    scopes = list(auth["required_scopes"])
    claims = {"client_id": f"marl-{role}", "scopes": scopes, "aud": auth[f"{role}_audience"]}
    return StaticTokenVerifier(tokens={token: claims}, required_scopes=scopes)
