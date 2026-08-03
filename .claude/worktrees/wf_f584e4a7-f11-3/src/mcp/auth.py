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


def build_verifier(cfg: dict, role: str, token: str | None = None) -> object:
    """Return the bearer verifier for ``role`` — the SINGLE seam both servers share.

    Stage-1 (localhost, default) is a :class:`StaticTokenVerifier`. Stage-2 (cloud,
    ``MCP_AUTH_MODE=jwt``) transparently swaps in the RS256 :class:`RevocableJWTVerifier`
    WITHOUT touching the servers — they only ever call this seam (ADR-D5-01).

    Args:
        cfg: Loaded config (reads ``mcp.auth`` scopes + per-role audience).
        role: ``"cop"`` or ``"thief"`` (selects the audience + ``{ROLE}_MCP_TOKEN``).
        token: Explicit bearer token (tests/dev); defaults to the env var (Stage-1 only).

    Returns:
        A FastMCP token verifier (static for localhost, JWT for the cloud).

    Raises:
        ValueError: If no token is supplied and ``{ROLE}_MCP_TOKEN`` is unset (Stage-1).
    """
    if os.environ.get("MCP_AUTH_MODE") == "jwt":
        from src.mcp.jwt_auth import build_jwt_verifier  # noqa: PLC0415 — cloud-only path

        return build_jwt_verifier(cfg, role)
    auth = cfg["mcp"]["auth"]
    token = token or os.environ.get(f"{role.upper()}_MCP_TOKEN")
    if not token:
        raise ValueError(f"no bearer token for role {role!r}: set {role.upper()}_MCP_TOKEN")
    scopes = list(auth["required_scopes"])
    claims = {"client_id": f"marl-{role}", "scopes": scopes, "aud": auth[f"{role}_audience"]}
    return StaticTokenVerifier(tokens={token: claims}, required_scopes=scopes)
