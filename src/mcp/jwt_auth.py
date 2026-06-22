"""Cloud JWT bearer auth — RS256 verifier with a jti deny-list (T8.1, P8 Stage-2).

Stage-2 swaps the localhost ``StaticTokenVerifier`` for an RS256 ``JWTVerifier`` through
the SAME ``build_verifier`` seam (selected by ``MCP_AUTH_MODE=jwt``) — the servers are
untouched. :class:`RevocableJWTVerifier` adds the revocation the base verifier lacks: it
rejects any token whose ``jti`` is in the ``REVOKED_TOKEN_JTIS`` deny-list (the §7.3d
revoke demo). The RS256 public key + issuer + per-role audience come from config + env
(``MCP_PUBLIC_KEY`` — NEVER tracked). The JWT import path is fastmcp-v3 verified (T8.0).
"""

from __future__ import annotations

import os

from fastmcp.server.auth.providers.jwt import JWTVerifier


class RevocableJWTVerifier(JWTVerifier):
    """A ``JWTVerifier`` plus a ``jti`` deny-list (revocation the base lacks)."""

    def __init__(self, *args: object, revoked_jtis: tuple[str, ...] = (), **kwargs: object) -> None:
        """Build the RS256 verifier and capture the revoked-``jti`` deny-set."""
        super().__init__(*args, **kwargs)
        self._revoked = frozenset(revoked_jtis)

    async def verify_token(self, token: str) -> object:
        """Verify RS256 / exp / aud / scope, then reject a revoked ``jti`` (``None`` ⇒ 401)."""
        access = await super().verify_token(token)
        if access is not None and access.claims.get("jti") in self._revoked:
            return None
        return access


def revoked_jtis(raw: str | None = None) -> tuple[str, ...]:
    """Parse the comma-separated ``REVOKED_TOKEN_JTIS`` deny-list (env by default)."""
    raw = raw if raw is not None else os.environ.get("REVOKED_TOKEN_JTIS", "")
    return tuple(jti.strip() for jti in raw.split(",") if jti.strip())


def build_jwt_verifier(cfg: dict, role: str, public_key: str | None = None) -> RevocableJWTVerifier:
    """Return the Stage-2 cloud RS256 verifier for ``role`` (``jti``-revocable).

    Args:
        cfg: Loaded config (``mcp.auth`` issuer / scopes / per-role audience).
        role: ``"cop"`` / ``"thief"`` (selects the audience).
        public_key: RS256 public-key PEM (defaults to ``MCP_PUBLIC_KEY`` in the env).

    Returns:
        A :class:`RevocableJWTVerifier` enforcing signature + audience + a jti deny-list.

    Raises:
        ValueError: If no public key is supplied and ``MCP_PUBLIC_KEY`` is unset.
    """
    auth = cfg["mcp"]["auth"]
    public_key = public_key or os.environ.get("MCP_PUBLIC_KEY")
    if not public_key:
        raise ValueError("no RS256 public key: set MCP_PUBLIC_KEY")
    return RevocableJWTVerifier(
        public_key=public_key,
        issuer=auth["issuer"],
        audience=auth[f"{role}_audience"],
        algorithm=auth["algorithm"],  # honor the configured RS256 (was relying on the FastMCP default)
        required_scopes=list(auth["required_scopes"]),
        revoked_jtis=revoked_jtis(),
    )
