"""The localhost Stage-1 HTTP server entrypoints build a real FastMCP server (§5.3 Stage-1).

Covers the module-level ``mcp`` build (fresh net + StaticTokenVerifier) that
``fastmcp run src/mcp/localhost_{role}.py:mcp --transport http`` serves. The full over-HTTP
match itself is exercised by the manual ``scripts/serve_match_http.py`` (real sockets/subprocess,
kept out of CI for determinism), not here.
"""

from __future__ import annotations

from fastmcp import FastMCP


def test_localhost_cop_entrypoint_builds_a_server():
    """The cop entrypoint exposes a real FastMCP ``mcp`` (per-run dev token when unset)."""
    from src.mcp import localhost_cop  # noqa: PLC0415 — import triggers the module-level build

    assert isinstance(localhost_cop.mcp, FastMCP)


def test_localhost_thief_entrypoint_builds_a_server():
    """The thief entrypoint exposes a real FastMCP ``mcp``."""
    from src.mcp import localhost_thief  # noqa: PLC0415

    assert isinstance(localhost_thief.mcp, FastMCP)
