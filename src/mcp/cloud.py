"""Cloud server factory — build a role's Stage-2 server (JWT auth + OLoRA net) (T8.1).

``build_cloud_server`` returns the SAME canonical-tool server as localhost but with the
RS256 JWT verifier: it scopes ``MCP_AUTH_MODE=jwt`` to the build so the shared
``build_verifier`` seam selects JWT, then restores the env (no global mutation). The
actor net is loaded from ``MODEL_PATH`` (the OLoRA-tuned ``.pt``) unless one is injected
(tests). The thin ``cloud_cop.py`` / ``cloud_thief.py`` modules expose ``mcp`` as the
FastMCP/Prefect deploy entrypoint (``src/mcp/cloud_cop.py:mcp``) — the only cloud-only
code (omitted from coverage: they import a real model on the worker).
"""

from __future__ import annotations

import os
from contextlib import contextmanager

from fastmcp import FastMCP

from src.mcp.cop_server import make_cop_server
from src.mcp.thief_server import make_thief_server
from src.sdk.sdk import MarlSDK
from src.utils.config_loader import load_config

_NET_AGENTS = {"cop": 2, "thief": 1}


@contextmanager
def _jwt_env():
    """Scope ``MCP_AUTH_MODE=jwt`` to the wrapped build, then restore the prior value."""
    prev = os.environ.get("MCP_AUTH_MODE")
    os.environ["MCP_AUTH_MODE"] = "jwt"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("MCP_AUTH_MODE", None)
        else:
            os.environ["MCP_AUTH_MODE"] = prev


def _load_actor(cfg: dict, role: str) -> object:  # pragma: no cover - needs MODEL_PATH + a real .pt
    """Load the role's OLoRA-tuned actor net from ``MODEL_PATH`` (cloud worker only)."""
    return MarlSDK(cfg).load_weights(role, os.environ["MODEL_PATH"], _NET_AGENTS[role])


def build_cloud_server(role: str, net: object = None, cfg: dict | None = None) -> FastMCP:
    """Build the role's CLOUD server: SAME tool contract as localhost, RS256 JWT auth.

    Args:
        role: ``"cop"`` / ``"thief"``.
        net: The actor net; when ``None`` it is loaded from ``MODEL_PATH`` (the cloud).
        cfg: Loaded config; defaults to :func:`load_config`.

    Returns:
        A :class:`FastMCP` server with identical tool names + ``/mcp`` path to localhost.
    """
    cfg = cfg or load_config()
    net = net if net is not None else _load_actor(cfg, role)
    factory = make_cop_server if role == "cop" else make_thief_server
    with _jwt_env():
        return factory(cfg, net)
