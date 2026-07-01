"""Localhost Stage-1 thief server entrypoint — module-level ``mcp`` for ``fastmcp run`` (§5.3 Stage-1).

Mirrors ``cloud_thief.py`` but for LOCALHOST: the Stage-1 :class:`StaticTokenVerifier`
(``THIEF_MCP_TOKEN`` from the env, or a per-run dev token) over a fresh thief net. Served as a
SEPARATE OS PROCESS on ``mcp.thief_port`` over Streamable HTTP by ``scripts/serve_match_http.py``::

    fastmcp run src/mcp/localhost_thief.py:mcp --transport http --host 127.0.0.1 --port 8002

Not imported at runtime by the in-memory path; ``fastmcp run`` imports it in the subprocess.
"""

from __future__ import annotations

import os
import secrets

from src.mcp.thief_server import make_thief_server
from src.sdk.sdk import MarlSDK
from src.utils.config_loader import load_config

_cfg = load_config()
mcp = make_thief_server(
    _cfg, MarlSDK(_cfg).fresh_net("thief"), token=os.environ.get("THIEF_MCP_TOKEN") or secrets.token_hex(8)
)
