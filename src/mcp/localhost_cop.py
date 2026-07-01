"""Localhost Stage-1 cop server entrypoint — module-level ``mcp`` for ``fastmcp run`` (§5.3 Stage-1).

Mirrors ``cloud_cop.py`` but for LOCALHOST: the Stage-1 :class:`StaticTokenVerifier`
(``COP_MCP_TOKEN`` from the env, or a per-run dev token) over a fresh cop net. Served as a
SEPARATE OS PROCESS on ``mcp.cop_port`` over Streamable HTTP by ``scripts/serve_match_http.py``::

    fastmcp run src/mcp/localhost_cop.py:mcp --transport http --host 127.0.0.1 --port 8001

Not imported at runtime by the in-memory path; ``fastmcp run`` imports it in the subprocess.
"""

from __future__ import annotations

import os
import secrets

from src.mcp.cop_server import make_cop_server
from src.sdk.sdk import MarlSDK
from src.utils.config_loader import load_config

_cfg = load_config()
mcp = make_cop_server(
    _cfg, MarlSDK(_cfg).fresh_net("cop"), token=os.environ.get("COP_MCP_TOKEN") or secrets.token_hex(8)
)
