"""Cloud thief deploy entrypoint — module-level ``mcp`` (T8.1; FastMCP/Prefect target).

Deployed as ``src/mcp/cloud_thief.py:mcp``. Serves the SAME canonical tool contract as
localhost over RS256 JWT auth, loading the OLoRA-tuned thief actor from ``MODEL_PATH``.
Imported ONLY by the cloud worker (needs ``MODEL_PATH`` + ``MCP_PUBLIC_KEY`` in the env);
omitted from coverage. Deploy steps: see ``deploy/runbook.md``.
"""

from src.mcp.cloud import build_cloud_server

mcp = build_cloud_server("thief")
