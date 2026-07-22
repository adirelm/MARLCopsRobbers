"""MCP package — the two FastMCP servers (cop/thief) and their tool handlers."""

# V3 §14 public surface. Submodule names (not eagerly imported symbols): the package is a
# module container, so `from ... import *` binds the modules and heavy/optional deps stay
# off the import path. Private `_*` helpers and CLI/deploy entrypoints are deliberately out.
__all__ = [
    "agent_runtime",
    "auth",
    "clients",
    "cloud",
    "cop_server",
    "jwt_auth",
    "match",
    "referee",
    "schemas",
    "server_builder",
    "thief_server",
]
