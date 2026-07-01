"""Runnable §5.3 Stage-1: the two MCP servers over REAL localhost HTTP + a match over the wire.

Boots the cop + thief FastMCP servers as SEPARATE OS PROCESSES on ``mcp.cop_port`` /
``mcp.thief_port`` over Streamable HTTP (``fastmcp run … --transport http`` — the SAME command
``deploy/render.yaml`` uses for the cloud), then plays a sub-game over HTTP with per-role bearer
tokens and prints the redacted cop<->thief comms trace: the SAME ``session_id`` on BOTH servers
over the wire (§7.3d proof). Manual (needs the two configured ports free), not a CI gate (real
sockets/subprocesses): ``uv run python scripts/serve_match_http.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import socket
import subprocess
import time

from src.mcp.clients import AgentClient, make_client
from src.mcp.referee import MatchRunner, Referee
from src.results.comms import _Capture
from src.utils.config_loader import load_config

_ROLES = ("cop", "thief")
_READY_TIMEOUT_S = 90


def _serve(role: str, port: int, token: str, host: str, transport: str) -> subprocess.Popen:
    """Start a role server as a SEPARATE OS process over ``transport`` (``fastmcp run``)."""
    env = {**os.environ, f"{role.upper()}_MCP_TOKEN": token}
    entry = f"src/mcp/localhost_{role}.py:mcp"
    cmd = ["uv", "run", "fastmcp", "run", entry]
    cmd += ["--transport", transport, "--host", host, "--port", str(port)]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _wait_ready(host: str, port: int) -> bool:
    """Poll until the server's TCP port accepts connections (or time out)."""
    for _ in range(_READY_TIMEOUT_S):
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(1)
    return False


async def _play(cfg: dict, urls: dict, tokens: dict) -> dict:
    """Play ONE sub-game over HTTP bearer clients; return the match summary."""
    cc = cfg["mcp"]["client"]
    kw = {
        "max_retries": int(cc["max_retries"]),
        "backoff_s": float(cc["backoff_s"]),
        "timeout_s": float(cc["timeout_s"]),
    }
    cop_c = AgentClient(make_client(urls["cop"], tokens["cop"]), label="cop", **kw)
    thief_c = AgentClient(make_client(urls["thief"], tokens["thief"]), label="thief", **kw)
    async with cop_c as cop, thief_c as thief:
        await cop.health()
        await thief.health()
        return await MatchRunner(Referee(cfg, 5, 5, 1), 1, 7).play_match(cop, thief)


def main() -> dict:
    """Serve both roles over HTTP, play a sub-game over the wire, print the comms proof."""
    cfg = load_config()
    m = cfg["mcp"]
    host, path = m["host"], m["path"]
    tokens = {r: secrets.token_hex(8) for r in _ROLES}
    procs = {r: _serve(r, m[f"{r}_port"], tokens[r], host, m["transport"]) for r in _ROLES}
    handler = _Capture()
    logger = logging.getLogger("marl.mcp.client")
    logger.addHandler(handler)
    logger.setLevel(cfg["logging"]["level"])  # config-driven level (mirror comms.capture; no hardcode)
    try:
        for r in _ROLES:
            if not _wait_ready(host, m[f"{r}_port"]):
                raise SystemExit(f"{r} server did not bind {host}:{m[f'{r}_port']}")
        urls = {r: f"http://{host}:{m[f'{r}_port']}{path}" for r in _ROLES}
        match = asyncio.run(_play(cfg, urls, tokens))
        traces = {
            ln.split("trace=")[1].split()[0]
            for ln in handler.lines
            if "request_move" in ln and "trace=" in ln
        }
        print(
            f"[stage1-http] {match['num_games']} valid sub-game(s) over REAL HTTP — "
            f"cop {host}:{m['cop_port']} <-> thief {host}:{m['thief_port']}; "
            f"shared request_move trace(s): {sorted(traces)}"
        )
        return {"match": match, "traces": sorted(traces)}
    finally:
        for p in procs.values():
            p.terminate()
        for p in procs.values():
            try:
                p.wait(timeout=10)
            except Exception:  # best-effort teardown
                p.kill()


if __name__ == "__main__":
    main()
