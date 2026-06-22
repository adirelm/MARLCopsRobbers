"""Local full-match orchestration over in-memory MCP (T6.1) — §3.5 assemble + dry-run.

Spins both agent servers, has the :class:`Referee` play ``num_games`` VALID
sub-games over the canonical MCP tool contract, assembles + validates the §3.5
:class:`MatchReport`, and dry-run sends it via the cop's ``send_final_report``. It
is IN-MEMORY (no subprocess) for CI determinism (R11); the subprocess/HTTP path
reuses the SAME contract. Tokens come from the env (``{ROLE}_MCP_TOKEN``); when
unset (local dev) a per-run token is generated — never a hardcoded literal.
"""

from __future__ import annotations

import os
import secrets

from fastmcp import Client

from src.mcp.clients import AgentClient
from src.mcp.cop_server import make_cop_server
from src.mcp.referee import MatchRunner, Referee
from src.mcp.thief_server import make_thief_server
from src.reporting.schema import Student, build_report, validate


def _token(role: str) -> str:
    """Return the role bearer token from the env, or a generated per-run dev token."""
    return os.environ.get(f"{role.upper()}_MCP_TOKEN") or secrets.token_hex(8)


async def run_local_match(  # noqa: PLR0913 — cfg + 2 nets + players + seed + stage + count are distinct
    cfg: dict,
    cop_net: object,
    thief_net: object,
    players: dict,
    seed: int,
    stage: tuple[int, int, int] = (5, 5, 1),
    num_games: int | None = None,
) -> dict:
    """Play a full local match, assemble+validate the §3.5 report, dry-run send it.

    Args:
        cfg: Loaded config (``game.num_games`` is the default match size).
        cop_net: The trained cop net served by the cop server.
        thief_net: The trained thief net served by the thief server.
        players: ``{"group": str, "students": [{"full_name","id"}, ...]}`` (placeholders).
        seed: Base seed for the match (per-sub-game seeds derive from it).
        stage: ``(h, w, num_cops)`` the match is played on.
        num_games: Valid sub-games to play (default ``game.num_games`` = 6).

    Returns:
        ``{"report": <§3.5 body>, "num_games": int, "ack": <ReportAck dry-run>}``.
    """
    h, w, num_cops = stage
    games = int(num_games if num_games is not None else cfg["game"]["num_games"])
    cop_srv = make_cop_server(cfg, cop_net, token=_token("cop"))
    thief_srv = make_thief_server(cfg, thief_net, token=_token("thief"))
    runner = MatchRunner(Referee(cfg, h, w, num_cops), games, int(seed))
    students = [Student(**student) for student in players["students"]]
    cc = cfg["mcp"]["client"]
    rt, bo, to = int(cc["max_retries"]), float(cc["backoff_s"]), float(cc["timeout_s"])
    async with (
        AgentClient(Client(cop_srv), max_retries=rt, label="cop", backoff_s=bo, timeout_s=to) as cop,
        AgentClient(Client(thief_srv), max_retries=rt, label="thief", backoff_s=bo, timeout_s=to) as thief,
    ):
        await cop.health()  # absorb (trivial in-memory) cold-start
        match = await runner.play_match(cop, thief)
        report = build_report(
            players["group_name"],
            students,
            players["github_repo"],
            cfg["project"]["timezone"],
            match["sub_games"],
        ).to_dict()
        validate(report)
        ack = await cop.send_final_report(report)  # dry-run — no email at P6
    return {"report": report, "num_games": match["num_games"], "ack": ack}
