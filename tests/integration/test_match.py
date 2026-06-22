"""Full local match integration (T6.2) — assemble §3.5 + 401 + decoupling.

The SDK plays N valid sub-games over the in-memory MCP, assembles + validates the
§3.5 report (totals == Σ), and DRY-RUN sends it once via the cop. Also: the static
verifier rejects an unknown token (the 401-equivalent), and query_opponent evidence
structurally cannot reach request_move. torch + the players are placeholders only.
"""

from __future__ import annotations

import asyncio
import copy
import inspect

import torch

from src.marl.nets.agent_net import RecurrentQNet
from src.mcp.agent_runtime import AgentController
from src.mcp.auth import build_verifier
from src.reporting.players import load_players
from src.sdk.sdk import MarlSDK

SEED = 7
_PLAYERS = {
    "group_name": "adrl-001",
    "students": [{"role": "A", "full_name": "Placeholder Student", "id": "000000000"}],
    "github_repo": "https://github.com/example/marl",
}


def test_full_local_match_assembles_validated_report(cfg):
    """A full match over MCP yields a validated §3.5 report (totals == Σ) + one dry-run send."""
    torch.manual_seed(SEED)
    cop = RecurrentQNet(cfg, "cop", 2)
    thief = RecurrentQNet(cfg, "thief", 1)
    out = MarlSDK(cfg).run_local_match(cop, thief, _PLAYERS, seed=SEED, stage=(5, 5, 1), num_games=2)
    assert out["num_games"] == 2
    assert len(out["report"]["sub_games"]) == 2
    totals = out["report"]["totals"]
    assert totals["cop"] == sum(g["scores"]["cop"] for g in out["report"]["sub_games"])
    assert out["ack"]["dry_run"] is True  # exactly one dry-run report send


def test_match_runs_with_prewarm_ping_disabled(cfg):
    """mcp.client.prewarm_ping=false skips the warm-up health pings; the match still runs."""
    c = copy.deepcopy(cfg)
    c["mcp"]["client"]["prewarm_ping"] = False
    cop, thief = RecurrentQNet(c, "cop", 2), RecurrentQNet(c, "thief", 1)
    out = MarlSDK(c).run_local_match(cop, thief, _PLAYERS, seed=SEED, stage=(2, 2, 1), num_games=1)
    assert out["num_games"] == 1


def test_verifier_rejects_unknown_token(cfg):
    """The 401-equivalent: the static verifier accepts the issued token, rejects others."""
    verifier = build_verifier(cfg, "cop", token="good-token")
    assert asyncio.run(verifier.verify_token("good-token")) is not None
    assert asyncio.run(verifier.verify_token("WRONG-token")) is None


def test_query_opponent_evidence_cannot_reach_request_move():
    """Structural decoupling — controller.act takes only obs + own tick (no opponent/global)."""
    params = set(inspect.signature(AgentController.act).parameters)
    assert params == {"self", "session_id", "tick", "image", "scalars", "legal_mask"}
    assert not ({"evidence", "opponent", "global_state"} & params)


def test_load_players_returns_placeholder_group_and_students():
    """load_players reads the tracked placeholder (group_name + students[role,name,id] + repo)."""
    players = load_players()
    assert players["group_name"] and players["github_repo"]
    assert players["students"]
    assert {"role", "full_name", "id"} <= set(players["students"][0])
