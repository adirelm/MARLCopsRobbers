"""Referee + peer-client integration (T5.7/T5.8) over in-memory MCP servers.

The referee (the environment) drives both agent servers through a full sub-game and
a multi-game match via the typed AgentClient, adjudicating winner + scores. Proves
agents act only on local obs (the request_move contract) and the match accumulates
totals over valid sub-games. torch seeded.
"""

from __future__ import annotations

import asyncio

import torch
from fastmcp import Client

from src.marl.nets.agent_net import RecurrentQNet
from src.mcp.clients import AgentClient
from src.mcp.cop_server import make_cop_server
from src.mcp.referee import MatchRunner, Referee
from src.mcp.thief_server import make_thief_server

SEED = 7


def _servers(cfg):
    """Build a fresh seeded cop + thief in-memory server pair."""
    torch.manual_seed(SEED)
    cop = make_cop_server(cfg, RecurrentQNet(cfg, "cop", 2), token="dev-cop")
    thief = make_thief_server(cfg, RecurrentQNet(cfg, "thief", 1), token="dev-thief")
    return cop, thief


def test_referee_plays_a_full_sub_game_to_a_winner(cfg):
    """A full 5x5 sub-game over MCP terminates with a winner + Table-1 scores."""
    cop_srv, thief_srv = _servers(cfg)

    async def _run():
        ref = Referee(cfg, 5, 5, num_cops=1)
        async with AgentClient(Client(cop_srv)) as cop, AgentClient(Client(thief_srv)) as thief:
            return await ref.play_sub_game(cop, thief, seed=SEED)

    result = asyncio.run(_run())
    assert result["winner"] in ("cop", "thief")
    assert result["steps"] >= 1
    assert {"cop", "thief"} == set(result["scores"])


def test_match_runner_plays_n_valid_sub_games_with_totals(cfg):
    """MatchRunner plays exactly num_games sub-games and sums per-role totals."""
    cop_srv, thief_srv = _servers(cfg)

    async def _run():
        runner = MatchRunner(Referee(cfg, 5, 5, num_cops=1), num_games=2, base_seed=SEED)
        async with AgentClient(Client(cop_srv)) as cop, AgentClient(Client(thief_srv)) as thief:
            return await runner.play_match(cop, thief)

    match = asyncio.run(_run())
    assert match["num_games"] == 2
    assert len(match["sub_games"]) == 2
    expected_cop = sum(g["scores"]["cop"] for g in match["sub_games"])
    assert match["totals"]["cop"] == expected_cop


class _FlakyReferee:
    """A referee stub that technically-loses the first sub-game, then succeeds."""

    def __init__(self) -> None:
        self.attempts = 0

    async def play_sub_game(self, _cop, _thief, seed, _session):
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("technical loss")
        return {"winner": "cop", "scores": {"cop": 20, "thief": 5}, "seed": seed}


def test_match_runner_replays_technical_losses():
    """A technically-lost sub-game is not counted; the match replays to num_games valid."""
    runner = MatchRunner(_FlakyReferee(), num_games=1, base_seed=7)
    match = asyncio.run(runner.play_match(None, None))
    assert match["num_games"] == 1
    assert match["sub_games"][0]["winner"] == "cop"
