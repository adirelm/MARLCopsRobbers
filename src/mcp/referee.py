"""Referee / MatchRunner — the environment drives the agents over MCP (T5.8).

The referee is the SOLE holder of ground-truth global state ``s`` (it owns the
env). Agents only ever receive their LOCAL masked obs via ``request_move`` and can
never read ``s`` through any tool. Each tick the referee encodes each agent's ego
obs, asks its server for an action, applies the simultaneous-move transition +
capture (ADR-001) via the env, and scores per Table 1. ``MatchRunner`` plays
exactly ``num_games`` VALID sub-games, replaying technical losses (§3.7). The
referee is the environment, NOT a third player.
"""

from __future__ import annotations

from src.marl.data.obs_encoder import encode_obs_batch
from src.marl.env.actions import Action
from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.marl.env.scorer import Scorer


def _encode_one(obs: dict) -> tuple[list, list]:
    """Encode one local Observation into JSON-safe ``(image, scalars)`` lists for MCP."""
    images, scalars = encode_obs_batch([obs])
    return images[0].tolist(), scalars[0].tolist()


class Referee:
    """The environment (sole ground-truth ``s`` holder) driving agents over MCP."""

    def __init__(self, cfg: dict, h: int, w: int, num_cops: int = 1) -> None:
        """Own the env + scorer for one curriculum grid (the referee is the env)."""
        self._env = CopsRobbersEnv(cfg, h=h, w=w, num_cops=int(num_cops))
        self._scorer = Scorer(cfg)
        self._grid = (h, w)

    async def play_sub_game(self, cop: object, thief: object, seed: int, session_id: str = "sg") -> dict:
        """Drive ONE full sub-game over MCP; adjudicate winner + scores (Table 1)."""
        obs, info = self._env.reset(seed=seed)
        state = self._env.state()
        await cop.new_sub_game(session_id, self._grid, state.cop_pos[0])
        await thief.new_sub_game(session_id, self._grid, state.thief_pos)
        terminated, tick, capture = False, 0, False
        while not terminated:
            masks = info["action_mask"]
            cop_img, cop_sc = _encode_one(obs["cop_0"])
            thf_img, thf_sc = _encode_one(obs["thief"])
            cop_mask = [bool(b) for b in masks["cop_0"]]  # JSON-safe (env mask is numpy bool)
            thf_mask = [bool(b) for b in masks["thief"]]
            cop_a = await cop.request_move(session_id, tick, cop_img, cop_sc, cop_mask)
            thf_a = await thief.request_move(session_id, tick, thf_img, thf_sc, thf_mask)
            obs, _r, terminated, info = self._env.step({"cop_0": Action(cop_a), "thief": Action(thf_a)})
            capture, tick = bool(info.get("capture")), tick + 1
        winner = info.get("winner") or ("cop" if capture else "thief")
        scores = self._scorer.score(winner)
        return {
            "game_id": session_id,
            "grid": [self._grid[0], self._grid[1]],
            "winner": winner,
            "capture": capture,
            "steps": tick,
            "seed": int(seed),
            "scores": {"cop": int(scores["cop"]), "thief": int(scores["thief"])},
        }


class MatchRunner:
    """Play exactly ``num_games`` VALID sub-games, replaying technical losses (§3.7)."""

    def __init__(self, referee: Referee, num_games: int, base_seed: int) -> None:
        """Bind the referee + the match size + the deterministic seed base."""
        self._ref = referee
        self._n = int(num_games)
        self._base = int(base_seed)

    async def play_match(self, cop: object, thief: object) -> dict:
        """Run the match; return ``{sub_games, totals, num_games}`` (totals over valid games)."""
        games: list[dict] = []
        offset = 0
        while len(games) < self._n and offset < self._n * 10:
            seed = self._base + offset
            offset += 1
            try:
                games.append(await self._ref.play_sub_game(cop, thief, seed, f"sg-{len(games)}"))
            except Exception:  # technical loss: not counted toward the 6, replay with the next seed
                continue
        totals = {role: sum(g["scores"][role] for g in games) for role in ("cop", "thief")}
        return {"sub_games": games, "totals": totals, "num_games": len(games)}
