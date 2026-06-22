"""SpectatorSession — drive a watchable god-view game into SpectatorFrames (T7.1).

The GUI's frame source (obtained via ``SDK.spectator_session``). It owns a god-view
env and steps a heuristic-driven sub-game, emitting one
:class:`~src.gui.spectator.SpectatorFrame` per move (full board + HUD). The
spectator reads the REFEREE's ground truth (:func:`render_state`), never agent
obs. It lives in the services layer (it imports ``src.marl``) so the GUI stays pure
(GUI imports only the frame type + the SDK).
"""

from __future__ import annotations

from random import Random

from src.gui.spectator import SpectatorFrame
from src.marl.data.heuristics import cop_expert, thief_expert
from src.marl.env.actions import Action
from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.marl.env.render_state import render_state
from src.marl.env.scorer import Scorer


class SpectatorSession:
    """Step a god-view sub-game (heuristic-driven), emitting frames per move."""

    def __init__(self, cfg: dict, h: int, w: int, num_cops: int = 1, seed: int = 0) -> None:
        """Build the god-view env + scorer for a watchable sub-game."""
        self._cfg = cfg
        self._env = CopsRobbersEnv(cfg, h=h, w=w, num_cops=int(num_cops))
        self._scorer = Scorer(cfg)
        self._rng = Random(int(seed))
        self._num_games = int(cfg["game"]["num_games"])
        self._sub_game = 1
        self._banked = 0  # finished sub-games already accumulated into totals (caps the match)
        self._totals = {"cop": 0, "thief": 0}
        self.reset()

    def reset(self) -> SpectatorFrame:
        """Restart the CURRENT sub-game; return its opening frame (keeps the counter + totals)."""
        self._env.reset(seed=self._rng.randrange(2**31))
        self._move = 0
        self._winner: str | None = None
        self._last_action: dict | None = None
        return self._frame()

    def next_sub_game(self) -> SpectatorFrame:
        """Bank the finished sub-game's score + advance to the next (capped at num_games).

        A no-op while the current sub-game is unfinished (no winner yet), so the 'n' key
        can't skip a live game, AND once all ``num_games`` are banked (the match is complete)
        so the final game can't be re-banked into inflated totals. Otherwise it accumulates
        the finished score, advances the counter, and resets the env for the next sub-game.
        """
        if self._winner is None or self._banked >= self._num_games:
            return self._frame()
        final = self._scorer.score(self._winner)
        for role in ("cop", "thief"):
            self._totals[role] += final[role]
        self._banked += 1
        if self._banked >= self._num_games:
            return self._frame()  # final sub-game banked: match complete, do not start another
        self._sub_game += 1
        return self.reset()

    def step(self) -> SpectatorFrame:
        """Advance one move via the heuristic joint action; return the new frame."""
        if self._winner is not None:
            return self._frame()  # terminal is idempotent
        state = self._env.state()
        joint = {f"cop_{i}": cop_expert(state, self._cfg, idx=i) for i in range(len(state.cop_pos))}
        joint["thief"] = thief_expert(state, self._cfg)
        _obs, _r, terminated, info = self._env.step(joint)
        self._move += 1
        self._last_action = {key: Action(int(action)).name for key, action in joint.items()}
        if terminated:
            self._winner = info.get("winner") or "thief"
        return self._frame()

    def _frame(self) -> SpectatorFrame:
        """Snapshot the god-view + HUD into a frozen :class:`SpectatorFrame`."""
        god = render_state(self._env.state(), self._cfg)
        scores = self._scorer.score(self._winner) if self._winner else {"cop": 0, "thief": 0}
        radius = int(self._cfg["env"]["view_radius_by_grid"][min(god["h"], god["w"])])
        return SpectatorFrame(
            grid=(god["h"], god["w"]),
            cop_positions=tuple(tuple(p) for p in god["cop_positions"]),
            thief_position=tuple(god["thief_position"]),
            barriers=tuple(tuple(b) for b in god["barriers"]),
            view_radius=radius,
            move=self._move,
            max_moves=god["max_moves"],
            sub_game=self._sub_game,
            num_games=self._num_games,
            scores=scores,
            totals=dict(self._totals),
            winner=self._winner,
            last_action=self._last_action,
        )
