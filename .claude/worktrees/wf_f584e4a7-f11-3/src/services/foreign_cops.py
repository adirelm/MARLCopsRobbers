"""Scripted FOREIGN-COP battery — three opponent cops stress-testing the shipped thief.

The shipped bonus thief (README §9) is flee-primary and switches to the trained net
only after SEEING a barrier; this battery models the foreign cops that decide which
mode it actually plays. All three expose the standard acting interface
(``act(obs_list, legal_masks, epsilon, rng, state=None)`` + ``reset()``, mirroring
:class:`~src.services.policy.RecurrentPolicy`) so ``matchup_eval`` drives them as-is:

* :class:`OracleBfsCop` — perfect-information UPPER BOUND: defers to the train-time
  ``cop_expert`` BFS (via :class:`~src.services.heuristic_policy.HeuristicPolicy`),
  reading the TRUE GlobalState passed as ``state=`` each tick. It NEVER places
  barriers — a documented choice: ``cop_expert`` only emits directional moves, so
  the oracle bounds pure pursuit; barrier play is :class:`BarrierPursuitCop`'s job.
* :class:`PursuitCop` — partial-obs mirror of ``ObsFleePolicy``: uniform-random over
  legal moves pre-sighting, then greedy distance-MINIMISING toward the last-seen
  offset (dead-reckoned across own moves; resumes searching after reaching it).
* :class:`BarrierPursuitCop` — PursuitCop + a visibility-gated barrier drop
  (placement lands on the cop's CURRENT cell and is the move), built both to be
  strong and to trigger ``AdaptiveThiefPolicy``'s barrier switch.
"""

from __future__ import annotations

from collections.abc import Callable
from random import Random

import numpy as np

from src.marl.env.actions import DELTAS, Action
from src.marl.env.types import Observation
from src.services.heuristic_policy import HeuristicPolicy

_MOVES = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)


def _one(obs_list: list[Observation]) -> Observation:
    """Return the single obs, raising loudly on a multi-agent call."""
    if len(obs_list) != 1:
        raise ValueError(f"foreign cops drive exactly one cop, got {len(obs_list)} obs")
    return obs_list[0]


class OracleBfsCop:
    """Perfect-information BFS pursuer — the ``cop_expert`` oracle as a match cop."""

    def __init__(self, cfg: dict) -> None:
        """Bind the privileged Manhattan/BFS expert for one cop."""
        self._expert = HeuristicPolicy("cop", cfg, 1)

    def reset(self) -> None:
        """No episode state (the expert is memoryless)."""
        self._expert.reset()

    def act(
        self,
        obs_list: list[Observation],
        legal_masks: list,
        epsilon: float,
        rng: Random,
        state: object = None,
    ) -> list[Action]:
        """Return the BFS shortest-path first step read from the TRUE state.

        Raises:
            ValueError: If ``state`` is None — the oracle needs the GlobalState
                (drive it through ``play_matchup``'s state passthrough).
        """
        if state is None:
            raise ValueError("OracleBfsCop is privileged: pass the true GlobalState via state=")
        _one(obs_list)
        return self._expert.act(obs_list, legal_masks, epsilon, rng, state)


class PursuitCop:
    """Partial-obs chaser — ``ObsFleePolicy`` mirrored into distance-MINIMISING."""

    def __init__(self, cfg: dict) -> None:
        """Bind the obs geometry (window center = ``env.view_radius_max``)."""
        self._center = int(cfg["env"]["view_radius_max"])
        self.reset()

    def reset(self) -> None:
        """Forget the last-seen thief offset (new sub-game)."""
        self._seen_rel: tuple[int, int] | None = None

    def _observe(self, obs: Observation) -> tuple[int, int] | None:
        """Refresh the last-seen RELATIVE offset; return the CURRENT sighting (or None)."""
        hits = np.argwhere(obs["image"][1] > 0)
        if not len(hits):
            return None
        row, col = hits[0]
        self._seen_rel = (int(row) - self._center, int(col) - self._center)
        return self._seen_rel

    def act(
        self,
        obs_list: list[Observation],
        legal_masks: list,
        epsilon: float,
        rng: Random,
        state: object = None,
    ) -> list[Action]:
        """Return the pursuit move for ONE cop (``state`` accepted but ignored)."""
        self._observe(_one(obs_list))
        legal = [m for m in _MOVES if legal_masks[0][int(m)]]
        if not legal:
            return [Action.UP]
        if self._seen_rel is None:
            return [rng.choice(legal)]
        best, best_d = legal[0], None
        for move in legal:
            delta = DELTAS[move]
            rel_after = (self._seen_rel[0] - delta[0], self._seen_rel[1] - delta[1])
            distance = abs(rel_after[0]) + abs(rel_after[1])
            if best_d is None or distance < best_d:
                best_d, best = distance, move
        delta = DELTAS[best]
        self._seen_rel = (self._seen_rel[0] - delta[0], self._seen_rel[1] - delta[1])
        if self._seen_rel == (0, 0):
            self._seen_rel = None  # reached the last-seen cell — resume searching
        return [best]


class BarrierPursuitCop(PursuitCop):
    """PursuitCop + a visibility-gated barrier drop (exercises the thief's net mode).

    Places a barrier (on its OWN cell, per the transition rules) whenever the thief
    is in sight — ADJACENT INCLUDED — the mask allows PLACE, the ``game.max_barriers``
    budget is unspent, and the previous tick was not already a placement (a placement
    is a stay — two in a row just lets the thief walk away). Adjacent placement is
    deliberate: under simultaneous resolution a move-in NEVER converts against a
    competent thief (the documented 0/180 barrier-less-chaser result), while a drop
    at sighting range lands inside the thief's own view (equal radii on the graded
    5x5) — the reliable trigger for ``AdaptiveThiefPolicy``'s net switch (fires in
    7/10 seed-1000 block games vs 1/10 for a beyond-capture-range-only rule).
    """

    def __init__(self, cfg: dict) -> None:
        """Bind pursuit geometry + the config barrier budget."""
        self._cap = int(cfg["game"]["max_barriers"])
        super().__init__(cfg)

    def reset(self) -> None:
        """Clear pursuit memory AND the per-sub-game placement bookkeeping."""
        super().reset()
        self._placed = 0
        self._just_placed = False

    def _should_place(self, sighting: tuple[int, int] | None, mask: list) -> bool:
        """Gate a placement on sighting, mask legality, budget, and cadence."""
        if sighting is None or self._just_placed or self._placed >= self._cap:
            return False
        return bool(mask[int(Action.PLACE_BARRIER)])

    def act(
        self,
        obs_list: list[Observation],
        legal_masks: list,
        epsilon: float,
        rng: Random,
        state: object = None,
    ) -> list[Action]:
        """Drop a barrier when warranted, else pursue exactly like :class:`PursuitCop`."""
        sighting = self._observe(_one(obs_list))
        if self._should_place(sighting, legal_masks[0]):
            self._placed += 1
            self._just_placed = True
            return [Action.PLACE_BARRIER]
        self._just_placed = False
        return super().act(obs_list, legal_masks, epsilon, rng, state)


def foreign_cop_factories(cfg: dict) -> dict[str, Callable[[], object]]:
    """Return fresh-instance factories for the whole battery (evaluator entry point)."""
    return {
        "oracle_bfs": lambda: OracleBfsCop(cfg),
        "pursuit": lambda: PursuitCop(cfg),
        "barrier_pursuit": lambda: BarrierPursuitCop(cfg),
    }
