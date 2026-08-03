"""§9 match policies — the obs-driven flee thief + the auto-adaptive wrapper.

The bonus-match thief lineup (README §9): PRIMARY = greedy flee — 0/180 captures
conceded to any barrier-less chaser in our evals — reimplemented here from the LOCAL
egocentric observation only (the MCP runtime never serves global state); CONTINGENCY =
the trained net, engaged by :class:`AdaptiveThiefPolicy` the moment a barrier is seen
in view. The switch is autonomous and permanent for the match — §3 forbids external
operator intervention mid-match. Both expose the :class:`RecurrentPolicy` acting
interface (``reset`` / ``act``) so ``SDK.build_policy`` passes them straight through
to the MCP ``AgentController``.
"""

from __future__ import annotations

from random import Random

import numpy as np

from src.marl.env.actions import DELTAS, Action
from src.marl.env.types import Observation
from src.services.policy import RecurrentPolicy

_MOVES = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)


class ObsFleePolicy:
    """Greedy-flee thief acting on the LOCAL egocentric observation only.

    Mirrors the evaluated god-view flee heuristic in relative coordinates: remember the
    opponent's last-seen offset (updated for own motion each tick), pick the legal
    directional move that maximizes the Manhattan distance to it; before first contact,
    move uniformly among legal directions.
    """

    def __init__(self, cfg: dict) -> None:
        """Bind the obs geometry (window center = ``env.view_radius_max``)."""
        self._center = int(cfg["env"]["view_radius_max"])
        self.reset()

    def reset(self) -> None:
        """Forget the last-seen opponent offset (new sub-game)."""
        self._seen_rel: tuple[int, int] | None = None

    def _observe(self, obs: Observation) -> None:
        """Refresh the last-seen RELATIVE offset from the obs opponent channel."""
        hits = np.argwhere(obs["image"][1] > 0)
        if len(hits):
            row, col = hits[0]
            self._seen_rel = (int(row) - self._center, int(col) - self._center)

    def act(
        self,
        obs_list: list[Observation],
        legal_masks: list,
        epsilon: float,
        rng: Random,
        state: object = None,
    ) -> list[Action]:
        """Return the flee move for ONE thief (interface-compatible with RecurrentPolicy).

        Raises:
            ValueError: If called with more than one agent's observation.
        """
        if len(obs_list) != 1:
            raise ValueError(f"ObsFleePolicy drives exactly one thief, got {len(obs_list)} obs")
        self._observe(obs_list[0])
        legal = [m for m in _MOVES if legal_masks[0][int(m)]]
        if not legal:
            return [Action.UP]
        if self._seen_rel is None:
            return [rng.choice(legal)]
        best, best_d = legal[0], -1
        for move in legal:
            delta = DELTAS[move]
            rel_after = (self._seen_rel[0] - delta[0], self._seen_rel[1] - delta[1])
            distance = abs(rel_after[0]) + abs(rel_after[1])
            if distance > best_d:
                best_d, best = distance, move
        delta = DELTAS[best]
        self._seen_rel = (self._seen_rel[0] - delta[0], self._seen_rel[1] - delta[1])
        return [best]


class AdaptiveThiefPolicy:
    """Flee until the opponent cop is SEEN using barriers — then act with the trained net.

    The trained net's GRU hidden state is warm-carried from the sub-game start (the net
    is advanced every tick even while the flee policy is acting), so a mid-sub-game
    switch hands over a live recurrent state, not a cold one. ``switched`` is sticky
    across ``reset()`` — one barrier sighting converts the whole remaining match, and
    that DELIBERATELY includes a voided attempt's sighting carrying into its same-seed
    replay: the switch is match-level adaptation (README §9 lineup; ANALYSIS §12
    sticky-switch battery), not per-attempt state.
    """

    def __init__(self, cfg: dict, net: object) -> None:
        """Wrap the flee policy + the contingency net policy."""
        self._flee = ObsFleePolicy(cfg)
        self._net_policy = RecurrentPolicy(net, 1)
        self.switched = False

    def reset(self) -> None:
        """Reset per-sub-game state (flee memory + GRU hidden); the switch is sticky."""
        self._flee.reset()
        self._net_policy.reset()

    def act(
        self,
        obs_list: list[Observation],
        legal_masks: list,
        epsilon: float,
        rng: Random,
        state: object = None,
    ) -> list[Action]:
        """Advance both policies; act by flee pre-switch, by the net after."""
        if not self.switched and bool(np.any(obs_list[0]["image"][2] > 0)):
            self.switched = True
        net_actions = self._net_policy.act(obs_list, legal_masks, epsilon, rng)  # warm hidden carry
        if self.switched:
            return net_actions
        return self._flee.act(obs_list, legal_masks, epsilon, rng)
