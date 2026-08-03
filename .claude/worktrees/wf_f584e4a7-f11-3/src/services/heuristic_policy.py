"""State-based Manhattan-heuristic policy — the opponent-pool seed (T4.6).

A frozen, competent opponent for the FIRST self-play rounds (random nets vs random
nets learn nothing). It shares the :class:`~src.services.policy.RecurrentPolicy`
acting interface (``reset`` / ``act``) but is PRIVILEGED: it reads the train-time
:class:`~src.marl.env.types.GlobalState` passed to ``act`` (not local obs) and
defers to the same greedy/ε-greedy Manhattan experts that label the BC dataset
(:func:`cop_expert` / :func:`thief_expert`), so the opponent pool is "seeded by
the Manhattan heuristic" exactly as the BC labels are. It holds no hidden state.
"""

from __future__ import annotations

from random import Random

from src.marl.data.heuristics import cop_expert, thief_expert
from src.marl.env.actions import Action
from src.marl.env.types import GlobalState


class HeuristicPolicy:
    """Privileged Manhattan-heuristic acting policy for one role (opponent seed)."""

    def __init__(self, role: str, cfg: dict, n_agents: int) -> None:
        """Bind the heuristic to a role + cop-team width.

        Args:
            role: ``"cop"`` or ``"thief"`` (selects the expert).
            cfg: Loaded config (the experts read reward/geometry knobs).
            n_agents: Number of same-role agents to produce actions for.
        """
        self._role = role
        self._cfg = cfg
        self._n = int(n_agents)

    def reset(self) -> None:
        """No-op: the heuristic is memoryless (kept for the policy interface)."""

    def act(
        self,
        obs_list: list,
        legal_masks: list,
        epsilon: float,
        rng: Random,
        state: GlobalState,
    ) -> list[Action]:
        """Return one ε-greedy Manhattan-expert action per agent from ``state``.

        Args:
            obs_list: Ignored (the heuristic is privileged — it uses ``state``).
            legal_masks: Ignored (the experts enforce legality internally).
            epsilon: ε for the expert's ε-greedy exploration.
            rng: The exploration RNG (seeded).
            state: The current global state the privileged expert reads.

        Returns:
            One :class:`Action` per agent (length ``n_agents``).
        """
        if self._role == "cop":
            return [cop_expert(state, self._cfg, idx=i, rng=rng, epsilon=epsilon) for i in range(self._n)]
        return [thief_expert(state, self._cfg, rng=rng, epsilon=epsilon)]
