"""Potential-based reward shaping — the RL TRAINING SIGNAL ONLY (T1.4).

This module is STRICTLY separate from the §3.4 scoreboard (see scorer.py, which
this file MUST NEVER import). The shaping potential is Ng-1999 policy-invariant:
``Phi = -min_i manhattan(cop_i, thief) / d_max`` with
``d_max = max(1, (H-1) + (W-1))`` derived per live grid (architect decision #2;
NOT a config literal; the ``max(1, ...)`` floor mirrors grid.sample_spawn so a
degenerate 1x1 grid cannot ZeroDivisionError), and ``Phi = 0`` on terminal
states so ``F = distance_weight*(gamma*Phi(s')-Phi(s))`` collapses correctly at
absorbing states. All coefficients come from config.
"""

from __future__ import annotations

from src.marl.env.grid import manhattan
from src.marl.env.types import GlobalState


def potential(state: GlobalState) -> float:
    """Return the shaping potential Phi for ``state``.

    ``Phi = -min_i manhattan(cop_i, thief) / d_max`` over the cop team, with
    ``d_max = max(1, (h-1) + (w-1))`` (the floor mirrors grid.sample_spawn so a
    1x1 grid cannot divide by zero). Terminal/absorbing states yield ``0.0`` so
    the Ng-1999 policy-invariance guarantee holds (``phi_terminal_zero``).

    Args:
        state: The global state to evaluate.

    Returns:
        The (non-positive) potential, or ``0.0`` if the state is terminal.
    """
    if state.terminal:
        return 0.0
    d_max = max(1, (state.h - 1) + (state.w - 1))
    nearest = min(manhattan(cop, state.thief_pos) for cop in state.cop_pos)
    return -nearest / d_max


class RewardModel:
    """Compute per-agent RL rewards from a state transition.

    Reads ``reward.*``, ``algo.gamma`` and ``env.reward_mode`` from config. The
    cop team shares a single shaped, penalized reward; the thief reward is
    ``-team`` in ``dec_pomdp`` (zero-sum proxy) or role-specific in ``posg``.
    """

    def __init__(self, cfg: dict) -> None:
        """Bind reward coefficients from the validated config.

        Args:
            cfg: The loaded project config (see config/config.yaml).
        """
        r = cfg["reward"]
        self._gamma = cfg["algo"]["gamma"]
        self._mode = cfg["env"]["reward_mode"]
        self._shaping_enabled = r["shaping_enabled"]
        self._shaping_eval_enabled = r["shaping_eval_enabled"]
        self._distance_weight = r["distance_weight"]
        self._step_penalty = r["step_penalty"]
        self._barrier_cost = r["barrier_cost"]
        self._capture_bonus = r["capture_bonus"]
        self._timeout_penalty = r["timeout_penalty"]
        self._evade_step = r["evade_step"]
        self._timeout_bonus = r["timeout_bonus"]

    def _shaping(self, prev: GlobalState, nxt: GlobalState, eval_mode: bool) -> float:
        """Return the potential-based shaping term F (0.0 when gated off)."""
        if not self._shaping_enabled:
            return 0.0
        if eval_mode and not self._shaping_eval_enabled:
            return 0.0
        return self._distance_weight * (self._gamma * potential(nxt) - potential(prev))

    def _cop_team_reward(
        self, prev: GlobalState, nxt: GlobalState, winner: str | None, eval_mode: bool
    ) -> float:
        """Aggregate the shared cop-team reward for one transition."""
        reward = self._shaping(prev, nxt, eval_mode) - self._step_penalty
        if nxt.barriers_used > prev.barriers_used:
            reward -= self._barrier_cost
        if nxt.terminal and winner == "cop":
            reward += self._capture_bonus
        elif nxt.terminal and winner == "thief":
            reward -= self._timeout_penalty
        return reward

    def compute(
        self,
        prev: GlobalState,
        nxt: GlobalState,
        winner: str | None = None,
        eval_mode: bool = False,
    ) -> dict[str, float]:
        """Compute the per-agent reward dict for the ``prev -> nxt`` transition.

        Args:
            prev: State before the joint action.
            nxt: State after the joint action.
            winner: ``"cop"`` (capture), ``"thief"`` (timeout), or ``None``.
            eval_mode: When True, shaping is gated by ``shaping_eval_enabled``.

        Returns:
            A dict keyed ``cop_0..cop_{n-1}`` plus ``thief``. In ``dec_pomdp`` the
            thief reward is ``-team``; in ``posg`` it is role-specific.
        """
        team = self._cop_team_reward(prev, nxt, winner, eval_mode)
        out: dict[str, float] = {f"cop_{i}": team for i in range(len(prev.cop_pos))}
        out["thief"] = self._thief_reward(team, nxt, winner)
        return out

    def _thief_reward(self, team: float, nxt: GlobalState, winner: str | None) -> float:
        """Return the thief reward for the active ``reward_mode``."""
        if self._mode == "posg":
            if nxt.terminal:
                return self._timeout_bonus if winner == "thief" else 0.0
            return self._evade_step
        return -team
