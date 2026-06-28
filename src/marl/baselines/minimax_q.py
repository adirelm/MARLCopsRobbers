"""Tabular Minimax-Q for a 2-player zero-sum game (P-bonus, L11 §2.2, eq 2.1).

``MinimaxQ`` keeps one Q-table ``Q(s, a_row, a_col)`` — the ROW (cop, maximizer) value;
the COLUMN player (thief) minimizes it. Each update bootstraps the next state's value
through the maximin LP (:func:`~src.marl.baselines.minimax_lp.solve_minimax_value`) — the
principled operator that replaces single-agent ``max`` (Littman 1994). Pure NumPy + scipy
LP, no deep nets: the equilibrium-learning contrast to our deep self-play (README §7.2).
"""

from __future__ import annotations

import numpy as np

from src.marl.baselines.minimax_lp import solve_minimax_value


class MinimaxQ:
    """A tabular Minimax-Q learner for a 2-player zero-sum stochastic game."""

    def __init__(self, n_row: int, n_col: int, alpha: float, gamma: float) -> None:
        """Bind the action counts + learning rate + discount; start an empty Q-table.

        Args:
            n_row: Number of ROW (cop / maximizer) actions.
            n_col: Number of COLUMN (thief / minimizer) actions.
            alpha: Tabular Q learning rate in (0, 1].
            gamma: Discount factor in [0, 1).
        """
        self._n_row = int(n_row)
        self._n_col = int(n_col)
        self._alpha = float(alpha)
        self._gamma = float(gamma)
        self._q: dict[object, np.ndarray] = {}
        self._sol: dict[object, tuple[float, np.ndarray, np.ndarray]] = {}

    @property
    def alpha(self) -> float:
        """The current (possibly annealed) learning rate."""
        return self._alpha

    def anneal(self, alpha_end: float, decay: float) -> None:
        """Geometrically decay the learning rate toward ``alpha_end`` (Littman 1994).

        Constant alpha does NOT converge — the minimax value drifts past its bound; a
        decaying rate (Robbins-Monro) is what makes Minimax-Q settle to the equilibrium.
        """
        self._alpha = max(float(alpha_end), self._alpha * float(decay))

    def q(self, state: object) -> np.ndarray:
        """Return the (mutable) ``(n_row, n_col)`` Q-matrix for ``state`` (lazy zeros).

        Mutate it only BEFORE the first :meth:`value`/strategy read of ``state`` — the LP
        solution is cached and invalidated by :meth:`update`, not by direct mutation.
        """
        if state not in self._q:
            self._q[state] = np.zeros((self._n_row, self._n_col), dtype=np.float64)
        return self._q[state]

    def _solve(self, state: object) -> tuple[float, np.ndarray, np.ndarray]:
        """Cache + return ``(game value, cop maximin π, thief minimax π)`` for ``state``."""
        sol = self._sol.get(state)
        if sol is None:
            mat = self.q(state)
            cop_pi, value = solve_minimax_value(mat)
            thief_pi, _ = solve_minimax_value(-mat.T)
            sol = (value, cop_pi, thief_pi)
            self._sol[state] = sol
        return sol

    def value(self, state: object) -> float:
        """Return the minimax game value of ``state`` (the LP value of its Q-matrix)."""
        return self._solve(state)[0]

    def cop_strategy(self, state: object) -> np.ndarray:
        """Return the ROW (cop, maximizer) maximin mixed strategy at ``state``."""
        return self._solve(state)[1]

    def thief_strategy(self, state: object) -> np.ndarray:
        """Return the COLUMN (thief, minimizer) optimal mixed strategy at ``state``.

        The minimizer's maximin game is the transposed, negated payoff, so the thief's
        strategy is the row solution of ``-Q.T``.
        """
        return self._solve(state)[2]

    def update(  # noqa: PLR0913 — a transition is exactly (s, a_row, a_col, r, s', done)
        self,
        state: object,
        a_row: int,
        a_col: int,
        reward: float,
        next_state: object,
        terminated: bool,
    ) -> None:
        """Apply the Minimax-Q update (eq 2.1) for one ``(s, a_row, a_col)`` transition.

        ``Q(s,r,c) <- (1-alpha)*Q(s,r,c) + alpha*[reward + gamma*minimax(Q[s'])]``, with the
        bootstrap term zero on a terminal step.

        Args:
            state: The current state key.
            a_row: The row (cop) action index taken.
            a_col: The column (thief) action index taken.
            reward: The ROW player's immediate zero-sum payoff.
            next_state: The next state key.
            terminated: Whether ``next_state`` is absorbing.
        """
        bootstrap = 0.0 if terminated else self.value(next_state)
        target = reward + self._gamma * bootstrap
        cur = self.q(state)
        cur[a_row, a_col] = (1.0 - self._alpha) * cur[a_row, a_col] + self._alpha * target
        self._sol.pop(state, None)  # Q[state] changed -> invalidate its cached LP solution
