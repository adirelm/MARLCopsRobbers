"""Maximin LP for a zero-sum matrix game (P-bonus, L11 §2.2 / §5).

The L11 worked example shows the naive ``max(min(·))`` UNDER-values a game (it returns
the pure-strategy value), whereas the true value needs the **mixed-strategy** linear
program. :func:`solve_minimax_value` returns the row (maximizer) optimal mixed strategy
and the game value ``V`` via :func:`scipy.optimize.linprog`; the column player minimizes.
Used by :class:`~src.marl.baselines.minimax_q.MinimaxQ` to evaluate each next state.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

_NDIM = 2  # a zero-sum payoff is a 2-D (rows x cols) matrix


def solve_minimax_value(payoff: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the maximin mixed strategy + game value of a zero-sum payoff matrix.

    Solves, for the ROW player (maximizer), the linear program

        maximize  V   s.t.   sum_i pi_i * payoff[i, j] >= V   for every column j,
                             sum_i pi_i = 1,   pi_i >= 0,

    where ``payoff[i, j]`` is the row player's payoff when it plays ``i`` and the
    (minimizing) column player plays ``j``. The decision vector is
    ``x = [pi_1 ... pi_m, V]``; ``maximize V`` is encoded as ``minimize -V``.

    Args:
        payoff: An ``(m, n)`` zero-sum payoff matrix from the row player's view.

    Returns:
        ``(π, V)`` — ``π`` is the ``(m,)`` maximin mixed strategy (sums to 1, ≥ 0) and
        ``V`` is the scalar game value.

    Raises:
        ValueError: If ``payoff`` is not a 2-D array, or the LP fails to solve.
    """
    p = np.asarray(payoff, dtype=np.float64)
    if p.ndim != _NDIM:
        raise ValueError(f"payoff must be a 2-D (m, n) matrix; got shape {p.shape}")
    m, n = p.shape
    # x = [pi_1..pi_m, V]; maximize V  ==  minimize -V
    c = np.concatenate([np.zeros(m), [-1.0]])
    # for each column j:  sum_i pi_i P[i,j] >= V   <=>   -(P^T) @ pi + V <= 0
    a_ub = np.hstack([-p.T, np.ones((n, 1))])
    b_ub = np.zeros(n)
    # sum_i pi_i = 1   (V coefficient 0)
    a_eq = np.concatenate([np.ones(m), [0.0]]).reshape(1, m + 1)
    b_eq = np.array([1.0])
    bounds = [(0.0, 1.0)] * m + [(None, None)]
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise ValueError(f"minimax LP failed: {res.message}")
    return res.x[:m], float(res.x[m])
