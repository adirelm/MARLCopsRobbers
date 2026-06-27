"""RED→GREEN tests for the zero-sum maximin LP (P-bonus, L11 §2.2.1 / §5).

The L11 worked example proves the naive ``max(min(...))`` under-values the game
(it gives the pure-strategy value -1) while the true value needs the mixed-strategy
LP (p = 0.4, V = 1.0, eq 2.3). These gates pin that, plus a pure saddle + RPS.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.marl.baselines.minimax_lp import solve_minimax_value


def test_l11_worked_example_mixed_strategy():
    """L11 §2.2.1: payoff ``[[-2,4],[3,-1]]`` → mixed ``π=[0.4,0.6]``, value ``V=1.0`` (eq 2.3)."""
    pi, v = solve_minimax_value(np.array([[-2.0, 4.0], [3.0, -1.0]]))
    assert v == pytest.approx(1.0, abs=1e-6)
    assert pi == pytest.approx([0.4, 0.6], abs=1e-6)
    assert float(pi.sum()) == pytest.approx(1.0, abs=1e-9)
    assert (pi >= -1e-9).all()


def test_pure_saddle_point():
    """A game with a pure saddle (maximin == minimax): ``[[4,3],[2,1]]`` → V=3, row-0 pure."""
    pi, v = solve_minimax_value(np.array([[4.0, 3.0], [2.0, 1.0]]))
    assert v == pytest.approx(3.0, abs=1e-6)
    assert pi[0] == pytest.approx(1.0, abs=1e-6)


def test_symmetric_zero_sum_has_value_zero():
    """Rock-paper-scissors is symmetric → game value 0, uniform mixed strategy."""
    rps = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
    pi, v = solve_minimax_value(rps)
    assert v == pytest.approx(0.0, abs=1e-6)
    assert pi == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-6)


def test_rejects_non_2d_payoff():
    """A non-2-D payoff raises ``ValueError`` (input guard)."""
    with pytest.raises(ValueError, match="2-D"):
        solve_minimax_value(np.array([1.0, 2.0, 3.0]))
