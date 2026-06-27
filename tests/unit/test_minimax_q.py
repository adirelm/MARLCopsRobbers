"""RED->GREEN tests for the tabular Minimax-Q learner (P-bonus, L11 §2.2 / eq 2.1)."""

from __future__ import annotations

import numpy as np
import pytest

from src.marl.baselines.minimax_q import MinimaxQ

_L11 = np.array([[-2.0, 4.0], [3.0, -1.0]])  # L11 §2.2.1 example; game value 1.0


def test_terminal_update_moves_q_toward_reward():
    """Eq 2.1 on a terminal step: Q(s,0,0) = (1-a)*0 + a*(r + 0) = a*r."""
    learner = MinimaxQ(n_row=2, n_col=2, alpha=0.1, gamma=0.95)
    learner.update("s0", 0, 0, reward=1.0, next_state="s1", terminated=True)
    assert learner.q("s0")[0, 0] == pytest.approx(0.1)


def test_value_of_unseen_state_is_zero():
    """An unseen state lazily inits to a zero Q-matrix → game value 0."""
    assert MinimaxQ(n_row=2, n_col=2, alpha=0.1, gamma=0.9).value("new") == pytest.approx(0.0)


def test_q_table_converges_to_game_value():
    """Repeated terminal updates from the L11 payoff drive value(s) -> the LP game value 1.0."""
    learner = MinimaxQ(n_row=2, n_col=2, alpha=0.2, gamma=0.95)
    for _ in range(800):
        for i in range(2):
            for j in range(2):
                learner.update("s", i, j, float(_L11[i, j]), "s", terminated=True)
    assert learner.value("s") == pytest.approx(1.0, abs=1e-2)


def test_anneal_decays_alpha_but_clamps_at_floor():
    """anneal() geometrically decays alpha and never drops below the floor (Littman 1994)."""
    learner = MinimaxQ(n_row=2, n_col=2, alpha=0.5, gamma=0.95)
    learner.anneal(alpha_end=0.1, decay=0.5)
    assert learner.alpha == pytest.approx(0.25)
    for _ in range(20):
        learner.anneal(alpha_end=0.1, decay=0.5)
    assert learner.alpha == pytest.approx(0.1)  # clamped at the floor, not below


def test_cop_and_thief_strategies_are_distributions():
    """cop_strategy (rows) + thief_strategy (cols) are valid probability vectors."""
    learner = MinimaxQ(n_row=2, n_col=2, alpha=0.1, gamma=0.9)
    learner.q("s")[:] = _L11
    cop, thief = learner.cop_strategy("s"), learner.thief_strategy("s")
    assert float(cop.sum()) == pytest.approx(1.0) and (cop >= -1e-9).all()
    assert float(thief.sum()) == pytest.approx(1.0) and (thief >= -1e-9).all()
    assert cop == pytest.approx([0.4, 0.6], abs=1e-6)  # the L11 maximin row strategy
