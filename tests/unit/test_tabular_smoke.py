"""Unit tests for src.marl.replay.tabular_smoke (T3.3 THROWAWAY smoke).

Pins the discrete ``state_key`` (distinct per distinct state), the tabular
Q-learning ``update`` (moves Q toward the TD target), the legal-mask-respecting
``greedy_action``, and the end-to-end ``run_smoke`` 2x2 optimal-capture gate
(greedy cop captures the heuristic thief in min-Manhattan steps over all seeds,
NaN-free). The learner under test is deleted in P4 (the GRU CTDE learner).
"""

from __future__ import annotations

import math

from src.marl.env.actions import Action
from src.marl.replay._smoke_solve import eval_optimal
from src.marl.replay.tabular_smoke import TabularQLearner, run_smoke, state_key


def test_state_key_distinct_per_distinct_state(make_state):
    """Two states differing in any discrete field hash to different keys."""
    base = make_state(cop_pos=(0, 0), thief_pos=(1, 1), h=2, w=2)
    diff_cop = make_state(cop_pos=(0, 1), thief_pos=(1, 1), h=2, w=2)
    diff_thief = make_state(cop_pos=(0, 0), thief_pos=(1, 0), h=2, w=2)
    diff_step = make_state(cop_pos=(0, 0), thief_pos=(1, 1), step=3, h=2, w=2)
    diff_barrier = make_state(cop_pos=(0, 0), thief_pos=(1, 1), barriers=[(0, 1)], barriers_used=1, h=2, w=2)
    keys = {
        state_key(base),
        state_key(diff_cop),
        state_key(diff_thief),
        state_key(diff_step),
        state_key(diff_barrier),
    }
    assert len(keys) == 5


def test_state_key_equal_for_identical_states(make_state):
    """Two independently built identical states produce the same key (hashable)."""
    a = make_state(cop_pos=(0, 0), thief_pos=(1, 1), h=2, w=2)
    b = make_state(cop_pos=(0, 0), thief_pos=(1, 1), h=2, w=2)
    assert state_key(a) == state_key(b)
    assert hash(state_key(a)) == hash(state_key(b))


def test_update_moves_q_toward_target(cfg):
    """A single (non-terminal) update nudges Q(s,a) toward r + gamma*max_legal Q'."""
    learner = TabularQLearner(cfg)
    s, ns = ("s",), ("ns",)
    legal = [True, True, True, True, False]
    learner.update(s, "cop_0", int(Action.DOWN), reward=1.0, next_s_key=ns, next_legal=legal, done=False)
    # From Q=0 with a fresh next (all Q'=0): new Q = alpha * (r + 0 - 0).
    alpha = cfg["p3_smoke"]["alpha"]
    assert math.isclose(learner.q_value(s, "cop_0", int(Action.DOWN)), alpha * 1.0)
    # A second update moves it strictly closer to the target (monotone).
    before = learner.q_value(s, "cop_0", int(Action.DOWN))
    learner.update(s, "cop_0", int(Action.DOWN), reward=1.0, next_s_key=ns, next_legal=legal, done=False)
    after = learner.q_value(s, "cop_0", int(Action.DOWN))
    assert before < after <= 1.0


def test_update_terminal_uses_reward_only_no_bootstrap(cfg):
    """At a terminal transition (done=True) the target is reward ONLY (no gamma*Q')."""
    learner = TabularQLearner(cfg)
    s, ns = ("s",), ("ns",)
    legal = [True, True, True, True, False]
    # Seed a NONZERO next-Q so a leaking bootstrap would be detectable.
    learner.set_q_value(ns, "cop_0", int(Action.UP), 5.0)
    learner.update(s, "cop_0", int(Action.DOWN), reward=1.0, next_s_key=ns, next_legal=legal, done=True)
    # Terminal target = r only -> new Q = alpha * (r - 0); the 5.0 next-Q is ignored.
    alpha = cfg["p3_smoke"]["alpha"]
    assert math.isclose(learner.q_value(s, "cop_0", int(Action.DOWN)), alpha * 1.0)


def test_update_nonterminal_applies_gamma_bootstrap(cfg):
    """A non-terminal update with NONZERO next-Q applies the gamma*max_legal Q' term."""
    learner = TabularQLearner(cfg)
    s, ns = ("s",), ("ns",)
    legal = [True, True, True, True, False]
    # Best legal next-Q = 4.0 (UP); the illegal slot (index 4) has a higher Q ignored.
    learner.set_q_value(ns, "cop_0", int(Action.UP), 4.0)
    learner.set_q_value(ns, "cop_0", 4, 99.0)  # illegal -> must be excluded from max
    learner.update(s, "cop_0", int(Action.DOWN), reward=1.0, next_s_key=ns, next_legal=legal, done=False)
    alpha, gamma = cfg["p3_smoke"]["alpha"], cfg["algo"]["gamma"]
    expected = alpha * (1.0 + gamma * 4.0)  # target = r + gamma*max_legal Q', from Q=0
    assert math.isclose(learner.q_value(s, "cop_0", int(Action.DOWN)), expected)


def test_greedy_action_respects_legal_mask(cfg):
    """greedy_action never returns a masked-out action even if its Q is highest."""
    learner = TabularQLearner(cfg)
    s = ("s",)
    # Make the illegal action (UP) the most valuable; it must NOT be chosen.
    learner.set_q_value(s, "cop_0", int(Action.UP), 10.0)
    learner.set_q_value(s, "cop_0", int(Action.RIGHT), 1.0)
    legal = [False, False, False, True, False]  # only RIGHT legal
    assert learner.greedy_action(s, "cop_0", legal) == Action.RIGHT


def test_greedy_action_lowest_index_tie_break(cfg):
    """Among equal-Q legal actions the lowest Action index wins."""
    learner = TabularQLearner(cfg)
    s = ("s",)
    legal = [True, True, True, True, False]  # all four moves, equal Q=0
    assert learner.greedy_action(s, "cop_0", legal) == Action.UP


def test_run_smoke_reaches_optimal_all_seeds_nan_free(cfg):
    """run_smoke over ALL training.seeds: pursuit-optimal capture from every 2x2
    start + NaN-free Q-values (the thief flees, so the gate is pursuit-optimal, not
    naive min-Manhattan; tabular is cheap so all 5 seeds run)."""
    seeds = cfg["training"]["seeds"]
    result = run_smoke(cfg, seeds)
    assert result["optimal"] is True
    assert result["nan_free"] is True
    assert result["seeds"] == list(seeds)
    # Every 2x2 start captured the fleeing thief within the pursuit-optimal budget.
    assert result["all_captured"] is True


def test_untrained_q_fails_optimal_gate(cfg):
    """NEGATIVE control: a zeroed/untrained Q-table FAILS the optimal gate.

    Proves the eval_optimal gate is not vacuous — an all-zero Q (greedy picks the
    lowest-index legal action everywhere) cannot pursue the fleeing thief optimally
    from every 2x2 start, so the gate must reject it.
    """
    untrained = TabularQLearner(cfg)  # empty dict Q -> every Q-value is 0.0
    assert eval_optimal(cfg, untrained) is False
