"""Unit tests for src/marl/env/reward.py (T1.4) — RL training signal only.

Covers the potential function (numeric shaping identity, terminal zero, -min over
the cop team), and RewardModel.compute (shaping toggles off at eval, dec_pomdp
cop entries equal + thief = -team, capture/timeout terminal signs).
"""

from __future__ import annotations

import copy

import pytest

from src.marl.env.reward import RewardModel, potential


def test_potential_numeric_3x3(make_state):
    # 3x3, cop(0,0) thief(2,2): manhattan=4, d_max=(3-1)+(3-1)=4 -> Phi=-1.0.
    state = make_state(cop_pos=(0, 0), thief_pos=(2, 2), h=3, w=3)
    assert potential(state) == pytest.approx(-1.0)


def test_potential_half_distance(make_state):
    # 5x5 d_max=8; cop(0,0) thief(0,4): manhattan=4 -> Phi=-0.5.
    state = make_state(cop_pos=(0, 0), thief_pos=(0, 4), h=5, w=5)
    assert potential(state) == pytest.approx(-0.5)


def test_potential_terminal_is_zero(make_state):
    # Even with non-zero distance, a terminal state has Phi=0 (Ng-1999 invariance).
    state = make_state(cop_pos=(0, 0), thief_pos=(2, 2), h=3, w=3, terminal=True)
    assert potential(state) == 0.0


def test_potential_min_over_two_cops_picks_nearer(make_state):
    # thief(2,2); cop_a(0,0) dist=4, cop_b(2,1) dist=1 -> -min uses nearer (1).
    state = make_state(cop_pos=[(0, 0), (2, 1)], thief_pos=(2, 2), h=3, w=3)
    assert potential(state) == pytest.approx(-1.0 / 4.0)


def test_potential_1x1_grid_no_zero_division(make_state):
    # Degenerate 1x1 grid: d_max=(0)+(0)=0 -> guard max(1, ...) avoids /0.
    # cop==thief so manhattan=0 -> Phi=-0/1=0.0; must NOT raise ZeroDivisionError.
    state = make_state(cop_pos=(0, 0), thief_pos=(0, 0), h=1, w=1)
    assert potential(state) == 0.0


def test_shaping_identity_numeric(cfg, make_state):
    # 3x3: prev cop(0,0) thief(2,2) Phi=-1.0; nxt cop(1,1) thief(2,2) dist=2 Phi=-0.5.
    prev = make_state(cop_pos=(0, 0), thief_pos=(2, 2), h=3, w=3, step=0)
    nxt = make_state(cop_pos=(1, 1), thief_pos=(2, 2), h=3, w=3, step=1)
    gamma = cfg["algo"]["gamma"]
    dw = cfg["reward"]["distance_weight"]
    sp = cfg["reward"]["step_penalty"]
    expected_f = dw * (gamma * potential(nxt) - potential(prev))
    rm = RewardModel(cfg)
    out = rm.compute(prev, nxt, winner=None, eval_mode=False)
    assert out["cop_0"] == pytest.approx(expected_f - sp)


def test_eval_mode_zeroes_shaping(cfg, make_state):
    # shaping_eval_enabled is False -> at eval only -step_penalty remains.
    prev = make_state(cop_pos=(0, 0), thief_pos=(2, 2), h=3, w=3, step=0)
    nxt = make_state(cop_pos=(1, 1), thief_pos=(2, 2), h=3, w=3, step=1)
    rm = RewardModel(cfg)
    out = rm.compute(prev, nxt, winner=None, eval_mode=True)
    assert out["cop_0"] == pytest.approx(-cfg["reward"]["step_penalty"])


def test_dec_pomdp_cop_entries_equal_and_thief_negates(cfg, make_state):
    # Two cops -> both cop entries equal (shared team reward); thief = -team.
    prev = make_state(cop_pos=[(0, 0), (4, 4)], thief_pos=(2, 2), h=5, w=5, step=0)
    nxt = make_state(cop_pos=[(1, 0), (4, 4)], thief_pos=(2, 2), h=5, w=5, step=1)
    rm = RewardModel(cfg)
    out = rm.compute(prev, nxt, winner=None, eval_mode=False)
    assert out["cop_0"] == pytest.approx(out["cop_1"])
    assert out["thief"] == pytest.approx(-out["cop_0"])


def test_capture_terminal_sign(cfg, make_state):
    # Capture (winner=cop): cop team gets +capture_bonus on top of step terms.
    prev = make_state(cop_pos=(0, 0), thief_pos=(1, 0), h=5, w=5, step=4)
    nxt = make_state(cop_pos=(1, 0), thief_pos=(1, 0), h=5, w=5, step=5, terminal=True)
    rm = RewardModel(cfg)
    out = rm.compute(prev, nxt, winner="cop", eval_mode=False)
    no_win = RewardModel(cfg).compute(prev, nxt, winner=None, eval_mode=False)
    assert out["cop_0"] == pytest.approx(no_win["cop_0"] + cfg["reward"]["capture_bonus"])
    assert out["cop_0"] > 0


def test_timeout_terminal_sign(cfg, make_state):
    # Timeout (winner=thief): cop team gets -timeout_penalty; cop reward negative.
    prev = make_state(cop_pos=(0, 0), thief_pos=(4, 4), h=5, w=5, step=24)
    nxt = make_state(cop_pos=(0, 1), thief_pos=(4, 4), h=5, w=5, step=25, terminal=True)
    rm = RewardModel(cfg)
    out = rm.compute(prev, nxt, winner="thief", eval_mode=False)
    no_win = RewardModel(cfg).compute(prev, nxt, winner=None, eval_mode=False)
    assert out["cop_0"] == pytest.approx(no_win["cop_0"] - cfg["reward"]["timeout_penalty"])
    assert out["cop_0"] < 0


def test_barrier_cost_applied_on_placement(cfg, make_state):
    # barriers_used delta of 1 -> cop team pays -barrier_cost vs no placement.
    prev = make_state(cop_pos=(0, 0), thief_pos=(2, 2), h=5, w=5, barriers_used=0, step=0)
    placed = make_state(
        cop_pos=(0, 0), thief_pos=(2, 2), h=5, w=5, barriers_used=1, step=1, barriers=[(0, 1)]
    )
    no_place = make_state(cop_pos=(0, 0), thief_pos=(2, 2), h=5, w=5, barriers_used=0, step=1)
    rm = RewardModel(cfg)
    with_cost = rm.compute(prev, placed, winner=None, eval_mode=False)
    without = rm.compute(prev, no_place, winner=None, eval_mode=False)
    assert with_cost["cop_0"] == pytest.approx(without["cop_0"] - cfg["reward"]["barrier_cost"])


def test_posg_thief_role_specific(cfg, make_state):
    # In posg mode the thief gets evade_step per non-terminal step (not -team).
    cfg_posg = dict(cfg)
    cfg_posg["env"] = {**cfg["env"], "reward_mode": "posg"}
    prev = make_state(cop_pos=(0, 0), thief_pos=(2, 2), h=5, w=5, step=0)
    nxt = make_state(cop_pos=(1, 0), thief_pos=(2, 2), h=5, w=5, step=1)
    rm = RewardModel(cfg_posg)
    out = rm.compute(prev, nxt, winner=None, eval_mode=False)
    assert out["thief"] == pytest.approx(cfg["reward"]["evade_step"])


def test_posg_thief_timeout_bonus(cfg, make_state):
    cfg_posg = dict(cfg)
    cfg_posg["env"] = {**cfg["env"], "reward_mode": "posg"}
    prev = make_state(cop_pos=(0, 0), thief_pos=(4, 4), h=5, w=5, step=24)
    nxt = make_state(cop_pos=(0, 1), thief_pos=(4, 4), h=5, w=5, step=25, terminal=True)
    rm = RewardModel(cfg_posg)
    out = rm.compute(prev, nxt, winner="thief", eval_mode=False)
    assert out["thief"] == pytest.approx(cfg["reward"]["timeout_bonus"])


def test_shaping_disabled_only_step_penalty(cfg, make_state):
    # shaping_enabled=False (deep-copy cfg) -> no shaping term even in training:
    # a non-terminal, non-eval cop transition collapses to exactly -step_penalty.
    cfg_off = copy.deepcopy(cfg)
    cfg_off["reward"]["shaping_enabled"] = False
    prev = make_state(cop_pos=(0, 0), thief_pos=(2, 2), h=3, w=3, step=0)
    nxt = make_state(cop_pos=(1, 1), thief_pos=(2, 2), h=3, w=3, step=1)
    rm = RewardModel(cfg_off)
    out = rm.compute(prev, nxt, winner=None, eval_mode=False)
    assert out["cop_0"] == pytest.approx(-cfg["reward"]["step_penalty"])


def test_distance_weight_is_real_coefficient(cfg, make_state):
    # distance_weight=3.0 (deep-copy cfg) pins dw as a true scalar on the
    # potential delta: cop_0 == 3.0*(gamma*Phi(nxt)-Phi(prev)) - step_penalty.
    cfg_dw = copy.deepcopy(cfg)
    cfg_dw["reward"]["distance_weight"] = 3.0
    prev = make_state(cop_pos=(0, 0), thief_pos=(2, 2), h=3, w=3, step=0)
    nxt = make_state(cop_pos=(1, 1), thief_pos=(2, 2), h=3, w=3, step=1)
    gamma = cfg["algo"]["gamma"]
    sp = cfg["reward"]["step_penalty"]
    expected_f = 3.0 * (gamma * potential(nxt) - potential(prev))
    rm = RewardModel(cfg_dw)
    out = rm.compute(prev, nxt, winner=None, eval_mode=False)
    assert out["cop_0"] == pytest.approx(expected_f - sp)


def test_terminal_value_includes_shaping(cfg, make_state):
    # Full terminal VALUE incl shaping with winner=None: Phi(nxt)=0 (terminal) so
    # F = dw*(gamma*0 - Phi(prev)) = dw*(-Phi(prev)); cop_0 = F - step_penalty.
    prev = make_state(cop_pos=(0, 0), thief_pos=(1, 0), h=5, w=5, step=4)
    nxt = make_state(cop_pos=(1, 0), thief_pos=(1, 0), h=5, w=5, step=5, terminal=True)
    dw = cfg["reward"]["distance_weight"]
    sp = cfg["reward"]["step_penalty"]
    rm = RewardModel(cfg)
    out = rm.compute(prev, nxt, winner=None, eval_mode=False)
    assert out["cop_0"] == pytest.approx(dw * (-potential(prev)) - sp)


def test_posg_capture_terminal_thief_zero_cop_bonus(cfg, make_state):
    # posg + winner=="cop" terminal: thief gets 0.0 (only timeout pays the thief),
    # while the cop team still earns +capture_bonus on top of the step terms.
    cfg_posg = copy.deepcopy(cfg)
    cfg_posg["env"]["reward_mode"] = "posg"
    prev = make_state(cop_pos=(0, 0), thief_pos=(1, 0), h=5, w=5, step=4)
    nxt = make_state(cop_pos=(1, 0), thief_pos=(1, 0), h=5, w=5, step=5, terminal=True)
    rm = RewardModel(cfg_posg)
    out = rm.compute(prev, nxt, winner="cop", eval_mode=False)
    no_win = RewardModel(cfg_posg).compute(prev, nxt, winner=None, eval_mode=False)
    assert out["thief"] == 0.0
    assert out["cop_0"] == pytest.approx(no_win["cop_0"] + cfg["reward"]["capture_bonus"])
