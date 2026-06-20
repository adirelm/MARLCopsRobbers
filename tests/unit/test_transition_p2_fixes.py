"""Regression tests for the validated P2 transition fixes (OWNER A).

Covers the adversarial-review defects on src/marl/env/transition.py:
- H: a co-located double-PLACE charges exactly ONE barrier (delta==1, one cell),
  so reward.py never double-charges the team for one physical barrier.
- F: ``env.move_resolution`` is honored — only ``"simultaneous"`` is implemented;
  ``cop_first`` / ``thief_first`` raise NotImplementedError (no silent fallback).
- The ``env.capture_on_swap`` toggle (swap is a capture only when enabled).
- A 2-cop swap-capture (one of two cops swaps cells with the thief).
- 2-cop double-occupancy (both cops onto one cell) is allowed and never crashes.
"""

from __future__ import annotations

import copy

import pytest

from src.marl.env.actions import Action
from src.marl.env.transition import resolve_joint_action


def test_colocated_double_place_charges_one_barrier(cfg, make_state):
    # Two CO-LOCATED cops both PLACE on the SAME cell. Only ONE distinct barrier
    # cell is placed, so barriers_used must advance by exactly 1 (else reward.py
    # double-charges the team for one physical barrier).
    state = make_state(
        cop_pos=[(2, 2), (2, 2)],
        thief_pos=(0, 4),
        barriers_used=0,
        step=1,
    )
    joint = {
        "cop_0": Action.PLACE_BARRIER,
        "cop_1": Action.PLACE_BARRIER,
        "thief": Action.DOWN,
    }
    res = resolve_joint_action(state, joint, cfg)
    assert res.next_state.barriers_used - state.barriers_used == 1  # delta == 1
    assert len(res.next_state.barriers) == 1  # one distinct placed cell
    assert (2, 2) in res.next_state.barriers
    assert res.next_state.cop_pos == ((2, 2), (2, 2))  # both stayed (placement-stay)


def test_swap_capture_disabled_via_config_override(cfg, make_state):
    # With env.capture_on_swap=False, a clean cop<->thief swap is NOT a capture.
    no_swap_cfg = copy.deepcopy(cfg)
    no_swap_cfg["env"]["capture_on_swap"] = False
    state = make_state(cop_pos=(2, 2), thief_pos=(2, 3), step=1)
    joint = {"cop_0": Action.RIGHT, "thief": Action.LEFT}
    res = resolve_joint_action(state, joint, no_swap_cfg)
    assert res.next_state.cop_pos == ((2, 3),)  # cells exchanged
    assert res.next_state.thief_pos == (2, 2)
    assert res.capture is False
    assert res.winner is None
    assert res.next_state.terminal is False


def test_two_cop_swap_capture(cfg, make_state):
    # Two cops; the SECOND cop swaps cells with the thief -> capture via swap.
    state = make_state(
        cop_pos=[(0, 0), (2, 2)],
        thief_pos=(2, 3),
        step=1,
    )
    joint = {
        "cop_0": Action.DOWN,
        "cop_1": Action.RIGHT,
        "thief": Action.LEFT,
    }
    res = resolve_joint_action(state, joint, cfg)
    assert res.next_state.cop_pos == ((1, 0), (2, 3))
    assert res.next_state.thief_pos == (2, 2)
    assert res.capture is True
    assert res.winner == "cop"
    assert res.next_state.terminal is True


def test_two_cop_double_occupancy_no_collision(cfg, make_state):
    # Both cops move onto the SAME empty cell -> double occupancy is ALLOWED
    # (cooperative team, 4x4 training): both land there, no crash, no capture.
    state = make_state(
        cop_pos=[(2, 1), (2, 3)],
        thief_pos=(0, 0),
        step=1,
    )
    joint = {
        "cop_0": Action.RIGHT,
        "cop_1": Action.LEFT,
        "thief": Action.DOWN,
    }
    res = resolve_joint_action(state, joint, cfg)
    assert res.next_state.cop_pos == ((2, 2), (2, 2))  # both on the shared cell
    assert res.capture is False
    assert res.winner is None
    assert res.next_state.terminal is False


def test_non_simultaneous_move_resolution_raises(cfg, make_state):
    # env.move_resolution other than "simultaneous" is an honest, unimplemented
    # limitation -> NotImplementedError (live the config key, no silent fallback).
    state = make_state(cop_pos=(2, 2), thief_pos=(0, 4), step=1)
    joint = {"cop_0": Action.UP, "thief": Action.DOWN}
    for mode in ("cop_first", "thief_first"):
        bad_cfg = copy.deepcopy(cfg)
        bad_cfg["env"]["move_resolution"] = mode
        with pytest.raises(NotImplementedError):
            resolve_joint_action(state, joint, bad_cfg)
