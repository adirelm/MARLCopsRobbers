"""Unit tests for src/marl/env/transition.py (T2.1) — simultaneous joint resolution.

Covers the frozen TransitionResult contract and resolve_joint_action: same-cell
capture, cop<->thief swap counts as capture, both-stay on a barrier/edge,
single-cop barrier budget cap, 2-cop simultaneous OVER-budget (TEAM cap,
deterministic by cop index), timeout exact tie (step+1==max_moves), the
capture-on-final-move > timeout precedence, step increment, and PURITY
(same input -> same output, no RNG). The validated P2 fixes (co-located
double-PLACE, the ``env.capture_on_swap`` / ``env.move_resolution`` toggles,
2-cop swap-capture, 2-cop double-occupancy) live in test_transition_p2_fixes.py.
"""

from __future__ import annotations

import copy
import dataclasses

from src.marl.env.actions import Action
from src.marl.env.transition import TransitionResult, resolve_joint_action
from src.marl.env.types import GlobalState


def test_transition_result_is_frozen():
    res = TransitionResult(
        next_state=GlobalState((0, 0), (0, 1), frozenset(), 0, 1, 5, 5),
        winner=None,
        capture=False,
    )
    assert dataclasses.is_dataclass(res)
    fields = {f.name for f in dataclasses.fields(res)}
    assert fields == {"next_state", "winner", "capture"}
    try:
        res.capture = True  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:  # pragma: no cover - guard the frozen contract
        raise AssertionError("TransitionResult must be frozen")


def test_same_cell_capture(cfg, make_state):
    # Cop one step left of thief; cop moves RIGHT onto the thief's old cell while
    # the thief moves RIGHT away -> cop ends on thief's PRE-move cell? No: both
    # resolve from pre-move state. Use a head-on: cop below thief moves UP, thief
    # stays-blocked so they collide on the thief cell.
    state = make_state(cop_pos=(2, 2), thief_pos=(1, 2), step=3)
    joint = {"cop_0": Action.UP, "thief": Action.UP}  # thief blocked? no -> moves up
    res = resolve_joint_action(state, joint, cfg)
    # cop -> (1,2) [thief old cell], thief -> (0,2). No same-cell, no swap.
    assert res.capture is False
    # Now make the thief unable to move (top edge) so cop lands on it.
    state2 = make_state(cop_pos=(1, 2), thief_pos=(0, 2), step=3)
    joint2 = {"cop_0": Action.UP, "thief": Action.UP}  # thief at top -> stay
    res2 = resolve_joint_action(state2, joint2, cfg)
    assert res2.next_state.cop_pos == ((0, 2),)
    assert res2.next_state.thief_pos == (0, 2)
    assert res2.capture is True
    assert res2.winner == "cop"
    assert res2.next_state.terminal is True


def test_swap_counts_as_capture(cfg, make_state):
    # Adjacent cop and thief exchange cells in one tick.
    state = make_state(cop_pos=(2, 2), thief_pos=(2, 3), step=1)
    joint = {"cop_0": Action.RIGHT, "thief": Action.LEFT}
    res = resolve_joint_action(state, joint, cfg)
    assert res.next_state.cop_pos == ((2, 3),)
    assert res.next_state.thief_pos == (2, 2)
    assert res.capture is True
    assert res.winner == "cop"
    assert res.next_state.terminal is True


def test_both_stay_on_edge_and_barrier(cfg, make_state):
    # Cop boxed by a barrier above; thief at the top edge moving up -> both stay.
    state = make_state(
        cop_pos=(2, 2),
        thief_pos=(0, 4),
        barriers=[(1, 2)],
        barriers_used=1,
        step=2,
    )
    joint = {"cop_0": Action.UP, "thief": Action.UP}
    res = resolve_joint_action(state, joint, cfg)
    assert res.next_state.cop_pos == ((2, 2),)  # barrier blocks -> stay
    assert res.next_state.thief_pos == (0, 4)  # top edge -> stay
    assert res.capture is False
    assert res.winner is None
    assert res.next_state.terminal is False


def test_single_cop_barrier_budget_cap(cfg, make_state):
    # With the team budget already exhausted, a PLACE drops to a stay (no barrier).
    max_b = cfg["game"]["max_barriers"]
    state = make_state(
        cop_pos=(2, 2),
        thief_pos=(4, 4),
        barriers_used=max_b,
        step=1,
    )
    joint = {"cop_0": Action.PLACE_BARRIER, "thief": Action.UP}
    res = resolve_joint_action(state, joint, cfg)
    assert res.next_state.barriers_used == max_b  # not incremented
    assert (2, 2) not in res.next_state.barriers
    assert res.next_state.cop_pos == ((2, 2),)  # placement-stay, cop unmoved

    # Below the cap, the placement IS honored and the cop stays on the placed cell.
    state2 = make_state(cop_pos=(2, 2), thief_pos=(4, 4), barriers_used=0, step=1)
    res2 = resolve_joint_action(state2, joint, cfg)
    assert res2.next_state.barriers_used == 1
    assert (2, 2) in res2.next_state.barriers
    assert res2.next_state.cop_pos == ((2, 2),)


def test_two_cop_over_budget_team_cap_deterministic(cfg, make_state):
    # Team has exactly one barrier left; BOTH cops PLACE simultaneously. The
    # lower-index cop's PLACE is honored, the higher-index cop's PLACE drops to a
    # stay (deterministic by cop index).
    max_b = cfg["game"]["max_barriers"]
    state = make_state(
        cop_pos=[(1, 1), (3, 3)],
        thief_pos=(0, 4),
        barriers_used=max_b - 1,
        step=1,
    )
    joint = {
        "cop_0": Action.PLACE_BARRIER,
        "cop_1": Action.PLACE_BARRIER,
        "thief": Action.DOWN,
    }
    res = resolve_joint_action(state, joint, cfg)
    assert res.next_state.barriers_used == max_b  # only ONE honored
    assert (1, 1) in res.next_state.barriers  # cop_0 (lower index) wins the slot
    assert (3, 3) not in res.next_state.barriers  # cop_1 dropped to a stay
    assert res.next_state.cop_pos == ((1, 1), (3, 3))  # both stayed on their cells


def test_timeout_exact_tie(cfg, make_state):
    # (step+1) == max_moves with no capture -> thief wins by timeout.
    max_moves = cfg["game"]["max_moves"]
    state = make_state(cop_pos=(0, 0), thief_pos=(4, 4), step=max_moves - 1)
    joint = {"cop_0": Action.DOWN, "thief": Action.UP}
    res = resolve_joint_action(state, joint, cfg)
    assert res.next_state.step == max_moves
    assert res.capture is False
    assert res.winner == "thief"
    assert res.next_state.terminal is True


def test_capture_on_final_move_beats_timeout(cfg, make_state):
    # On the very last move the cop captures: capture is checked BEFORE timeout,
    # so winner is "cop", not "thief".
    max_moves = cfg["game"]["max_moves"]
    state = make_state(cop_pos=(0, 1), thief_pos=(0, 0), step=max_moves - 1)
    joint = {"cop_0": Action.LEFT, "thief": Action.RIGHT}  # swap-capture
    res = resolve_joint_action(state, joint, cfg)
    assert res.next_state.step == max_moves
    assert res.capture is True
    assert res.winner == "cop"
    assert res.next_state.terminal is True


def test_step_increments_and_dims_preserved(cfg, make_state):
    state = make_state(cop_pos=(0, 0), thief_pos=(4, 4), step=7, h=5, w=5)
    joint = {"cop_0": Action.DOWN, "thief": Action.UP}
    res = resolve_joint_action(state, joint, cfg)
    assert res.next_state.step == 8
    assert res.next_state.h == 5
    assert res.next_state.w == 5
    assert res.winner is None
    assert res.next_state.terminal is False


def test_purity_same_input_same_output(cfg, make_state):
    state = make_state(
        cop_pos=[(1, 1), (3, 3)],
        thief_pos=(2, 2),
        barriers=[(0, 0)],
        barriers_used=1,
        step=4,
    )
    joint = {
        "cop_0": Action.PLACE_BARRIER,
        "cop_1": Action.RIGHT,
        "thief": Action.LEFT,
    }
    snapshot = copy.deepcopy(state)
    res_a = resolve_joint_action(state, joint, cfg)
    res_b = resolve_joint_action(state, joint, cfg)
    assert res_a == res_b  # frozen dataclasses + frozen state -> value-equal
    assert state == snapshot  # input state not mutated
