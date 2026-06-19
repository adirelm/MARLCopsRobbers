"""Unit tests for src/marl/env/actions.py (T1.2) — action enum + legality mask.

Covers pinned IntEnum integer values, DELTAS, and action_mask: thief PLACE
always masked, OOB move masked, barrier-blocked move masked, cop PLACE masked
when barriers_used==max_barriers, the enable_stay path, role/idx validation,
and the NEVER-all-False MOVES-ONLY boxed-in fallback guarantee.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from src.marl.env.actions import DELTAS, Action, action_mask


def test_intenum_values_pinned():
    assert int(Action.UP) == 0
    assert int(Action.DOWN) == 1
    assert int(Action.LEFT) == 2
    assert int(Action.RIGHT) == 3
    assert int(Action.PLACE_BARRIER) == 4
    assert int(Action.STAY) == 5


def test_move_indices_contiguous():
    assert [int(a) for a in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)] == [0, 1, 2, 3]


def test_deltas_values():
    assert DELTAS[Action.UP] == (-1, 0)
    assert DELTAS[Action.DOWN] == (1, 0)
    assert DELTAS[Action.LEFT] == (0, -1)
    assert DELTAS[Action.RIGHT] == (0, 1)
    assert DELTAS[Action.PLACE_BARRIER] == (0, 0)
    assert DELTAS[Action.STAY] == (0, 0)


def test_mask_length_is_a_cop(cfg, make_state):
    state = make_state(cop_pos=(2, 2), thief_pos=(0, 0))
    mask = action_mask(state, "cop", cfg, idx=0)
    assert mask.dtype == np.bool_
    assert mask.shape == (cfg["env"]["actions"]["a_cop"],)
    assert mask.shape == (5,)


def test_cop_center_all_moves_and_place_legal(cfg, make_state):
    state = make_state(cop_pos=(2, 2), thief_pos=(0, 0))
    mask = action_mask(state, "cop", cfg, idx=0)
    # UP, DOWN, LEFT, RIGHT all legal from the center, PLACE legal (budget left).
    assert mask.tolist() == [True, True, True, True, True]


def test_cop_oob_move_masked(cfg, make_state):
    # Cop at top-left corner: UP and LEFT are OOB.
    state = make_state(cop_pos=(0, 0), thief_pos=(4, 4))
    mask = action_mask(state, "cop", cfg, idx=0)
    assert not mask[int(Action.UP)]
    assert not mask[int(Action.LEFT)]
    assert mask[int(Action.DOWN)]
    assert mask[int(Action.RIGHT)]


def test_cop_barrier_blocked_move_masked(cfg, make_state):
    # Barrier directly below the cop blocks DOWN.
    state = make_state(cop_pos=(2, 2), thief_pos=(0, 0), barriers=[(3, 2)])
    mask = action_mask(state, "cop", cfg, idx=0)
    assert not mask[int(Action.DOWN)]
    assert mask[int(Action.UP)]


def test_thief_place_always_masked(cfg, make_state):
    state = make_state(cop_pos=(0, 0), thief_pos=(2, 2))
    mask = action_mask(state, "thief", cfg, idx=0)
    assert not mask[int(Action.PLACE_BARRIER)]
    # thief still has its four moves from the center.
    assert mask[int(Action.UP)]
    assert mask[int(Action.DOWN)]
    assert mask[int(Action.LEFT)]
    assert mask[int(Action.RIGHT)]


def test_cop_place_masked_when_budget_exhausted(cfg, make_state):
    max_b = cfg["game"]["max_barriers"]
    state = make_state(cop_pos=(2, 2), thief_pos=(0, 0), barriers_used=max_b)
    mask = action_mask(state, "cop", cfg, idx=0)
    assert not mask[int(Action.PLACE_BARRIER)]


def test_cop_place_legal_below_budget(cfg, make_state):
    max_b = cfg["game"]["max_barriers"]
    state = make_state(cop_pos=(2, 2), thief_pos=(0, 0), barriers_used=max_b - 1)
    mask = action_mask(state, "cop", cfg, idx=0)
    assert mask[int(Action.PLACE_BARRIER)]


def test_boxed_in_never_all_false(cfg, make_state):
    # Cop boxed in on all four sides by barriers, and budget exhausted: no legal
    # action exists -> the mask MUST fall back to all-True over the move indices.
    surround = [(0, 1), (1, 0), (1, 2), (2, 1)]
    max_b = cfg["game"]["max_barriers"]
    state = make_state(
        cop_pos=(1, 1),
        thief_pos=(4, 4),
        barriers=surround,
        barriers_used=max_b,
    )
    mask = action_mask(state, "cop", cfg, idx=0)
    assert mask.any(), "action_mask must never be all-False"
    # The 4 move indices are all True in the boxed-in fallback; PLACE stays masked
    # because the budget is exhausted (it is not a move).
    assert mask[int(Action.UP)] and mask[int(Action.DOWN)]
    assert mask[int(Action.LEFT)] and mask[int(Action.RIGHT)]


def test_boxed_in_corner_never_all_false(cfg, make_state):
    # Corner cop with both in-board neighbors barriered and budget exhausted.
    max_b = cfg["game"]["max_barriers"]
    state = make_state(
        cop_pos=(0, 0),
        thief_pos=(4, 4),
        barriers=[(0, 1), (1, 0)],
        barriers_used=max_b,
    )
    mask = action_mask(state, "cop", cfg, idx=0)
    assert mask.any()


def test_two_cop_idx_selects_position(cfg, make_state):
    # idx selects which cop's position drives the move legality.
    state = make_state(cop_pos=[(0, 0), (2, 2)], thief_pos=(4, 4))
    m0 = action_mask(state, "cop", cfg, idx=0)
    m1 = action_mask(state, "cop", cfg, idx=1)
    # cop 0 at corner: UP illegal; cop 1 at center: UP legal.
    assert not m0[int(Action.UP)]
    assert m1[int(Action.UP)]


def test_enable_stay_adds_stay_index(cfg, make_state):
    # Deep-copy the real config so the toggle never leaks across tests.
    stay_cfg = copy.deepcopy(cfg)
    stay_cfg["env"]["actions"]["enable_stay"] = True
    state = make_state(cop_pos=(2, 2), thief_pos=(0, 0))
    mask = action_mask(state, "cop", stay_cfg, idx=0)
    # Length grows to a_cop + 1, and STAY is True for a free cop.
    assert mask.shape == (stay_cfg["env"]["actions"]["a_cop"] + 1,)
    assert mask.shape == (6,)
    assert mask[int(Action.STAY)]


def test_boxed_in_cop_with_budget_moves_all_true(cfg, make_state):
    # Cop boxed in on all four sides but WITH barrier budget remaining: the
    # MOVES-ONLY fallback must fire regardless of PLACE legality, so all four
    # move indices are True (not the degenerate [F,F,F,F,T] forced self-PLACE).
    surround = [(0, 1), (1, 0), (1, 2), (2, 1)]
    state = make_state(
        cop_pos=(1, 1),
        thief_pos=(4, 4),
        barriers=surround,
        barriers_used=0,  # budget remains -> PLACE would be legal
    )
    mask = action_mask(state, "cop", cfg, idx=0)
    assert mask[int(Action.UP)] and mask[int(Action.DOWN)]
    assert mask[int(Action.LEFT)] and mask[int(Action.RIGHT)]
    assert mask[: len(_MOVE_IDXS)].all()


def test_boxed_in_thief_moves_all_true_place_false(cfg, make_state):
    # A boxed-in THIEF gets the same MOVES-ONLY fallback; PLACE stays False.
    surround = [(0, 1), (1, 0), (1, 2), (2, 1)]
    state = make_state(cop_pos=(4, 4), thief_pos=(1, 1), barriers=surround)
    mask = action_mask(state, "thief", cfg, idx=0)
    assert mask[: len(_MOVE_IDXS)].all()
    assert not mask[int(Action.PLACE_BARRIER)]


def test_invalid_role_raises(cfg, make_state):
    state = make_state(cop_pos=(2, 2), thief_pos=(0, 0))
    with pytest.raises(ValueError, match="role"):
        action_mask(state, "robber", cfg, idx=0)


def test_cop_idx_out_of_range_raises(cfg, make_state):
    state = make_state(cop_pos=(2, 2), thief_pos=(0, 0))
    with pytest.raises(ValueError, match="idx"):
        action_mask(state, "cop", cfg, idx=1)


# Number of directional move indices (UP, DOWN, LEFT, RIGHT) at the front of the mask.
_MOVE_IDXS = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)
