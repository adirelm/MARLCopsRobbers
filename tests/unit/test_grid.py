"""Unit tests for src/marl/env/grid.py (T1.3) — pure board geometry.

Covers in_grid corners/OOB, manhattan symmetry, can_enter (self-cell exception,
barrier block, OOB), resolve_move (edge==wall stay, barrier stay, free move),
and sample_spawn (dist>min_dist seeded + ValueError on infeasible).
"""

from __future__ import annotations

import random

import pytest

from src.marl.env.grid import (
    can_enter,
    in_grid,
    manhattan,
    resolve_move,
    sample_spawn,
)


def test_in_grid_corners_inside():
    for pos in [(0, 0), (0, 4), (4, 0), (4, 4), (2, 2)]:
        assert in_grid(pos, 5, 5) is True


def test_in_grid_oob_each_edge():
    for pos in [(-1, 0), (0, -1), (5, 0), (0, 5), (5, 5), (-1, -1)]:
        assert in_grid(pos, 5, 5) is False


def test_in_grid_non_square():
    assert in_grid((2, 1), 3, 2) is True
    assert in_grid((0, 2), 3, 2) is False
    assert in_grid((3, 0), 3, 2) is False


def test_manhattan_zero_and_value():
    assert manhattan((1, 1), (1, 1)) == 0
    assert manhattan((0, 0), (2, 2)) == 4
    assert manhattan((0, 0), (0, 3)) == 3


def test_manhattan_symmetry():
    a, b = (1, 4), (3, 0)
    assert manhattan(a, b) == manhattan(b, a)


def test_can_enter_free_cell():
    assert can_enter((1, 1), (0, 1), frozenset(), 5, 5) is True


def test_can_enter_oob_false():
    assert can_enter((-1, 0), (0, 0), frozenset(), 5, 5) is False
    assert can_enter((5, 5), (4, 4), frozenset(), 5, 5) is False


def test_can_enter_barrier_blocks():
    assert can_enter((1, 1), (0, 1), frozenset({(1, 1)}), 5, 5) is False


def test_can_enter_self_cell_exception_even_on_barrier():
    # target == actor is always enterable (STAY/no-op), even if a barrier sits there.
    assert can_enter((2, 2), (2, 2), frozenset({(2, 2)}), 5, 5) is True


def test_resolve_move_free():
    assert resolve_move((0, 0), (1, 0), (0, 0), frozenset(), 5, 5) == (1, 0)


def test_resolve_move_edge_is_wall_stays():
    # Moving UP from the top row hits the OOB wall -> stay put.
    assert resolve_move((0, 0), (-1, 0), (0, 0), frozenset(), 5, 5) == (0, 0)
    # Moving RIGHT off the last column -> stay put.
    assert resolve_move((2, 4), (0, 1), (2, 4), frozenset(), 5, 5) == (2, 4)


def test_resolve_move_barrier_stays():
    assert resolve_move((0, 0), (1, 0), (0, 0), frozenset({(1, 0)}), 5, 5) == (0, 0)


def test_resolve_move_zero_delta_stays_on_self():
    assert resolve_move((3, 3), (0, 0), (3, 3), frozenset(), 5, 5) == (3, 3)


def test_resolve_move_zero_delta_on_barriered_self_cell_stays():
    # delta=(0,0) with the actor standing ON a barriered cell: the self-cell
    # exception makes STAY legal, so the actor stays put rather than erroring.
    assert resolve_move((2, 2), (0, 0), (2, 2), frozenset({(2, 2)}), 5, 5) == (2, 2)


def test_sample_spawn_respects_min_dist_seeded():
    rng = random.Random(7)
    for _ in range(50):
        cop, thief = sample_spawn(rng, 5, 5, min_dist=2)
        assert in_grid(cop, 5, 5)
        assert in_grid(thief, 5, 5)
        assert manhattan(cop, thief) > 2


def test_sample_spawn_min_dist_zero_distinct_cells():
    rng = random.Random(1)
    cop, thief = sample_spawn(rng, 5, 5, min_dist=0)
    assert manhattan(cop, thief) > 0


def test_sample_spawn_infeasible_raises():
    rng = random.Random(0)
    # d_max on 5x5 is (5-1)+(5-1)=8; min_dist >= 8 is infeasible.
    with pytest.raises(ValueError):
        sample_spawn(rng, 5, 5, min_dist=8)
    with pytest.raises(ValueError):
        sample_spawn(rng, 5, 5, min_dist=9)


def test_sample_spawn_empty_board_raises():
    rng = random.Random(0)
    # Degenerate boards (h=0 or w=0) must raise ValueError, not IndexError:
    # the d_max-only guard let h=0,w=5 (d_max=3) slip past for small min_dist.
    with pytest.raises(ValueError):
        sample_spawn(rng, 0, 5, min_dist=2)
    with pytest.raises(ValueError):
        sample_spawn(rng, 5, 0, min_dist=2)


def test_sample_spawn_near_boundary_min_dist_seeded_terminates():
    rng = random.Random(7)
    # min_dist=7 on 5x5 (d_max=8) is feasible but tight: the only valid pairs
    # are the diagonally-opposite corners (manhattan==8). Looped to prove the
    # rejection sampler terminates and always hits the unique achievable dist.
    for _ in range(20):
        cop, thief = sample_spawn(rng, 5, 5, min_dist=7)
        assert manhattan(cop, thief) == 8
