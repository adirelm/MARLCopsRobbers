"""Unit tests for src.marl.data.heuristics (T3.2 train-time oracles).

Pins the cop BFS expert (shortest, barrier-respecting, deterministic lowest-index
tie-break) and the thief greedy distance-maximizer, plus epsilon determinism.
"""

from __future__ import annotations

from random import Random

from src.marl.data.heuristics import cop_expert, thief_expert
from src.marl.env.actions import DELTAS, Action
from src.marl.env.grid import manhattan


def _apply(pos, action):
    """Return the cell reached by ``action`` from ``pos`` (no legality check)."""
    drow, dcol = DELTAS[Action(action)]
    return (pos[0] + drow, pos[1] + dcol)


def test_cop_step_is_on_a_shortest_path(make_state, cfg):
    """On a 3x3 the cop's first step strictly reduces Manhattan to the thief."""
    state = make_state(cop_pos=(0, 0), thief_pos=(2, 2), h=3, w=3)
    action = cop_expert(state, cfg, idx=0)
    nxt = _apply((0, 0), action)
    assert manhattan(nxt, (2, 2)) == manhattan((0, 0), (2, 2)) - 1


def test_cop_is_deterministic_lowest_index_tie_break(make_state, cfg):
    """Two equal-cost first moves resolve to the lowest Action enum index (UP=0)."""
    state = make_state(cop_pos=(2, 2), thief_pos=(0, 0), h=3, w=3)
    # UP and LEFT both shrink the distance equally; UP (0) must win the tie.
    assert cop_expert(state, cfg, idx=0) == Action.UP


def test_cop_routes_around_a_barrier(make_state, cfg):
    """A barrier on the greedy cell forces the BFS detour (still shortest, legal)."""
    # Direct down-step (1,0) is blocked; the cop must step RIGHT toward the thief.
    state = make_state(cop_pos=(0, 0), thief_pos=(2, 0), barriers=[(1, 0)], h=3, w=3)
    action = cop_expert(state, cfg, idx=0)
    nxt = _apply((0, 0), action)
    assert nxt != (1, 0)  # never steps onto the barrier
    assert nxt in {(0, 1)}  # the only legal first step on a shortest detour


def test_cop_chases_stationary_thief_optimally_on_2x2(make_state, cfg):
    """Against a STATIONARY thief the 2x2 chase IS min-Manhattan optimal.

    The thief is reconstructed at the SAME fixed cell every iteration (it never
    moves), so this pins the cop's pursuit of a non-moving target — which is
    exactly min-Manhattan. It does NOT claim the cop captures the *fleeing*
    thief_expert: a greedy equal-speed pursuer need not catch an optimal evader
    (see the NOT-FIXED note in planning/P3_FIXES.md), so that is not asserted.
    """
    cop = (0, 0)
    steps = 0
    while cop != (1, 1) and steps < 4:
        s = make_state(cop_pos=cop, thief_pos=(1, 1), h=2, w=2)
        cop = _apply(cop, cop_expert(s, cfg, idx=0))
        steps += 1
    assert cop == (1, 1)
    assert steps == manhattan((0, 0), (1, 1))


def test_cop_walled_off_returns_legal_fallback(make_state, cfg):
    """A cop fully walled off from the thief returns a legal fallback, no crash.

    Barriers seal a 2x2 pocket around the cop so the thief is unreachable: the
    BFS returns None (heuristics.py:97) and cop_expert must fall back to the
    lowest-index legal move (heuristics.py:128), still on-board (can_enter).
    """
    # Cop boxed into the top-left 2x2 pocket; thief outside, no path through.
    barriers = [(0, 2), (1, 2), (2, 0), (2, 1)]
    state = make_state(cop_pos=(0, 0), thief_pos=(4, 4), barriers=barriers, h=5, w=5)
    action = cop_expert(state, cfg, idx=0)
    nxt = _apply((0, 0), action)
    assert nxt in {(0, 1), (1, 0)}  # the only legal first steps inside the pocket


def test_cop_already_on_thief_returns_legal_move(make_state, cfg):
    """When the cop already shares the thief cell the BFS is a no-op (line 81).

    start == goal makes _bfs_first_step return None; cop_expert must still emit a
    legal on-board move rather than raise.
    """
    state = make_state(cop_pos=(0, 0), thief_pos=(0, 0), h=3, w=3)
    nxt = _apply((0, 0), cop_expert(state, cfg, idx=0))
    assert 0 <= nxt[0] < 3 and 0 <= nxt[1] < 3


def test_cop_fully_boxed_in_with_epsilon_does_not_crash(make_state, cfg):
    """A fully-walled cop with epsilon>0 + rng returns a safe action, no crash.

    Every directional neighbour is sealed by a barrier, so _legal_moves is empty.
    The epsilon-explore path must NOT call rng.choice([]) (which raises); it must
    return a safe no-op fallback action (B1).
    """
    # Seal all four neighbours of the corner cop on a 5x5 board.
    barriers = [(0, 1), (1, 0)]
    state = make_state(cop_pos=(0, 0), thief_pos=(4, 4), barriers=barriers, h=1, w=1)
    action = cop_expert(state, cfg, idx=0, rng=Random(0), epsilon=1.0)
    assert isinstance(action, Action)


def test_thief_fully_boxed_in_with_epsilon_does_not_crash(make_state, cfg):
    """A fully-walled thief with epsilon>0 + rng returns a safe action, no crash."""
    state = make_state(cop_pos=(4, 4), thief_pos=(0, 0), h=1, w=1)
    action = thief_expert(state, cfg, rng=Random(0), epsilon=1.0)
    assert isinstance(action, Action)


def test_cop_epsilon_explore_returns_a_legal_move(make_state, cfg):
    """epsilon=1 with a non-empty legal set takes the rng.choice explore path.

    Forces the successful exploration branch (a random LEGAL move) and asserts
    the result is one of the cop's legal directional moves, not the greedy step.
    """
    state = make_state(cop_pos=(2, 2), thief_pos=(0, 0), h=5, w=5)
    action = cop_expert(state, cfg, idx=0, rng=Random(3), epsilon=1.0)
    nxt = _apply((2, 2), action)
    assert 0 <= nxt[0] < 5 and 0 <= nxt[1] < 5


def test_thief_epsilon_explore_returns_a_legal_move(make_state, cfg):
    """epsilon=1 with a non-empty legal set takes the thief explore path."""
    state = make_state(cop_pos=(0, 0), thief_pos=(2, 2), h=5, w=5)
    action = thief_expert(state, cfg, rng=Random(3), epsilon=1.0)
    nxt = _apply((2, 2), action)
    assert 0 <= nxt[0] < 5 and 0 <= nxt[1] < 5


def test_cop_epsilon_zero_is_deterministic(make_state, cfg):
    """epsilon=0 ignores rng: repeated calls give an identical action."""
    state = make_state(cop_pos=(0, 0), thief_pos=(2, 2), h=3, w=3)
    a1 = cop_expert(state, cfg, idx=0, rng=Random(1), epsilon=0.0)
    a2 = cop_expert(state, cfg, idx=0, rng=Random(999), epsilon=0.0)
    assert a1 == a2


def test_cop_returns_legal_action(make_state, cfg):
    """The chosen cop action keeps the cop on-board (legal under can_enter)."""
    state = make_state(cop_pos=(0, 0), thief_pos=(2, 2), h=3, w=3)
    nxt = _apply((0, 0), cop_expert(state, cfg, idx=0))
    assert 0 <= nxt[0] < 3 and 0 <= nxt[1] < 3


def test_thief_increases_distance_to_nearest_cop(make_state, cfg):
    """The thief's greedy step does not decrease distance to the nearest cop."""
    state = make_state(cop_pos=(0, 0), thief_pos=(1, 1), h=3, w=3)
    before = manhattan((1, 1), (0, 0))
    nxt = _apply((1, 1), thief_expert(state, cfg))
    assert manhattan(nxt, (0, 0)) >= before


def test_thief_flees_the_nearest_of_two_cops(make_state, cfg):
    """With two cops the thief maximizes distance to the CLOSEST one."""
    state = make_state(cop_pos=[(0, 0), (4, 4)], thief_pos=(1, 1), h=5, w=5)
    nxt = _apply((1, 1), thief_expert(state, cfg))
    assert manhattan(nxt, (0, 0)) >= manhattan((1, 1), (0, 0))


def test_thief_epsilon_zero_is_deterministic(make_state, cfg):
    """epsilon=0 ignores rng for the thief too."""
    state = make_state(cop_pos=(0, 0), thief_pos=(2, 2), h=3, w=3)
    a1 = thief_expert(state, cfg, rng=Random(1), epsilon=0.0)
    a2 = thief_expert(state, cfg, rng=Random(7), epsilon=0.0)
    assert a1 == a2


def test_thief_lowest_index_tie_break(make_state, cfg):
    """Equal-distance flee moves resolve to the lowest Action index."""
    # Centered thief with a single cop equidistant on two axes; UP must win.
    state = make_state(cop_pos=(2, 2), thief_pos=(0, 0), h=3, w=3)
    # Moving DOWN or RIGHT both increase distance from (2,2) equally for (0,0)?
    # From (0,0): DOWN->(1,0) dist 3, RIGHT->(0,1) dist 3 — tie; DOWN(1)<RIGHT(3).
    assert thief_expert(state, cfg) == Action.DOWN
