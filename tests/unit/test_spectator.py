"""Tests for the god-view SpectatorFrame + SpectatorSession (T7.1).

The frame is frozen; the session (via the SDK single entry) drives a watchable
sub-game emitting full-board god-view frames + HUD to a winner. Pure (no pygame).
"""

from __future__ import annotations

import dataclasses

import pytest

from src.gui.spectator import SpectatorFrame
from src.sdk.sdk import MarlSDK


def _frame(**over) -> SpectatorFrame:
    """Build a minimal SpectatorFrame (override fields as needed)."""
    base = {
        "grid": (5, 5),
        "cop_positions": ((0, 0),),
        "thief_position": (4, 4),
        "barriers": (),
        "view_radius": 2,
        "move": 0,
        "max_moves": 25,
        "sub_game": 1,
        "num_games": 6,
        "scores": {"cop": 0, "thief": 0},
        "totals": {"cop": 0, "thief": 0},
        "winner": None,
        "last_action": None,
    }
    return SpectatorFrame(**{**base, **over})


def test_spectator_frame_is_frozen():
    """SpectatorFrame is immutable (a snapshot the GUI cannot mutate)."""
    frame = _frame()
    with pytest.raises(dataclasses.FrozenInstanceError):
        frame.move = 1


def test_session_reset_returns_opening_frame(cfg):
    """reset() yields the opening frame (move 0, no winner, config view radius)."""
    session = MarlSDK(cfg).spectator_session(5, 5, num_cops=1, seed=7)
    frame = session.reset()
    assert frame.grid == (5, 5)
    assert frame.move == 0
    assert frame.winner is None
    assert frame.last_action is None
    assert frame.view_radius == cfg["env"]["view_radius_by_grid"][5]


def test_session_step_advances_move_and_records_joint_action(cfg):
    """step() advances the move and records the god-view joint action by name."""
    session = MarlSDK(cfg).spectator_session(5, 5, num_cops=1, seed=7)
    session.reset()
    frame = session.step()
    assert frame.move == 1
    assert set(frame.last_action) == {"cop_0", "thief"}


def test_session_runs_to_a_winner_with_scores(cfg):
    """Stepping to termination yields a winner + non-negative scores."""
    session = MarlSDK(cfg).spectator_session(5, 5, num_cops=1, seed=7)
    session.reset()
    frame = session.reset()
    for _ in range(cfg["game"]["max_moves"] + 1):
        frame = session.step()
        if frame.winner is not None:
            break
    assert frame.winner in ("cop", "thief")
    assert frame.scores[frame.winner] >= 0
    after = session.step()  # stepping a finished sub-game is idempotent
    assert after.winner == frame.winner
    assert after.move == frame.move


def test_frame_is_god_view_with_full_positions(cfg):
    """The spectator sees BOTH absolute positions (god-view), unlike a partial-obs agent."""
    frame = MarlSDK(cfg).spectator_session(5, 5, num_cops=1, seed=3).reset()
    assert len(frame.thief_position) == 2
    assert all(len(p) == 2 for p in frame.cop_positions)


def test_next_sub_game_advances_counter_and_accumulates_totals(cfg):
    """next_sub_game banks the finished score + advances the counter (the 'n' key path)."""
    session = MarlSDK(cfg).spectator_session(5, 5, num_cops=1, seed=7)
    session.reset()
    assert session.next_sub_game().sub_game == 1  # mid-game (no winner) -> no-op
    frame = None
    for _ in range(cfg["game"]["max_moves"] + 1):
        frame = session.step()
        if frame.winner is not None:
            break
    won_role, banked = frame.winner, frame.scores
    advanced = session.next_sub_game()
    assert advanced.sub_game == 2  # the counter advanced (was statically 1 before the fix)
    assert advanced.winner is None and advanced.move == 0  # a fresh next sub-game
    assert advanced.totals[won_role] == banked[won_role]  # finished score banked into running totals
