"""The radar/neon effect layer — pure geometry + epistemic state, no pygame.

These helpers exist so the GUI can render what the COP KNOWS (its Manhattan knowledge
halo, and a ghosted thief when the thief is outside it) rather than only what the
referee knows. That distinction is the Dec-POMDP modelling claim (§2.1/§4), so it is
worth testing as logic, not eyeballing as decoration.

Heading and character geometry live in ``test_gui_sprites.py``.
"""

from __future__ import annotations

from src.gui.effects import TrailTracker, halo_cells, thief_is_seen
from src.gui.spectator import SpectatorFrame


def _frame(**over) -> SpectatorFrame:
    base = {
        "grid": (5, 5),
        "cop_positions": ((2, 2),),
        "thief_position": (0, 0),
        "barriers": (),
        "view_radius": 2,
        "move": 3,
        "max_moves": 25,
        "sub_game": 1,
        "num_games": 6,
        "scores": {"cop": 0, "thief": 0},
        "totals": {"cop": 0, "thief": 0},
        "winner": None,
        "last_action": {"cop_0": "UP", "thief": "LEFT"},
        "max_barriers": 5,
    }
    return SpectatorFrame(**{**base, **over})


def test_halo_is_the_manhattan_disk_not_a_square() -> None:
    """A radius-2 halo around (2,2) is the 13-cell diamond — a 5x5 square would be wrong."""
    cells = set(halo_cells(_frame()))
    assert (0, 2) in cells and (2, 0) in cells  # diamond tips
    assert (0, 0) not in cells and (4, 4) not in cells  # square corners excluded
    assert len(cells) == 13


def test_halo_is_clipped_to_the_board() -> None:
    """A cop in the corner cannot know cells that do not exist."""
    cells = halo_cells(_frame(cop_positions=((0, 0),)))
    assert all(0 <= r < 5 and 0 <= c < 5 for r, c in cells)
    assert (0, 0) in cells


def test_halo_unions_every_cop() -> None:
    """With two cops the halo is the UNION of their disks (team knowledge, per CTDE)."""
    one = set(halo_cells(_frame(cop_positions=((0, 0),))))
    two = set(halo_cells(_frame(cop_positions=((0, 0), (4, 4)))))
    assert one < two and (4, 4) in two


def test_thief_seen_only_inside_the_disk() -> None:
    """The thief at distance 2 is seen; at distance 3 it is not."""
    assert thief_is_seen(_frame(cop_positions=((2, 2),), thief_position=(2, 4)))
    assert not thief_is_seen(_frame(cop_positions=((2, 2),), thief_position=(2, 4), view_radius=1))


def test_thief_seen_by_any_cop() -> None:
    """One cop in range is enough — the team shares observations at execution boundaries."""
    frame = _frame(cop_positions=((4, 4), (0, 1)), thief_position=(0, 0))
    assert thief_is_seen(frame)


def test_trail_records_where_each_agent_came_from() -> None:
    """After three observed frames the cop trail holds its two PREVIOUS cells, oldest first."""
    tracker = TrailTracker(length=4)
    for move, pos in enumerate(((0, 0), (0, 1), (0, 2))):
        tracker.observe(_frame(move=move, cop_positions=(pos,)))
    assert tracker.trail("cop_0") == ((0, 0), (0, 1))


def test_trail_excludes_the_current_cell() -> None:
    """The token is drawn at the current cell; a trail dot under it would double-draw."""
    tracker = TrailTracker(length=4)
    tracker.observe(_frame(move=0, cop_positions=((1, 1),)))
    tracker.observe(_frame(move=1, cop_positions=((1, 2),)))
    assert (1, 2) not in tracker.trail("cop_0")


def test_trail_is_bounded() -> None:
    """The trail never grows without limit — it is a fading tail, not a full history."""
    tracker = TrailTracker(length=3)
    for col in range(10):
        tracker.observe(_frame(move=col, cop_positions=((0, col % 5),)))
    assert len(tracker.trail("cop_0")) <= 2


def test_trail_resets_between_sub_games() -> None:
    """A new sub-game is a new board — carrying the old tail across would be a lie."""
    tracker = TrailTracker(length=4)
    tracker.observe(_frame(sub_game=1, move=0, cop_positions=((0, 0),)))
    tracker.observe(_frame(sub_game=1, move=1, cop_positions=((0, 1),)))
    tracker.observe(_frame(sub_game=2, move=0, cop_positions=((3, 3),)))
    assert tracker.trail("cop_0") == ()


def test_trail_resets_when_a_sub_game_is_restarted() -> None:
    """The 'r' reset key rewinds to move 0 without changing sub_game — also a fresh board."""
    tracker = TrailTracker(length=4)
    tracker.observe(_frame(move=4, cop_positions=((0, 0),)))
    tracker.observe(_frame(move=5, cop_positions=((0, 1),)))
    tracker.observe(_frame(move=0, cop_positions=((2, 2),)))
    assert tracker.trail("cop_0") == ()


def test_trail_tracks_the_thief_too() -> None:
    """Both roles leave a tail — the chase is only legible if you can see both paths."""
    tracker = TrailTracker(length=4)
    tracker.observe(_frame(move=0, thief_position=(0, 0)))
    tracker.observe(_frame(move=1, thief_position=(0, 1)))
    assert tracker.trail("thief") == ((0, 0),)


def test_pausing_does_not_flush_the_trail() -> None:
    """The window repaints every vsync; a paused tick must not append itself repeatedly.

    Without this the tail fills with copies of the current cell and visually vanishes.
    """
    tracker = TrailTracker(length=4)
    tracker.observe(_frame(move=0, cop_positions=((0, 0),)))
    tracker.observe(_frame(move=1, cop_positions=((0, 1),)))
    for _ in range(30):  # 30 repaints while paused on move 1
        tracker.observe(_frame(move=1, cop_positions=((0, 1),)))
    assert tracker.trail("cop_0") == ((0, 0),)
