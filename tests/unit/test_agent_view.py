"""Agent view — the board rendering what the COPS KNOW, not what the referee knows.

These are the load-bearing §10.2 assertions: with the overlay on, the cops' Manhattan halo
is lit and a thief they cannot observe is dimmed; with it off the board is the plain god
view. Getting this backwards would make every agent-view screenshot claim the opposite of
the Dec-POMDP model it illustrates.

Split from test_draw_board.py at the 150-LOC cap; board mechanics stay there.
"""

from __future__ import annotations

from src.gui import palette
from src.gui.draw_board import build_board_plan
from tests.unit._board_frames import VIEW as _VIEW
from tests.unit._board_frames import board_frame as _frame


def _thief_token(plan) -> dict:
    """The thief's own sprite: the pac wedge (a poly), or the fallback disc if it has no heading.

    Trail dots are also THIEF-coloured circles, so the fallback picks the largest.
    """
    wedge = [op for op in plan if op["kind"] == "poly" and op["color"] == palette.THIEF]
    if wedge:
        return wedge[0]
    circles = [op for op in plan if op["kind"] == "circle" and op["color"] == palette.THIEF]
    return max(circles, key=lambda op: op["rect"][2])


def test_god_view_never_ghosts_the_thief() -> None:
    """With the overlay off the board is the referee's view — the thief is simply known."""
    assert _thief_token(build_board_plan(_frame(), _VIEW)).get("alpha") is None


def test_agent_view_ghosts_a_thief_outside_the_halo() -> None:
    """Distance 6 > radius 2: the cops cannot see it, so it must not be drawn as solid."""
    plan = build_board_plan(_frame(), _VIEW, show_radius=True)
    assert _thief_token(plan)["alpha"] == palette.GHOST_ALPHA


def test_agent_view_keeps_a_seen_thief_solid() -> None:
    """Inside the halo the thief IS observed — ghosting it would understate the cops."""
    frame = _frame(cop_positions=((2, 2),), thief_position=(2, 3))
    assert _thief_token(build_board_plan(frame, _VIEW, show_radius=True)).get("alpha") is None


def test_the_whole_unseen_thief_sprite_fades_not_just_part_of_it() -> None:
    """Ghosting must apply to the sprite the viewer actually sees — the wedge itself."""
    plan = build_board_plan(_frame(), _VIEW, show_radius=True)
    thief_ops = [op for op in plan if op["color"] == palette.THIEF and op["kind"] == "poly"]
    assert thief_ops and all(op["alpha"] == palette.GHOST_ALPHA for op in thief_ops)


def test_halo_is_drawn_only_in_agent_view() -> None:
    """The knowledge disk is the agent-view affordance; god view has no epistemic limit."""
    halo = [op for op in build_board_plan(_frame(), _VIEW) if op.get("color") == palette.VIEW_RADIUS]
    assert halo == []
    assert [op for op in build_board_plan(_frame(), _VIEW, True) if op.get("color") == palette.VIEW_RADIUS]


def test_an_unseen_thief_with_no_heading_is_still_ghosted() -> None:
    """The spawn-tick fallback disc must carry the ghosting too.

    Without this the alpha could be dropped from the no-heading branch and every test
    stayed green, because the other agent-view tests all supply a heading.
    """
    plan = build_board_plan(_frame(last_action=None), _VIEW, show_radius=True)
    discs = [op for op in plan if op["kind"] == "circle" and op["color"] == palette.THIEF]
    assert discs and all(op.get("alpha") == palette.GHOST_ALPHA for op in discs)


def test_a_seen_thief_with_no_heading_is_not_ghosted() -> None:
    """The complement, so the assertion above cannot pass by ghosting unconditionally."""
    frame = _frame(cop_positions=((2, 2),), thief_position=(2, 3), last_action=None)
    plan = build_board_plan(frame, _VIEW, show_radius=True)
    discs = [op for op in plan if op["kind"] == "circle" and op["color"] == palette.THIEF]
    assert discs and all(op.get("alpha") is None for op in discs)
