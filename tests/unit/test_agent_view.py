"""Agent view — the board rendering what the COPS KNOW, not what the referee knows.

These are the load-bearing §10.2 assertions: with the overlay on, the cops' Manhattan halo
is lit and anything they cannot observe (the thief sprite AND the path it walked) is
dimmed; with it off the board is the plain god view. Getting this backwards would make
every agent-view screenshot claim the opposite of the Dec-POMDP model it illustrates.

Split from test_draw_board.py at the 150-LOC cap; board mechanics stay there.
"""

from __future__ import annotations

from src.gui import palette
from src.gui.draw_board import build_board_plan
from src.gui.spectator import SpectatorFrame
from src.gui.transform import GridView

_VIEW = GridView(640, 480, 5, 5)


def _frame(**over) -> SpectatorFrame:
    base = {
        "grid": (5, 5),
        "cop_positions": ((1, 1),),
        "thief_position": (4, 4),
        "barriers": (),
        "view_radius": 2,
        "move": 4,
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


def test_an_unseen_thief_s_trail_fades_with_it() -> None:
    """Agent view must be consistent: a bright path under a ghosted thief still leaks its route.

    The cops cannot see the thief, so they did not watch it walk that path either — the
    tail has to dim with the sprite, exactly as the sprite's own mouth does.
    """
    trails = {"thief": ((0, 0), (0, 1))}
    god = build_board_plan(_frame(), _VIEW, trails=trails)
    agent = build_board_plan(_frame(), _VIEW, show_radius=True, trails=trails)

    def tail(plan):
        return [op["alpha"] for op in plan if op["kind"] == "circle" and op["color"] == palette.THIEF]

    assert tail(agent) and all(a < b for a, b in zip(tail(agent), tail(god), strict=True))


def test_a_seen_thief_keeps_a_full_strength_trail() -> None:
    """Inside the halo the cops DO observe it, so dimming the tail would understate them."""
    frame = _frame(cop_positions=((2, 2),), thief_position=(2, 3))
    trails = {"thief": ((2, 1), (2, 2))}
    god = build_board_plan(frame, _VIEW, trails=trails)
    agent = build_board_plan(frame, _VIEW, show_radius=True, trails=trails)

    def tail(plan):
        return [op["alpha"] for op in plan if op["kind"] == "circle" and op["color"] == palette.THIEF]

    assert tail(agent) == tail(god)


def test_the_cop_trail_is_never_dimmed() -> None:
    """The cops always know their own path — dimming it would be nonsense."""
    trails = {"cop_0": ((0, 0), (0, 1)), "thief": ((4, 4),)}
    god = build_board_plan(_frame(), _VIEW, trails=trails)
    agent = build_board_plan(_frame(), _VIEW, show_radius=True, trails=trails)

    def tail(plan):
        return [op["alpha"] for op in plan if op["kind"] == "circle" and op["color"] == palette.COP]

    assert tail(agent) == tail(god)
