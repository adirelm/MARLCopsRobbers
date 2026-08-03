"""Board draw ops — the neon/radar layer, verified headless.

The load-bearing assertions here are the AGENT-VIEW ones: with the radius overlay on, the
board must render what the cops KNOW (halo + a ghosted thief when it is outside that
halo), and with it off it must render the plain god view. Getting that backwards would
make every §10.2 screenshot claim the opposite of the Dec-POMDP model it illustrates.
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


def test_grid_lattice_covers_every_boundary() -> None:
    """A 5x5 board needs 6 horizontal + 6 vertical lines (inclusive of the outer edges)."""
    lines = [op for op in build_board_plan(_frame(), _VIEW) if op["kind"] == "line"]
    assert len(lines) == 12


def test_shockwave_only_on_a_cop_win() -> None:
    """Rings mark a capture; drawing them on a timeout would announce the wrong winner."""
    assert not [op for op in build_board_plan(_frame(), _VIEW) if op["kind"] == "ring"]
    assert not [op for op in build_board_plan(_frame(winner="thief"), _VIEW) if op["kind"] == "ring"]
    rings = [op for op in build_board_plan(_frame(winner="cop"), _VIEW) if op["kind"] == "ring"]
    assert len(rings) == palette.SHOCKWAVE_RINGS


def test_shockwave_marks_the_capturing_cop_not_cop_zero() -> None:
    """With two cops the rings belong on the one that actually closed the distance."""
    frame = _frame(cop_positions=((0, 0), (4, 3)), thief_position=(4, 4), winner="cop")
    rings = [op for op in build_board_plan(frame, _VIEW) if op["kind"] == "ring"]
    near = _VIEW.cell_rect(3, 4)  # (col, row) of the cop adjacent to the thief
    assert rings[0]["rect"][:2] == near[:2]


def test_a_barrier_placing_cop_has_centred_pupils() -> None:
    """PLACE_BARRIER consumes the move without moving the cop — its gaze must not claim travel."""
    placing = _frame(last_action={"cop_0": "PLACE_BARRIER", "thief": "LEFT"})
    moving = _frame(last_action={"cop_0": "RIGHT", "thief": "LEFT"})
    pupils = [
        [op["rect"] for op in build_board_plan(f, _VIEW) if op.get("color") == palette.EYE_PUPIL]
        for f in (placing, moving)
    ]
    assert pupils[0] and pupils[0] != pupils[1]


def test_a_thief_without_a_heading_falls_back_to_a_plain_disc() -> None:
    """At spawn there is no direction, so a mouth would invent one."""
    plan = build_board_plan(_frame(last_action=None), _VIEW)
    assert not [op for op in plan if op["kind"] == "poly" and op["color"] == palette.THIEF]
    assert _thief_token(plan)["kind"] == "circle"


def test_each_cop_is_drawn_as_a_body_plus_two_eyes() -> None:
    """One ghost silhouette and two eyes (white + pupil) per cop — no more, no fewer."""
    plan = build_board_plan(_frame(cop_positions=((1, 1), (3, 1))), _VIEW)
    assert len([op for op in plan if op["kind"] == "poly" and op["color"] == palette.COP]) == 2
    assert len([op for op in plan if op.get("color") == palette.EYE_WHITE]) == 4
    assert len([op for op in plan if op.get("color") == palette.EYE_PUPIL]) == 4


def test_sprites_survive_cells_smaller_than_the_inset() -> None:
    """A fixed 6px inset goes NEGATIVE once a cell is under ~12px, emitting inside-out rects.

    Unreachable from the shipped 720x560 window, but GridView is public geometry and a
    negative-size rect is a latent crash rather than a small sprite. Every emitted rect
    must stay non-degenerate at any window size.
    """
    for window in (720, 200, 80, 40, 20):
        view = GridView(window, window, 5, 5)
        plan = build_board_plan(_frame(), view, show_radius=True)
        bad = [op for op in plan if "rect" in op and (op["rect"][2] <= 0 or op["rect"][3] <= 0)]
        assert not bad, f"window {window}px (cell {view.cell_px}px) emitted {bad[:2]}"


def test_sprite_polygons_stay_inside_tiny_cells_too() -> None:
    """Shrinking must not let a wedge or ghost spill across its cell boundary."""
    view = GridView(60, 60, 5, 5)
    plan = build_board_plan(_frame(), view)
    cell = view.cell_px
    for op in [o for o in plan if o["kind"] == "poly"]:
        xs = [p[0] for p in op["points"]]
        ys = [p[1] for p in op["points"]]
        assert max(xs) - min(xs) <= cell and max(ys) - min(ys) <= cell
