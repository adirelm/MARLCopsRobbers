"""The executor's alpha path must size its scratch layer from the op's REAL extent.

Why this exists: an alpha op is drawn onto a scratch RGBA surface sized by
``render._bounds`` and then blitted. Ops carrying ``rect`` were covered, but the
POLYGON ops — which is what the ghosted thief is — take their extent from ``points``.
A ``_bounds`` that fell back to a 1x1 box rendered the ghosted thief into a single
pixel at the origin: the agent view's headline feature silently vanished from the
board while every draw-PLAN test stayed green, because the plan was correct and only
the rasterisation was wrong.

These tests therefore assert on PIXELS, not on ops.
"""

from __future__ import annotations

import pygame
import pytest

from src.gui import palette
from src.gui.draw_board import build_board_plan
from src.gui.render import _bounds, execute_plan
from src.gui.spectator import SpectatorFrame
from src.gui.transform import GridView


def _frame(**over) -> SpectatorFrame:
    base = {
        "grid": (5, 5),
        "cop_positions": ((4, 4),),
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
        "last_action": {"cop_0": "UP", "thief": "UP"},
        "max_barriers": 5,
    }
    return SpectatorFrame(**{**base, **over})


def test_bounds_of_a_polygon_covers_all_its_points() -> None:
    """A poly op has no `rect`; its box must come from `points`, not a fallback."""
    op = {"kind": "poly", "points": ((10, 20), (40, 25), (30, 60)), "color": (1, 2, 3)}
    x, y, w, h = _bounds(op)
    assert (x, y) == (10, 20)
    assert x + w > 40 and y + h > 60


def test_bounds_of_a_line_covers_both_endpoints() -> None:
    """Same rule for line ops, which also carry no `rect`."""
    x, y, w, h = _bounds({"kind": "line", "start": (5, 7), "end": (55, 9), "color": (1, 2, 3)})
    assert (x, y) == (5, 7)
    assert x + w > 55 and y + h > 9


@pytest.mark.parametrize("show_radius", [True, False])
def test_the_thief_is_actually_painted_on_the_board(show_radius: bool) -> None:
    """REGRESSION: the ghosted thief must reach the SURFACE, not just the draw plan.

    Rendered at the thief's own cell, some pixel must carry thief-red — in god view at
    full strength, in agent view blended toward the background but still present.
    """
    surface = pygame.Surface((palette.WINDOW_W, palette.WINDOW_H))
    surface.fill(palette.BG)
    frame = _frame()
    view = GridView(palette.WINDOW_W, palette.WINDOW_H, 5, 5, top_reserved=0)
    execute_plan(surface, None, build_board_plan(frame, view, show_radius))

    x, y, w, h = view.cell_rect(0, 0)  # (col, row) of the thief
    reddest = max(
        (surface.get_at((px, py))[:3] for px in range(x, x + w) for py in range(y, y + h)),
        key=lambda c: c[0] - (c[1] + c[2]) / 2,
    )
    assert reddest[0] > palette.BG[0] + 40, f"thief not painted (show_radius={show_radius}): {reddest}"


def test_agent_view_thief_is_dimmer_than_god_view_thief() -> None:
    """The ghosting must be visible as pixels too — not merely present in the plan."""
    view = GridView(palette.WINDOW_W, palette.WINDOW_H, 5, 5, top_reserved=0)
    x, y, w, h = view.cell_rect(0, 0)

    def peak_red(show_radius: bool) -> int:
        surface = pygame.Surface((palette.WINDOW_W, palette.WINDOW_H))
        surface.fill(palette.BG)
        execute_plan(surface, None, build_board_plan(_frame(), view, show_radius))
        return max(surface.get_at((px, py))[0] for px in range(x, x + w) for py in range(y, y + h))

    assert peak_red(True) < peak_red(False)
