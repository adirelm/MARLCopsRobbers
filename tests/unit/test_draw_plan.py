"""Tests for the pure board/HUD draw-plan logic (T7.4) — verified headless.

Asserts the plan draws a background, the cop + thief tokens, barriers, the capture
flash on a cop win, the optional view-radius overlay, and the HUD lines — i.e. the
rendering DECISIONS are correct without needing pygame.
"""

from __future__ import annotations

from src.gui import palette
from src.gui.draw_plan import build_board_plan, build_hud_plan
from src.gui.spectator import SpectatorFrame
from src.gui.transform import GridView


def _frame(**over) -> SpectatorFrame:
    base = {
        "grid": (5, 5),
        "cop_positions": ((1, 1),),
        "thief_position": (3, 3),
        "barriers": ((0, 0),),
        "view_radius": 2,
        "move": 4,
        "max_moves": 25,
        "sub_game": 2,
        "num_games": 6,
        "scores": {"cop": 0, "thief": 0},
        "totals": {"cop": 0, "thief": 0},
        "winner": None,
        "last_action": {"cop_0": "UP", "thief": "LEFT"},
    }
    return SpectatorFrame(**{**base, **over})


def _colors(plan) -> list:
    return [op["color"] for op in plan]


def test_board_plan_draws_background_tokens_and_barrier():
    """The plan starts with a background and draws the cop, thief, and a barrier."""
    plan = build_board_plan(_frame(), GridView(640, 480, 5, 5))
    assert plan[0]["kind"] == "background"
    cols = _colors(plan)
    assert palette.COP in cols
    assert palette.THIEF in cols
    assert palette.BARRIER in cols


def test_capture_flash_only_on_cop_win():
    """The capture-flash op appears only when the cop wins."""
    view = GridView(640, 480, 5, 5)
    assert palette.CAPTURE_FLASH not in _colors(build_board_plan(_frame(winner=None), view))
    assert palette.CAPTURE_FLASH in _colors(build_board_plan(_frame(winner="cop"), view))


def test_view_radius_overlay_is_opt_in():
    """The view-radius overlay is drawn only when requested (off by default)."""
    view = GridView(640, 480, 5, 5)
    assert palette.VIEW_RADIUS not in _colors(build_board_plan(_frame(), view, show_radius=False))
    assert palette.VIEW_RADIUS in _colors(build_board_plan(_frame(), view, show_radius=True))


def test_hud_plan_has_move_scores_and_winner():
    """The HUD renders the move, scores, last action, and a winner banner at terminal."""
    texts = [op["text"] for op in build_hud_plan(_frame(winner="cop"))]
    assert any("Move 4/25" in t for t in texts)
    assert any("Scores" in t for t in texts)
    assert any("Last" in t for t in texts)
    assert any("WINNER: COP" in t for t in texts)
    assert all(op["kind"] == "text" for op in build_hud_plan(_frame()))


def test_hud_plan_omits_last_and_winner_on_opening_frame():
    """The opening frame (no last_action, no winner) shows neither line."""
    texts = [op["text"] for op in build_hud_plan(_frame(last_action=None, winner=None))]
    assert not any("Last" in t for t in texts)
    assert not any("WINNER" in t for t in texts)
