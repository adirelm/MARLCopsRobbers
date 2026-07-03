"""Tests for the pure board/HUD draw-plan logic (T7.4) — verified headless.

Asserts the plan draws a background, the cop + thief tokens, barriers, the capture
flash on a cop win, the optional view-radius overlay, and the HUD lines — i.e. the
rendering DECISIONS are correct without needing pygame.
"""

from __future__ import annotations

from src.gui import palette
from src.gui.draw_plan import build_board_plan, build_hud_plan, hud_height
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


def test_view_radius_overlay_marks_the_manhattan_disk():
    """The overlay covers EVERY in-grid cell within Manhattan radius of the cop — not one cell.

    Cop at (1,1) on 5x5 with radius 2: |r-1|+|c-1| <= 2 has exactly 11 in-grid cells.
    """
    plan = build_board_plan(_frame(), GridView(640, 480, 5, 5), show_radius=True)
    marked = [op for op in plan if op.get("color") == palette.VIEW_RADIUS]
    assert len(marked) == 11


def test_hud_plan_has_move_scores_and_winner():
    """The HUD renders move, scores, TOTALS, last action, a winner banner, and the help line."""
    texts = [op["text"] for op in build_hud_plan(_frame(winner="cop"))]
    assert any("Move 4/25" in t for t in texts)
    assert any("Scores" in t for t in texts)
    assert any("Totals" in t for t in texts)
    assert any("Last" in t for t in texts)
    assert any("WINNER: COP" in t for t in texts)
    assert all(op["kind"] == "text" for op in build_hud_plan(_frame()))


def test_board_reserves_the_hud_strip_at_the_shipped_window_size():
    """At the shipped 720x560 window NO board op renders under the HUD (round-4 regression guard).

    The HUD is 7-8 text lines; the board must letterbox strictly BELOW hud_height(frame),
    so text is never painted over cells or agent tokens (grid_4x4/5x5 screenshot defect).
    """
    frame = _frame(winner="cop")  # the tallest HUD (all optional lines present)
    strip = hud_height(frame)
    view = GridView(720, 560, 5, 5, top_reserved=strip)
    board_ops = [op for op in build_board_plan(frame, view, show_radius=True) if "rect" in op]
    assert min(op["rect"][1] for op in board_ops) >= strip


def test_hud_help_line_is_derived_from_the_real_bindings():
    """The persistent help + legend lines list every bound command (derived, so they can't drift)."""
    texts = [op["text"] for op in build_hud_plan(_frame())]
    help_line, legend = texts[-2], texts[-1]
    assert help_line.startswith("Keys")
    for fragment in ("space pause", "v radius", "escape quit"):
        assert fragment in help_line
    assert legend.startswith("Legend") and "cop=blue" in legend


def test_hud_plan_omits_last_and_winner_on_opening_frame():
    """The opening frame (no last_action, no winner) shows neither line."""
    texts = [op["text"] for op in build_hud_plan(_frame(last_action=None, winner=None))]
    assert not any("Last" in t for t in texts)
    assert not any("WINNER" in t for t in texts)
