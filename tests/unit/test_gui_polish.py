"""The §10 HUD/board polish: a visible barrier budget and a board that isn't bottom-flush.

Two gaps this closes, both found by inspecting the committed §7.3 screenshots:
  * `max_barriers` (5) is a core §3.3 resource driving the cop's whole strategy, yet the
    HUD never showed it — a spectator had to count grey cells.
  * the board letterboxed into ALL remaining space, so its bottom edge sat flush against
    the window with zero breathing room.
"""

from __future__ import annotations

from src.gui import palette
from src.gui.draw_plan import build_hud_plan, hud_height
from src.gui.spectator import SpectatorFrame
from src.gui.transform import GridView


def _frame(**over) -> SpectatorFrame:
    base = {
        "grid": (5, 5),
        "cop_positions": ((1, 1),),
        "thief_position": (3, 3),
        "barriers": ((0, 0), (2, 2)),
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


def _texts(frame) -> list[str]:
    return [op["text"] for op in build_hud_plan(frame)]


def test_hud_shows_barriers_placed_over_budget() -> None:
    """Two barriers placed out of a budget of 5 must read 'Barriers  2/5'."""
    line = next(t for t in _texts(_frame(max_barriers=5)) if t.startswith("Barriers"))
    assert line == "Barriers  2/5"


def test_barrier_line_tracks_the_actual_placements() -> None:
    """The count is derived from the frame's barriers — it cannot drift from the board."""
    frame = _frame(barriers=((0, 0), (1, 1), (2, 2), (3, 3)), max_barriers=5)
    assert "Barriers  4/5" in _texts(frame)


def test_exhausted_budget_is_visible() -> None:
    """A cop with no barriers left must be able to see that from the HUD alone."""
    frame = _frame(barriers=tuple((0, i) for i in range(5)), max_barriers=5)
    assert "Barriers  5/5" in _texts(frame)


def test_hud_omits_the_barrier_line_when_the_budget_is_unknown() -> None:
    """Frame sources that do not carry a budget (replay/demo) must not render '0/0'."""
    assert not [t for t in _texts(_frame()) if t.startswith("Barriers")]


def test_hud_height_absorbs_the_extra_line() -> None:
    """hud_height is derived, so the taller HUD must reserve more board space, not overlap."""
    assert hud_height(_frame(max_barriers=5)) > hud_height(_frame())


def test_board_is_not_flush_with_the_window_bottom() -> None:
    """The board's bottom edge must clear the window by the configured margin."""
    rows = cols = 5
    view = GridView(720, 560, cols, rows, top_reserved=hud_height(_frame(max_barriers=5)))
    bottom = view.cell_rect(cols - 1, rows - 1)[1] + view.cell_px
    assert 560 - bottom >= palette.BOARD_MARGIN_PX


def test_board_still_clears_the_hud() -> None:
    """The margin must not be bought by sliding the board up under the HUD text."""
    top = hud_height(_frame(max_barriers=5))
    view = GridView(720, 560, 5, 5, top_reserved=top)
    assert view.cell_rect(0, 0)[1] >= top


def test_window_size_has_a_single_source() -> None:
    """720x560 was duplicated in run_app and capture_screens; palette is now the one source."""
    assert (palette.WINDOW_W, palette.WINDOW_H) == (720, 560)


def test_readme_barrier_caption_matches_the_configured_budget(cfg) -> None:
    """README claims the demo HUD reads 'Barriers 2/5' — the /5 must track config, not prose.

    Mutation-checked: dropping game.max_barriers to 4 without editing the caption fails here.
    """
    import re  # noqa: PLC0415 - only this guard needs it
    from pathlib import Path  # noqa: PLC0415

    readme = Path(__file__).resolve().parents[2] / "README.md"
    claims = re.findall(r"`Barriers (\d+)/(\d+)`", readme.read_text(encoding="utf-8"))
    assert claims, "README no longer states the barrier readout — caption/feature drift"
    budget = int(cfg["game"]["max_barriers"])
    for placed, cap in claims:
        assert int(cap) == budget, f"README says /{cap}, config says /{budget}"
        assert int(placed) <= budget, "caption claims more barriers placed than the budget allows"
