"""Draw-plan builders — the PURE rendering logic (T7.4). No pygame.

Turn a :class:`SpectatorFrame` (+ :class:`GridView`) into an ordered list of draw
OPS — board (back-to-front: background, checkerboard, barriers, optional
view-radius overlay, thief, cop, capture flash) and HUD (sub-game / move / scores /
last action / winner). The thin pygame executor (``src/gui/render.py``) just runs
the ops, so WHICH cells/tokens/colours/text are drawn is testable headless.
"""

from __future__ import annotations

from src.gui import palette
from src.gui.spectator import SpectatorFrame
from src.gui.transform import GridView


def _token(rect: tuple, color: tuple) -> dict:
    """Return an inset filled-circle token op centred in ``rect``."""
    x, y, w, h = rect
    inset = palette.TOKEN_INSET
    return {"kind": "circle", "rect": (x + inset, y + inset, w - 2 * inset, h - 2 * inset), "color": color}


def build_board_plan(frame: SpectatorFrame, view: GridView, show_radius: bool = False) -> list[dict]:
    """Return the ordered board draw ops for one frame (back-to-front)."""
    rows, cols = frame.grid
    ops: list[dict] = [{"kind": "background", "color": palette.BG}]
    ops += [
        {"kind": "fill", "rect": view.cell_rect(c, r), "color": palette.CHECKER}
        for r in range(rows)
        for c in range(cols)
        if (r + c) % 2
    ]
    ops += [
        {"kind": "fill", "rect": view.cell_rect(bc, br), "color": palette.BARRIER}
        for br, bc in frame.barriers
    ]
    if show_radius:
        cr, cc = frame.cop_positions[0]
        ops.append({"kind": "rect", "rect": view.cell_rect(cc, cr), "color": palette.VIEW_RADIUS})
    tr, tc = frame.thief_position
    ops.append(_token(view.cell_rect(tc, tr), palette.THIEF))
    ops += [_token(view.cell_rect(cc, cr), palette.COP) for cr, cc in frame.cop_positions]
    if frame.winner == "cop":
        cr, cc = frame.cop_positions[0]
        ops.append({"kind": "rect", "rect": view.cell_rect(cc, cr), "color": palette.CAPTURE_FLASH})
    return ops


def build_hud_plan(frame: SpectatorFrame) -> list[dict]:
    """Return the HUD text ops (sub-game / move / scores / last action / winner)."""
    lines = [
        f"Sub-game {frame.sub_game}/{frame.num_games}",
        f"Move {frame.move}/{frame.max_moves}",
        f"Scores  cop {frame.scores['cop']}  thief {frame.scores['thief']}",
    ]
    if frame.last_action:
        lines.append("Last  " + "  ".join(f"{k}:{v}" for k, v in frame.last_action.items()))
    if frame.winner:
        lines.append(f"WINNER: {frame.winner.upper()}")
    return [
        {"kind": "text", "pos": (8, 8 + i * (palette.FONT_PX + 4)), "text": t, "color": palette.TEXT}
        for i, t in enumerate(lines)
    ]
