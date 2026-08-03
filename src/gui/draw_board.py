"""Board draw ops — the neon/radar board layer (PURE; no pygame).

Split out of :mod:`src.gui.draw_plan` when the board grew a knowledge halo and arcade
character sprites (the 150-LOC cap). Ops are emitted
back-to-front: background, checkerboard, grid lines, halo, barriers, characters,
capture rings.

The one op with modelling meaning rather than decoration is the halo + ghosted thief:
together they render what the COP TEAM KNOWS, not just where the pieces are. See
:mod:`src.gui.effects`.
"""

from __future__ import annotations

from src.gui import palette
from src.gui.effects import halo_cells, thief_is_seen
from src.gui.spectator import SpectatorFrame
from src.gui.sprites import capture_rings, pursued_ops, pursuer_ops
from src.gui.transform import GridView


def _token(rect: tuple, color: tuple, alpha: int | None = None, scale: float = 1.0) -> dict:
    """Return an inset filled-circle token op centred in ``rect``.

    ``scale`` shrinks the circle about the cell centre. Trail dots use it: at full token
    size a four-cell tail reads as four agents on the board rather than as one agent's path.
    """
    x, y, w, h = rect
    inset = _inset_px(w, h)
    inner_w, inner_h = w - 2 * inset, h - 2 * inset
    dx, dy = inner_w * (1 - scale) / 2, inner_h * (1 - scale) / 2
    box = (x + inset + dx, y + inset + dy, max(1, inner_w * scale), max(1, inner_h * scale))
    op = {"kind": "circle", "rect": tuple(round(v) for v in box), "color": color}
    if alpha is not None:
        op["alpha"] = alpha
    return op


def _inset_px(w: int, h: int) -> int:
    """The gap to leave around a sprite, CLAMPED so it can never consume the cell.

    ``TOKEN_INSET`` is a fixed 6px, which goes negative once a cell drops under ~12px
    (reachable through GridView, which is public geometry). A tiny sprite is fine; an
    inside-out rect is not.
    """
    return max(0, min(palette.TOKEN_INSET, (min(w, h) - 2) // 2))


def _sprite_rect(rect: tuple) -> tuple:
    """Shrink a cell rect by the token inset so a sprite never touches the grid lines."""
    x, y, w, h = rect
    inset = _inset_px(w, h)
    return (x + inset, y + inset, w - 2 * inset, h - 2 * inset)


def _grid_lines(view: GridView, rows: int, cols: int) -> list[dict]:
    """Return the neon lattice ops — one line per row/column boundary, outer edges included."""
    x0, y0 = view.cell_rect(0, 0)[:2]
    width, height = view.cell_px * cols, view.cell_px * rows
    ops = [
        {
            "kind": "line",
            "start": (x0, y0 + r * view.cell_px),
            "end": (x0 + width, y0 + r * view.cell_px),
            "color": palette.GRID_LINE,
        }
        for r in range(rows + 1)
    ]
    ops += [
        {
            "kind": "line",
            "start": (x0 + c * view.cell_px, y0),
            "end": (x0 + c * view.cell_px, y0 + height),
            "color": palette.GRID_LINE,
        }
        for c in range(cols + 1)
    ]
    return ops


def build_board_plan(frame: SpectatorFrame, view: GridView, show_radius: bool = False) -> list[dict]:
    """Return the ordered board draw ops for one frame (back-to-front).

    ``show_radius`` switches on the AGENT-VIEW reading of the board: the cops' knowledge
    halo is drawn, and the thief is ghosted whenever it sits outside that halo — i.e. the
    board shows what the cops know. With it off the board is the plain god view.
    """
    rows, cols = frame.grid
    ops: list[dict] = [{"kind": "background", "color": palette.BG}]
    ops += [
        {"kind": "fill", "rect": view.cell_rect(c, r), "color": palette.CHECKER}
        for r in range(rows)
        for c in range(cols)
        if (r + c) % 2
    ]
    ops += _grid_lines(view, rows, cols)
    if show_radius:
        ops += [
            {
                "kind": "fill",
                "rect": view.cell_rect(c, r),
                "color": palette.VIEW_RADIUS,
                "alpha": palette.HALO_ALPHA,
            }
            for r, c in halo_cells(frame)
        ]
    ops += [
        {"kind": "fill", "rect": view.cell_rect(bc, br), "color": palette.BARRIER}
        for br, bc in frame.barriers
    ]
    # Ghost the thief only in agent view: in god view its position is simply known.
    unseen = palette.GHOST_ALPHA if (show_radius and not thief_is_seen(frame)) else None

    last = frame.last_action or {}
    tr, tc = frame.thief_position
    ops += pursued_ops(_sprite_rect(view.cell_rect(tc, tr)), last.get("thief"), unseen)
    for index, (cr, cc) in enumerate(frame.cop_positions):
        ops += pursuer_ops(_sprite_rect(view.cell_rect(cc, cr)), last.get(f"cop_{index}"))
    if frame.winner == "cop":
        ops += capture_rings(view, frame)
    return ops
