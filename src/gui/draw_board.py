"""Board draw ops — the neon/radar board layer (PURE; no pygame).

Split out of :mod:`src.gui.draw_plan` when the board grew a knowledge halo, motion
trails, facing wedges and capture shockwaves (the 150-LOC cap). Ops are emitted
back-to-front: background, checkerboard, grid lines, halo, barriers, trails, tokens,
wedges, shockwave.

The one op with modelling meaning rather than decoration is the halo + ghosted thief:
together they render what the COP TEAM KNOWS, not just where the pieces are. See
:mod:`src.gui.effects`.
"""

from __future__ import annotations

from src.gui import palette
from src.gui.effects import halo_cells, thief_is_seen
from src.gui.spectator import SpectatorFrame
from src.gui.sprites import pursued_ops, pursuer_ops
from src.gui.transform import GridView


def _token(rect: tuple, color: tuple, alpha: int | None = None, scale: float = 1.0) -> dict:
    """Return an inset filled-circle token op centred in ``rect``.

    ``scale`` shrinks the circle about the cell centre. Trail dots use it: at full token
    size a four-cell tail reads as four agents on the board rather than as one agent's path.
    """
    x, y, w, h = rect
    inset = palette.TOKEN_INSET
    dx, dy = (w - 2 * inset) * (1 - scale) / 2, (h - 2 * inset) * (1 - scale) / 2
    box = (x + inset + dx, y + inset + dy, (w - 2 * inset) * scale, (h - 2 * inset) * scale)
    op = {"kind": "circle", "rect": tuple(round(v) for v in box), "color": color}
    if alpha is not None:
        op["alpha"] = alpha
    return op


def _sprite_rect(rect: tuple) -> tuple:
    """Shrink a cell rect by the token inset so a sprite never touches the grid lines."""
    x, y, w, h = rect
    inset = palette.TOKEN_INSET
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


def _trail_ops(
    view: GridView, trails: dict, rows: int, cols: int, thief_alpha: int | None = None
) -> list[dict]:
    """Return fading tail ops for every agent, oldest cell faintest.

    ``thief_alpha`` dims the THIEF's tail in agent view when the cops cannot see it: they
    did not watch it walk that path either, so a full-strength route under a ghosted
    sprite would leak knowledge the sprite itself is careful not to claim. The cops' own
    tails are never dimmed — they always know where they have been.

    Cells outside the board are skipped rather than raising: a trail is decoration, and a
    stale entry must never crash the window.
    """
    ops: list[dict] = []
    for agent, cells in sorted(trails.items()):
        is_thief = agent == "thief"
        color = palette.THIEF if is_thief else palette.COP
        dim = (thief_alpha / 255) if (is_thief and thief_alpha is not None) else 1.0
        for index, (r, c) in enumerate(cells):
            if not (0 <= r < rows and 0 <= c < cols):
                continue
            fade = (index + 1) / (len(cells) + 1)  # oldest faintest+smallest, newest strongest
            ops.append(
                _token(
                    view.cell_rect(c, r),
                    color,
                    alpha=max(1, int(palette.TRAIL_ALPHA * fade * dim)),
                    scale=palette.TRAIL_SCALE * fade,
                )
            )
    return ops


def _shockwave(view: GridView, frame: SpectatorFrame) -> list[dict]:
    """Return concentric capture rings on the CAPTURING cop (nearest the thief)."""
    tr, tc = frame.thief_position
    cr, cc = min(frame.cop_positions, key=lambda pos: abs(pos[0] - tr) + abs(pos[1] - tc))
    x, y, w, h = view.cell_rect(cc, cr)
    ops = []
    for ring in range(palette.SHOCKWAVE_RINGS):
        grow = int(w * 0.18 * ring)
        ops.append(
            {
                "kind": "ring",
                "rect": (x - grow, y - grow, w + 2 * grow, h + 2 * grow),
                "color": palette.CAPTURE_FLASH,
                "alpha": max(20, palette.SHOCKWAVE_ALPHA - ring * 45),
            }
        )
    return ops


def build_board_plan(
    frame: SpectatorFrame, view: GridView, show_radius: bool = False, trails: dict | None = None
) -> list[dict]:
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
    ops += _trail_ops(view, trails or {}, rows, cols, thief_alpha=unseen)

    last = frame.last_action or {}
    tr, tc = frame.thief_position
    ops += pursued_ops(_sprite_rect(view.cell_rect(tc, tr)), last.get("thief"), unseen)
    for index, (cr, cc) in enumerate(frame.cop_positions):
        ops += pursuer_ops(_sprite_rect(view.cell_rect(cc, cr)), last.get(f"cop_{index}"))
    if frame.winner == "cop":
        ops += _shockwave(view, frame)
    return ops
