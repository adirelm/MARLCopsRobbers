"""Draw-plan builders — the PURE rendering logic (T7.4). No pygame.

Turn a :class:`SpectatorFrame` (+ :class:`GridView`) into an ordered list of draw
OPS — board (back-to-front: background, checkerboard, barriers, optional
view-radius overlay for EVERY cop, thief, cops, capture flash on the CAPTURING cop)
and HUD (sub-game / move / scores / last action / winner). The thin pygame executor
(``src/gui/render.py``) just runs the ops, so WHICH cells/tokens/colours/text are
drawn is testable headless.
"""

from __future__ import annotations

from src.gui import palette
from src.gui.input_map import bindings
from src.gui.spectator import SpectatorFrame
from src.gui.transform import GridView

# Short HUD labels for the help line (§10.2 Nielsen 6: recognition over recall).
_CMD_SHORT = {
    "toggle_pause": "pause",
    "speed_up": "speed+",
    "slow_down": "speed-",
    "next_sub_game": "next",
    "reset": "reset",
    "toggle_view_radius": "radius",
    "quit": "quit",
}


def _help_line(supported_commands=None) -> str:
    """The persistent HUD help/legend line — DERIVED from the real key bindings (never drifts).

    ``supported_commands`` (a set, or ``None`` for all) drops keys the current frame
    source cannot honestly execute — e.g. 'n next' when the client has no
    ``next_sub_game`` — so the HUD never advertises a command that silently no-ops.
    """
    keys_by_cmd: dict[str, list[str]] = {}
    for key, cmd in bindings().items():
        if supported_commands is None or cmd in supported_commands:
            keys_by_cmd.setdefault(cmd, []).append(key)
    parts = ["/".join(sorted(keys)) + " " + _CMD_SHORT[cmd] for cmd, keys in sorted(keys_by_cmd.items())]
    return "Keys  " + "  ".join(parts)


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
    x0, y0 = view.cell_rect(0, 0)[:2]
    board = (x0, y0, view.cell_px * cols, view.cell_px * rows)
    ops.append({"kind": "rect", "rect": board, "color": palette.CHECKER})  # board outline (2x2 legibility)
    ops += [
        {"kind": "fill", "rect": view.cell_rect(bc, br), "color": palette.BARRIER}
        for br, bc in frame.barriers
    ]
    if show_radius:  # EVERY cop's Manhattan view disk (radius = frame.view_radius), clipped to the grid
        ops += [
            {"kind": "rect", "rect": view.cell_rect(c, r), "color": palette.VIEW_RADIUS}
            for r in range(rows)
            for c in range(cols)
            if any(abs(r - cr) + abs(c - cc) <= frame.view_radius for cr, cc in frame.cop_positions)
        ]
    tr, tc = frame.thief_position
    ops.append(_token(view.cell_rect(tc, tr), palette.THIEF))
    ops += [_token(view.cell_rect(cc, cr), palette.COP) for cr, cc in frame.cop_positions]
    if frame.winner == "cop":
        # Flash the CAPTURING cop — the one nearest the thief (same-cell capture: distance
        # 0; swap capture: distance 1) — not blindly cop 0 (wave-2 finding G6).
        cr, cc = min(frame.cop_positions, key=lambda pos: abs(pos[0] - tr) + abs(pos[1] - tc))
        ops.append({"kind": "rect", "rect": view.cell_rect(cc, cr), "color": palette.CAPTURE_FLASH})
    return ops


def hud_height(frame: SpectatorFrame) -> int:
    """Pixel height of the HUD strip for ``frame`` — the board's ``top_reserved`` value.

    Derived from the actual HUD plan so board reservation can never drift from what
    :func:`build_hud_plan` draws (the round-4 audit found text painted over tokens).
    """
    return 8 + len(build_hud_plan(frame)) * (palette.FONT_PX + 4) + 2


def build_hud_plan(frame: SpectatorFrame, supported_commands=None) -> list[dict]:
    """Return the HUD text ops (sub-game / move / scores / totals / barriers / last / winner / help).

    ``supported_commands`` filters the help line per frame-source capability (see
    :func:`_help_line`) WITHOUT changing the line count. The barrier and winner lines do
    change it — which is exactly why :func:`hud_height` measures this plan instead of
    hard-coding a row count.
    """
    lines = [
        f"Sub-game {frame.sub_game}/{frame.num_games}",
        f"Move {frame.move}/{frame.max_moves}",
        f"Scores  cop {frame.scores['cop']}  thief {frame.scores['thief']}",
        f"Totals  cop {frame.totals['cop']}  thief {frame.totals['thief']}",
    ]
    if frame.max_barriers:  # §3.3 budget — omitted when the frame source doesn't know it
        lines.append(f"Barriers  {len(frame.barriers)}/{frame.max_barriers}")
    if frame.last_action:
        lines.append("Last  " + "  ".join(f"{k}:{v}" for k, v in frame.last_action.items()))
    if frame.winner:
        lines.append(f"WINNER: {frame.winner.upper()}")
    lines.append(_help_line(supported_commands))  # persistent key-bindings help (§10.2 Nielsen 6 + 10)
    lines.append("Legend  cop=blue  thief=red  barrier=grey")  # token legend (own line — fits 720px)
    return [
        {"kind": "text", "pos": (8, 8 + i * (palette.FONT_PX + 4)), "text": t, "color": palette.TEXT}
        for i, t in enumerate(lines)
    ]
