"""HUD draw-plan builders — the PURE rendering logic (T7.4). No pygame.

Turn a :class:`SpectatorFrame` into the ordered HUD draw OPS (panel, sub-game / move /
scores / totals / barrier budget / last action / winner / help). The BOARD ops live in
:mod:`src.gui.draw_board` (split at the 150-LOC cap) and are re-exported here so callers
keep a single import. The thin pygame executor (``src/gui/render.py``) just runs the ops,
so WHICH cells/tokens/colours/text are drawn is testable headless.
"""

from __future__ import annotations

from src.gui import palette
from src.gui.draw_board import build_board_plan  # noqa: F401 — re-export (150-LOC split)
from src.gui.input_map import bindings
from src.gui.spectator import SpectatorFrame

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


# Which commands belong on which help row. Split because all eight on one line rendered
# 789px wide in a 720px window — it ran off the screen edge.
_PLAYBACK = ("toggle_pause", "speed_up", "slow_down", "next_sub_game", "reset")


def _help_lines(supported_commands=None) -> list[str]:
    """The persistent HUD help rows — DERIVED from the real key bindings (never drift).

    ``supported_commands`` (a set, or ``None`` for all) drops keys the current frame
    source cannot honestly execute — e.g. 'n next' when the client has no
    ``next_sub_game`` — so the HUD never advertises a command that silently no-ops.

    Two rows, playback then view: one row overflowed the window, and eight bindings run
    together were unreadable even when they fitted. Each key is bracketed because
    "r reset - speed-" gives no cue which token is the key and which is the action.
    """
    keys_by_cmd: dict[str, list[str]] = {}
    for key, cmd in bindings().items():
        if supported_commands is None or cmd in supported_commands:
            keys_by_cmd.setdefault(cmd, []).append(key)

    def row(label: str, cmds) -> str | None:
        parts = [
            f"[{'/'.join(sorted(keys_by_cmd[c]))}] {_CMD_SHORT[c]}" for c in sorted(cmds) if c in keys_by_cmd
        ]
        return f"{label}  " + " · ".join(parts) if parts else None

    rows = [row("Playback", _PLAYBACK), row("View", set(keys_by_cmd) - set(_PLAYBACK))]
    return [r for r in rows if r]


def hud_height(frame: SpectatorFrame) -> int:
    """Pixel height of the HUD strip for ``frame`` — the board's ``top_reserved`` value.

    Derived from the actual HUD plan so board reservation can never drift from what
    :func:`build_hud_plan` draws (the round-4 audit found text painted over tokens).
    Counts TEXT ops only: the panel/rule ops are backdrop for the same strip, and
    counting them would reserve phantom rows and shrink the board.
    """
    rows = sum(1 for op in build_hud_plan(frame) if op["kind"] == "text")
    return 8 + rows * (palette.FONT_PX + 4) + 2


def build_hud_plan(frame: SpectatorFrame, supported_commands=None, width: int | None = None) -> list[dict]:
    """Return the HUD ops (sub-game / move / scores / totals / barriers / last / winner / help).

    ``supported_commands`` filters the help line per frame-source capability (see
    :func:`_help_line`) WITHOUT changing the line count. The barrier and winner lines do
    change it — which is exactly why :func:`hud_height` measures this plan instead of
    hard-coding a row count.

    ``width`` (the window width) adds the panel backdrop + separator rule behind the text.
    It is optional so the HUD stays computable without a surface, which is what lets
    :func:`hud_height` call this before any window exists.
    """
    lines = [
        f"Sub-game {frame.sub_game}/{frame.num_games}",
        f"Move {frame.move}/{frame.max_moves}",
        f"This sub-game   cop {frame.scores['cop']}   thief {frame.scores['thief']}",
        f"Match totals   cop {frame.totals['cop']}   thief {frame.totals['thief']}",
    ]
    if frame.max_barriers:  # §3.3 budget — omitted when the frame source doesn't know it
        lines.append(f"Barriers used  {len(frame.barriers)}/{frame.max_barriers}")
    if frame.last_action:
        lines.append("Last move   " + "   ".join(f"{k} {v.lower()}" for k, v in frame.last_action.items()))
    if frame.winner:
        lines.append(f"WINNER: {frame.winner.upper()}")
    lines += _help_lines(supported_commands)  # persistent key-bindings help (§10.2 Nielsen 6 + 10)
    # Names the SHAPES, not just the colours: since the arcade sprites landed, "cop=blue"
    # described a hue but not the blue GHOST the viewer is actually looking at.
    lines.append("Legend  blue ghost = cop · red wedge = thief (mouth leads) · grey block = barrier")
    text_ops = [
        {"kind": "text", "pos": (8, 8 + i * (palette.FONT_PX + 4)), "text": t, "color": palette.TEXT}
        for i, t in enumerate(lines)
    ]
    if width is None:
        return text_ops
    strip = 8 + len(lines) * (palette.FONT_PX + 4) + 2
    panel = [
        {"kind": "fill", "rect": (0, 0, width, strip), "color": palette.HUD_PANEL},
        {"kind": "line", "start": (0, strip - 1), "end": (width, strip - 1), "color": palette.HUD_RULE},
    ]
    return panel + text_ops  # backdrop first — the text must land on top of it
