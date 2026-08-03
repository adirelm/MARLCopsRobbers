"""Shared board-frame builder for the GUI draw tests.

Extracted when the agent-view assertions were split out of ``test_draw_board.py``: the
same ~20-line ``_frame`` factory and ``GridView`` had been copied byte-for-byte into the
new file, so a change to the default frame would have silently desynced the two suites.
"""

from __future__ import annotations

from src.gui.spectator import SpectatorFrame
from src.gui.transform import GridView

VIEW = GridView(640, 480, 5, 5)

_BASE = {
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


def board_frame(**over) -> SpectatorFrame:
    """A 5x5 god-view frame: one cop at (1,1), thief at (4,4) — Manhattan distance 6."""
    return SpectatorFrame(**{**_BASE, **over})
