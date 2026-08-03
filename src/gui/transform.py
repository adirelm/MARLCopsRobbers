"""GridView geometry — capped square cells + letterbox centering (T7.3).

PURE math (no pygame): maps a ``(cols, rows)`` board to centered square cell rects
inside a window. Cells are square and capped at ``CELL_PX_CAP``; the board is
letterboxed (centered) in the leftover space, so 2x2..5x5 all render crisply with
one view. The renderer (grid_view, pygame) consumes these plain ``(x, y, w, h)``
rects — keeping the geometry testable headless.
"""

from __future__ import annotations

from src.gui.palette import BOARD_MARGIN_PX, CELL_PX_CAP


class GridView:
    """Maps board cells to centered square pixel rects within a window."""

    def __init__(self, window_w: int, window_h: int, cols: int, rows: int, top_reserved: int = 0) -> None:
        """Compute the square cell size + the letterbox origin for this window/board.

        ``top_reserved`` excludes a top strip (the HUD) from the letterbox area, so the
        board never renders under the HUD text (it centers in the remaining space). A
        ``BOARD_MARGIN_PX`` strip is excluded at the BOTTOM for the same reason in reverse —
        otherwise the board letterboxes into every remaining pixel and its bottom edge sits
        flush against the window frame.
        """
        self._cols = int(cols)
        self._rows = int(rows)
        top = max(0, int(top_reserved))
        # Keep the margin only while it still leaves a usable strip — on a very short window
        # a visible board beats a pretty gap.
        margin = BOARD_MARGIN_PX if window_h - top > 2 * BOARD_MARGIN_PX else 0
        board_h = max(1, window_h - top - margin)
        usable = min(window_w / max(1, self._cols), board_h / max(1, self._rows))
        self._cell = max(1, int(min(CELL_PX_CAP, usable)))
        self._x0 = (window_w - self._cell * self._cols) // 2
        self._y0 = top + (board_h - self._cell * self._rows) // 2

    @property
    def cell_px(self) -> int:
        """The (capped) square cell size in pixels."""
        return self._cell

    def cell_rect(self, col: int, row: int) -> tuple[int, int, int, int]:
        """Return the ``(x, y, w, h)`` pixel rect of on-board cell ``(col, row)``.

        Raises:
            ValueError: If ``(col, row)`` is off the board.
        """
        if not (0 <= col < self._cols and 0 <= row < self._rows):
            raise ValueError(f"cell ({col},{row}) out of {self._cols}x{self._rows}")
        return (self._x0 + col * self._cell, self._y0 + row * self._cell, self._cell, self._cell)
