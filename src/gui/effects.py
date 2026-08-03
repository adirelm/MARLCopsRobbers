"""Radar layer — PURE epistemic state (no pygame).

The GUI's job is not only to show where the agents ARE (the referee's ground truth) but
what the cop team KNOWS: its Manhattan knowledge halo, and whether the thief currently
sits inside it. That distinction is the Dec-POMDP modelling claim (§2.1/§4), so it is
computed here as testable logic and consumed by :mod:`src.gui.draw_board`.

Heading and character geometry are NOT here — they belong to :mod:`src.gui.sprites`,
which owns the single heading table (``_HEADING_DEG``). This module deliberately keeps
no direction table of its own; an earlier arrow-marker helper did, and having two
representations of the same four headings is exactly the duplication to avoid.

Everything is derived from a single :class:`~src.gui.spectator.SpectatorFrame`, so nothing
here can drift from the frame being rendered.
"""

from __future__ import annotations

from src.gui.spectator import SpectatorFrame


def halo_cells(frame: SpectatorFrame) -> tuple[tuple[int, int], ...]:
    """Return the board cells inside ANY cop's Manhattan view radius, clipped to the grid.

    The UNION across cops is deliberate: under CTDE the cop team is one decision-maker at
    training time, so the halo shows TEAM knowledge, not one agent's.
    """
    rows, cols = frame.grid
    radius = int(frame.view_radius)
    return tuple(
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if any(abs(r - cr) + abs(c - cc) <= radius for cr, cc in frame.cop_positions)
    )


def thief_is_seen(frame: SpectatorFrame) -> bool:
    """True when the thief lies within some cop's view radius (i.e. the cops observe it)."""
    tr, tc = frame.thief_position
    radius = int(frame.view_radius)
    return any(abs(tr - cr) + abs(tc - cc) <= radius for cr, cc in frame.cop_positions)
