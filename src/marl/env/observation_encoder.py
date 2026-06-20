"""Egocentric image-channel encoder for the LOCAL observation (T2.2).

Builds the PINNED ``(obs_channels, W_v, W_v)`` float32 egocentric window centered
on the acting agent. Channel order is frozen by the P2 spec / Observation
TypedDict: ``0 self, 1 other_visible, 2 barrier, 3 out_of_bounds, 4 time_norm``.
The window ALWAYS spans the max footprint (``W_v = 2*view_radius_max + 1``) so one
encoder spans 2x2->5x5 boards. The visible region is the Manhattan disk of
``radius``: a cell is ``out_of_bounds`` if it is off-board OR on-board but beyond
the radius (fog), so a 2x2 radius-0 window is near-blind (only the self cell). A
barrier or any of the ``opponents`` is drawn ONLY at a visible on-board cell —
no absolute opponent position leaks beyond the local disk. PURE (no RNG).
"""

from __future__ import annotations

import numpy as np

from src.marl.env.grid import in_grid, manhattan
from src.marl.env.types import Pos

# Frozen channel indices (mirror the Observation TypedDict docstring).
_SELF, _OTHER, _BARRIER, _OOB, _TIME = 0, 1, 2, 3, 4


def opponent_in_view(center: Pos, opp: Pos, radius: int) -> bool:
    """Return whether ``opp`` is within the Manhattan ``radius`` of ``center``.

    The single source of "can this agent see that cell/opponent" — used by the
    image encoder AND the env-owned visibility-memory update (DRY: no duplicated
    ``manhattan(...) <= radius`` checks). Re-exported by :mod:`observation`.

    Args:
        center: The acting agent's absolute board cell.
        opp: An opponent (or candidate) absolute board cell.
        radius: The Manhattan view radius.

    Returns:
        True iff ``manhattan(center, opp) <= radius``.
    """
    return manhattan(center, opp) <= radius


def encode_image(  # noqa: PLR0913 — one arg per egocentric-window input is intentional
    center: Pos,
    opponents: list[Pos],
    barriers: frozenset[Pos],
    h: int,
    w: int,
    radius: int,
    radius_max: int,
    channels: int,
    time_norm: float,
) -> np.ndarray:
    """Return the egocentric ``(channels, W_v, W_v)`` float32 image window.

    The window is centered on ``center`` (the acting agent) at window index
    ``(radius_max, radius_max)``; each window cell ``(wr, wc)`` maps to board cell
    ``(center_r + wr - radius_max, center_c + wc - radius_max)``. A cell is visible
    only when it is on-board AND ``manhattan(center, cell) <= radius``; every
    non-visible cell (off-board OR beyond the radius) sets ``out_of_bounds = 1``
    and carries no barrier/opponent. On a visible cell, ``barrier = 1`` when a
    barrier sits there and ``other_visible = 1`` when any of ``opponents`` sits
    there. ``time_norm`` is broadcast across the whole window.

    Args:
        center: The acting agent's absolute board cell (window center).
        opponents: The agent's opponent cells (one thief for a cop; ALL cops for
            the thief). Each on-board opponent within ``radius`` is marked.
        barriers: All placed barrier cells (train-time full set; only the visible
            subset inside the disk is encoded).
        h: Board rows.
        w: Board columns.
        radius: The Manhattan view radius gating cell visibility.
        radius_max: The padding footprint radius (``env.view_radius_max``).
        channels: Number of image channels (``env.obs_channels``).
        time_norm: ``state.step / game.max_moves`` broadcast over the window.

    Returns:
        A float32 ``np.ndarray`` of shape ``(channels, W_v, W_v)`` with
        ``W_v = 2 * radius_max + 1``.
    """
    wv = 2 * radius_max + 1
    image = np.zeros((channels, wv, wv), dtype=np.float32)
    image[_SELF, radius_max, radius_max] = 1.0
    image[_TIME, :, :] = np.float32(time_norm)

    for wr in range(wv):
        for wc in range(wv):
            cell: Pos = (center[0] + wr - radius_max, center[1] + wc - radius_max)
            if not in_grid(cell, h, w) or not opponent_in_view(center, cell, radius):
                image[_OOB, wr, wc] = 1.0  # off-board OR beyond the view radius (fog)
                continue
            if cell in barriers:
                image[_BARRIER, wr, wc] = 1.0
            if any(cell == opp for opp in opponents):
                image[_OTHER, wr, wc] = 1.0
    return image
