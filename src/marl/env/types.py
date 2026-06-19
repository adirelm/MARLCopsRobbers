"""Core environment value types for the CTDE train/exec split.

`GlobalState` is the TRAIN-TIME global state (centralized training, decentralized
execution) and is NEVER sent to an executor or across the MCP boundary.
`Observation` is the EXEC-TIME LOCAL observation and carries ONLY local fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np

Pos = tuple[int, int]  # (row, col), 0-indexed


@dataclass(frozen=True)
class GlobalState:
    """Immutable global game state held only by the trainer/referee (CTDE).

    Attributes:
        cop_pos: Cop positions; one entry (graded 5x5) or two (4x4 stage).
        thief_pos: The thief's absolute position.
        barriers: Every placed barrier cell (full-board knowledge, train-only).
        barriers_used: Count of barriers placed so far this sub-game.
        step: Current move index within the sub-game.
        h: Board rows (architect decision #1).
        w: Board columns.
        terminal: Whether this state is absorbing (capture/timeout/swap).
    """

    cop_pos: tuple[Pos, ...]
    thief_pos: Pos
    barriers: frozenset[Pos]
    barriers_used: int
    step: int
    h: int
    w: int
    terminal: bool = False


class Observation(TypedDict):
    """Egocentric LOCAL observation sent to an executor (architect decision #3).

    Attributes:
        image: (obs_channels, Wv, Wv) float32 egocentric window; channels are
            self, other_visible, barrier, out_of_bounds, time_norm.
        scalars: (obs_scalars,) float32 aliasing-memory hooks: own_r_norm,
            own_c_norm, step_norm, barriers_left_norm, other_seen_now,
            steps_since_seen_norm.
    """

    image: np.ndarray
    scalars: np.ndarray
