"""Radar/neon effect layer — PURE geometry + epistemic state (no pygame).

The GUI's job is not only to show where the agents ARE (the referee's ground truth) but
what the cop team KNOWS: its Manhattan knowledge halo, and whether the thief currently
sits inside it. That distinction is the Dec-POMDP modelling claim (§2.1/§4), so it is
computed here as testable logic and consumed by :mod:`src.gui.draw_board`.

Everything is derived from a single :class:`~src.gui.spectator.SpectatorFrame` except the
motion trail, which needs history — :class:`TrailTracker` accumulates that GUI-side so no
frame source has to change (and so a trail can never contradict the frames it came from).
"""

from __future__ import annotations

from collections import deque

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


# Unit direction per movement action; PLACE_BARRIER is absent on purpose (not a direction).
_DIRECTIONS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}


def facing_wedge(rect: tuple[int, int, int, int], action: str | None) -> tuple | None:
    """Return triangle points marking the direction ``action`` moved, or ``None``.

    ``None`` for a missing or non-directional action: an arrow on a PLACE_BARRIER tick
    would assert a movement that did not happen.
    """
    step = _DIRECTIONS.get(action or "")
    if step is None:
        return None
    dx, dy = step
    x, y, w, h = rect
    cx, cy = x + w / 2, y + h / 2
    reach, half = w * 0.46, w * 0.22
    apex = (round(cx + dx * reach), round(cy + dy * reach))
    # The base is perpendicular to the direction of travel, inset from the token centre.
    base_x, base_y = cx + dx * half, cy + dy * half
    return (
        apex,
        (round(base_x - dy * half), round(base_y - dx * half)),
        (round(base_x + dy * half), round(base_y + dx * half)),
    )


class TrailTracker:
    """Accumulate each agent's recent cells across frames, for the fading motion tail.

    GUI-local by design: :class:`SpectatorFrame` is a frozen SNAPSHOT, and threading a
    history through every frame source (live session, wire replay, hand-built demos) would
    let a trail drift from the positions actually rendered. Feeding this the same frames
    the renderer draws makes that impossible.
    """

    def __init__(self, length: int) -> None:
        """Keep at most ``length`` recent cells per agent (including the current one)."""
        self._length = max(1, int(length))
        self._cells: dict[str, deque] = {}
        self._key: tuple[int, int] | None = None

    def observe(self, frame: SpectatorFrame) -> None:
        """Record this frame's positions, resetting first if the board restarted.

        Re-observing the SAME tick is a no-op, because the window redraws continuously:
        while paused the identical frame arrives every vsync, and appending it would
        flush the tail with copies of one cell.
        """
        key = (int(frame.sub_game), int(frame.move))
        if key == self._key:
            return  # same tick redrawn (pause / idle repaint)
        # A new sub-game, or a rewind to an earlier move (the 'r' reset key), means a fresh
        # board — carrying the previous tail across it would draw a path never walked.
        if self._key is not None and (key[0] != self._key[0] or key[1] < self._key[1]):
            self._cells.clear()
        self._key = key
        for index, pos in enumerate(frame.cop_positions):
            self._push(f"cop_{index}", tuple(pos))
        self._push("thief", tuple(frame.thief_position))

    def _push(self, agent: str, cell: tuple[int, int]) -> None:
        """Append ``cell`` to ``agent``'s bounded history."""
        self._cells.setdefault(agent, deque(maxlen=self._length)).append(cell)

    def trail(self, agent: str) -> tuple[tuple[int, int], ...]:
        """Return ``agent``'s PREVIOUS cells, oldest first, excluding where it stands now.

        The current cell is excluded because the token is already drawn there; a trail dot
        underneath it would just darken the token.
        """
        return tuple(self._cells.get(agent, ()))[:-1]
