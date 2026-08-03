"""Pure board geometry for the CopsRobbersEnv (T1.3).

Stateless helpers over an H x W grid: bounds checks, Manhattan distance, cell
entry legality (with the self-cell / STAY exception and barrier blocking),
deterministic move resolution (edge == wall ⇒ stay), and seeded spawn sampling.
No config or RL state lives here — this is the geometry kernel imported by
actions.py, reward.py, and the env runtime.
"""

from __future__ import annotations

from random import Random

from src.marl.env.types import Pos


def in_grid(pos: Pos, h: int, w: int) -> bool:
    """Return whether ``pos`` lies inside the H x W board.

    Args:
        pos: A (row, col) cell, 0-indexed.
        h: Board rows.
        w: Board columns.

    Returns:
        True iff ``0 <= row < h`` and ``0 <= col < w``.
    """
    row, col = pos
    return 0 <= row < h and 0 <= col < w


def manhattan(a: Pos, b: Pos) -> int:
    """Return the L1 (Manhattan) distance between two cells.

    Args:
        a: First (row, col) cell.
        b: Second (row, col) cell.

    Returns:
        ``|a.row - b.row| + |a.col - b.col|`` (symmetric, non-negative).
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def can_enter(target: Pos, actor: Pos, barriers: frozenset[Pos], h: int, w: int) -> bool:
    """Return whether ``actor`` may occupy ``target`` next tick.

    A cell is enterable iff it is in-bounds and either is not a barrier or is the
    actor's own current cell (the self-cell / STAY exception — staying put is
    always legal even if a barrier sits on the actor's cell).

    Args:
        target: Candidate destination cell.
        actor: The actor's current cell.
        barriers: All placed barrier cells.
        h: Board rows.
        w: Board columns.

    Returns:
        True iff the move into ``target`` is geometrically legal.
    """
    if not in_grid(target, h, w):
        return False
    return target not in barriers or target == actor


def resolve_move(  # noqa: PLR0913 — signature pinned by P1 spec for integration
    pos: Pos, delta: Pos, actor: Pos, barriers: frozenset[Pos], h: int, w: int
) -> Pos:
    """Apply ``delta`` to ``pos``, falling back to ``pos`` on a blocked move.

    The grid edge behaves as a wall and barriers block entry, so an illegal move
    is a no-op (the actor stays put) rather than an error.

    Args:
        pos: The actor's current cell.
        delta: A (drow, dcol) step from the action's DELTA.
        actor: The actor's current cell (for the self-cell exception).
        barriers: All placed barrier cells.
        h: Board rows.
        w: Board columns.

    Returns:
        The destination cell if enterable, otherwise ``pos`` (stay).
    """
    target: Pos = (pos[0] + delta[0], pos[1] + delta[1])
    return target if can_enter(target, actor, barriers, h, w) else pos


def sample_spawn(rng: Random, h: int, w: int, min_dist: int) -> tuple[Pos, Pos]:
    """Sample distinct cop/thief start cells with Manhattan distance > min_dist.

    Args:
        rng: A seeded ``random.Random`` for reproducibility.
        h: Board rows.
        w: Board columns.
        min_dist: Spawns must satisfy ``manhattan(cop, thief) > min_dist``.

    Returns:
        A ``(cop, thief)`` pair of cells satisfying the distance constraint.

    Raises:
        ValueError: If the board is empty/degenerate (``h <= 0`` or ``w <= 0``)
            or ``min_dist >= (h - 1) + (w - 1)`` (the max achievable distance),
            making the constraint infeasible.
    """
    d_max = (h - 1) + (w - 1)
    if h <= 0 or w <= 0 or min_dist >= d_max:
        raise ValueError(f"min_dist {min_dist} infeasible on {h}x{w} grid (d_max={d_max})")
    cells = [(r, c) for r in range(h) for c in range(w)]
    while True:
        cop = rng.choice(cells)
        thief = rng.choice(cells)
        if manhattan(cop, thief) > min_dist:
            return cop, thief
