"""Train-time Manhattan-heuristic ORACLES for behavior cloning (T3.2).

Two deterministic experts that consume a :class:`~src.marl.env.types.GlobalState`
and return a single :class:`~src.marl.env.actions.Action`:

* :func:`cop_expert` runs a barrier-aware BFS from ``cop_pos[idx]`` toward the
  thief and plays the FIRST step on a shortest path (lowest ``Action`` enum index
  tie-break among equal-cost first moves).
* :func:`thief_expert` greedily picks the legal 1-step move that maximizes the
  Manhattan distance to the NEAREST cop (lowest-index tie-break).

Both honor ``can_enter`` (grid edges + barriers) and support epsilon-greedy
label diversity (a random LEGAL move with probability ``epsilon`` via ``rng``).
These are pure given their inputs; no config-mutating side effects.
"""

from __future__ import annotations

from collections import deque
from random import Random

from src.marl.env.actions import DELTAS, Action
from src.marl.env.grid import can_enter, manhattan
from src.marl.env.types import GlobalState, Pos

# The four directional moves in ascending Action-index order (the tie-break order).
_MOVES: tuple[Action, ...] = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)


def _legal_moves(pos: Pos, state: GlobalState) -> list[Action]:
    """Return the directional moves that keep an actor on-board past barriers.

    Args:
        pos: The actor's current cell.
        state: The global state (supplies barriers + board bounds).

    Returns:
        The subset of ``_MOVES`` whose target cell is enterable, in index order.
    """
    legal: list[Action] = []
    for move in _MOVES:
        drow, dcol = DELTAS[move]
        target: Pos = (pos[0] + drow, pos[1] + dcol)
        if can_enter(target, pos, state.barriers, state.h, state.w):
            legal.append(move)
    return legal


def _maybe_explore(legal: list[Action], rng: Random | None, epsilon: float) -> Action | None:
    """Return a random legal move with prob ``epsilon`` (else ``None``).

    Guards the boxed-in case (B1): when ``legal`` is empty (a fully-walled
    actor) it returns ``None`` instead of calling ``rng.choice([])`` (which
    raises ``IndexError``), letting the caller fall back to a safe no-op move.

    Args:
        legal: The legal directional moves (possibly empty when boxed in).
        rng: A seeded RNG; required when ``epsilon > 0``.
        epsilon: Exploration probability in ``[0, 1]``.

    Returns:
        A randomly chosen legal :class:`Action`, or ``None`` to act greedily.
    """
    if legal and epsilon > 0.0 and rng is not None and rng.random() < epsilon:
        return rng.choice(legal)
    return None


def _bfs_first_step(start: Pos, goal: Pos, state: GlobalState) -> Action | None:
    """Return the lowest-index first move on a shortest barrier-aware path.

    Breadth-first search expands cells in ascending ``Action`` order so the first
    time ``goal`` is dequeued its recorded opening move is a lowest-index shortest
    step. Returns ``None`` when ``goal`` is unreachable.

    Args:
        start: The cop's current cell.
        goal: The thief's cell (the BFS target).
        state: The global state (barriers + bounds).

    Returns:
        The first :class:`Action` on a shortest path, or ``None`` if unreachable.
    """
    if start == goal:
        return None
    first: dict[Pos, Action] = {}
    queue: deque[Pos] = deque([start])
    seen: set[Pos] = {start}
    while queue:
        cell = queue.popleft()
        for move in _MOVES:
            drow, dcol = DELTAS[move]
            nxt: Pos = (cell[0] + drow, cell[1] + dcol)
            if nxt in seen or not can_enter(nxt, cell, state.barriers, state.h, state.w):
                continue
            seen.add(nxt)
            first[nxt] = first.get(cell, move)
            if nxt == goal:
                return first[nxt]
            queue.append(nxt)
    return None


def cop_expert(
    state: GlobalState,
    cfg: dict,
    idx: int = 0,
    rng: Random | None = None,
    epsilon: float = 0.0,
) -> Action:
    """Return the BFS shortest-path first step for cop ``idx`` toward the thief.

    Args:
        state: The current global state.
        cfg: The loaded config (unused geometry-wise; kept for signature parity).
        idx: Which cop to drive (default 0).
        rng: Optional seeded RNG for epsilon-greedy label diversity.
        epsilon: Probability of a random LEGAL move instead of the greedy step.

    Returns:
        The chosen :class:`Action` (a legal directional move; falls back to the
        lowest-index legal move when the thief is already reached/unreachable).
    """
    pos = state.cop_pos[idx]
    legal = _legal_moves(pos, state)
    explore = _maybe_explore(legal, rng, epsilon)
    if explore is not None:
        return explore
    step = _bfs_first_step(pos, state.thief_pos, state)
    if step is not None:
        return step
    return legal[0] if legal else Action.UP


def thief_expert(
    state: GlobalState,
    cfg: dict,
    rng: Random | None = None,
    epsilon: float = 0.0,
) -> Action:
    """Return the greedy 1-step move maximizing distance to the nearest cop.

    Among the legal directional moves, picks the one whose resulting cell has the
    largest Manhattan distance to the closest cop; ties resolve to the lowest
    ``Action`` index. Supports epsilon-greedy diversity via ``rng``.

    Args:
        state: The current global state.
        cfg: The loaded config (kept for signature parity; geometry only here).
        rng: Optional seeded RNG for epsilon-greedy label diversity.
        epsilon: Probability of a random LEGAL move instead of the greedy step.

    Returns:
        The chosen :class:`Action` (a legal directional move).
    """
    pos = state.thief_pos
    legal = _legal_moves(pos, state)
    explore = _maybe_explore(legal, rng, epsilon)
    if explore is not None:
        return explore
    best_move = legal[0] if legal else Action.UP
    best_dist = -1
    for move in legal:
        drow, dcol = DELTAS[move]
        nxt: Pos = (pos[0] + drow, pos[1] + dcol)
        dist = min(manhattan(nxt, cop) for cop in state.cop_pos)
        if dist > best_dist:
            best_dist = dist
            best_move = move
    return best_move
