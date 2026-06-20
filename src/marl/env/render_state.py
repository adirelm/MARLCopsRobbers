"""God-view spectator serializer for the GUI / MCP report path (T2.3).

`render_state` flattens a :class:`GlobalState` into a PLAIN, JSON-serializable
dict for the Pygame god-view spectator and the report path. This is the
SANCTIONED spectator seam (NOT the agent observation path): full board dims +
absolute positions + the barrier set are allowed here for DRAWING, but the
function returns NO :class:`GlobalState` instance (it never re-exports the
train-only type across the boundary). The agent-facing executor path
(env.step/reset) stays strictly LOCAL — see cops_robbers_env.py.
"""

from __future__ import annotations

from src.marl.env.types import GlobalState


def render_state(state: GlobalState, cfg: dict) -> dict:
    """Return a plain serializable god-view dict for the spectator / report.

    Args:
        state: The train-time global state (read-only; nothing is mutated).
        cfg: The loaded config (reads ``game.max_moves`` / ``game.max_barriers``
            for the normalized progress fields).

    Returns:
        A plain dict of ints/lists/bools (no GlobalState instance) carrying the
        board dims, cop/thief positions, barrier cells, budget, and progress —
        everything the god-view needs to DRAW the board.
    """
    game = cfg["game"]
    return {
        "h": state.h,
        "w": state.w,
        "step": state.step,
        "max_moves": game["max_moves"],
        "cop_positions": [list(pos) for pos in state.cop_pos],
        "thief_position": list(state.thief_pos),
        "barriers": [list(cell) for cell in sorted(state.barriers)],
        "barriers_used": state.barriers_used,
        "max_barriers": game["max_barriers"],
        "terminal": state.terminal,
    }
