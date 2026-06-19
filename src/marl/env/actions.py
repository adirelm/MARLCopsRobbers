"""Action space and per-agent legality mask for the CopsRobbersEnv (T1.2).

`Action` pins the joint move indices (UP..STAY); `DELTAS` maps each to a grid
step; `action_mask` returns the boolean legality vector the policy samples over.
Barrier placement is cop-only and budgeted; STAY is config-gated OFF by default.
All tunable bounds (action count, barrier budget, toggles) are read from config —
nothing is hardcoded here (CLAUDE.md §4).
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from src.marl.env.grid import can_enter
from src.marl.env.types import GlobalState, Pos


class Action(IntEnum):
    """Joint discrete action set; integer values are pinned and frozen.

    Move indices UP..RIGHT are contiguous 0..3, PLACE_BARRIER is 4, and STAY is
    a reserved 5 that is config-gated OFF (excluded from the mask by default).
    """

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    PLACE_BARRIER = 4
    STAY = 5


# Per-action grid step (drow, dcol). PLACE_BARRIER and STAY are no-op moves.
DELTAS: dict[Action, Pos] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
    Action.PLACE_BARRIER: (0, 0),
    Action.STAY: (0, 0),
}

# The four directional moves, in mask-index order (0..3).
_MOVES: tuple[Action, ...] = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)


def action_mask(state: GlobalState, role: str, cfg: dict, idx: int = 0) -> np.ndarray:
    """Return the boolean legality mask over the cop action set.

    Indices 0..3 are the directional moves and index 4 is PLACE_BARRIER. STAY
    (index 5) is included only when ``env.actions.enable_stay`` is True (the mask
    length grows to ``a_cop + 1`` in that case). For ``role == "thief"`` the mask
    is still length ``a_cop`` (PLACE_BARRIER at index 4 is forced False); it only
    becomes ``a_cop + 1`` when STAY is enabled. The mask is NEVER all-False over
    its move indices: a boxed-in actor falls back to all-True over the four move
    indices (transition no-ops any blocked move to a stay), REGARDLESS of whether
    PLACE is legal — a boxed-in cop with budget still gets the move no-op, not a
    forced self-cell PLACE.

    Args:
        state: The current global state (provides positions, barriers, budget).
        role: Either ``"cop"`` or ``"thief"``; only cops may place barriers.
        cfg: The loaded config dict (reads env.actions.* and game.max_barriers).
        idx: Which cop to mask when there are multiple cops (default 0).

    Returns:
        A boolean ``np.ndarray`` of length ``env.actions.a_cop`` (plus one if
        STAY is enabled), True at every legal action index.

    Raises:
        ValueError: If ``role`` is not ``"cop"`` or ``"thief"``, or if ``idx``
            is out of range for ``state.cop_pos`` on the cop branch.
    """
    if role not in ("cop", "thief"):
        raise ValueError(f"unknown role {role!r} (expected 'cop' or 'thief')")
    actions_cfg = cfg["env"]["actions"]
    length = actions_cfg["a_cop"] + (1 if actions_cfg["enable_stay"] else 0)
    mask = np.zeros(length, dtype=np.bool_)

    if role == "cop":
        if not 0 <= idx < len(state.cop_pos):
            raise ValueError(f"cop idx {idx} out of range for {len(state.cop_pos)} cops")
        pos: Pos = state.cop_pos[idx]
    else:
        pos = state.thief_pos
    n_moves = len(_MOVES)
    for i, move in enumerate(_MOVES):
        delta = DELTAS[move]
        target: Pos = (pos[0] + delta[0], pos[1] + delta[1])
        mask[i] = can_enter(target, pos, state.barriers, state.h, state.w)

    if role == "cop":
        budget_left = state.barriers_used < cfg["game"]["max_barriers"]
        mask[int(Action.PLACE_BARRIER)] = actions_cfg["enable_barrier"] and budget_left

    if actions_cfg["enable_stay"]:
        mask[int(Action.STAY)] = True

    moves_legal = mask[:n_moves].any()
    if not moves_legal:
        # Boxed-in fallback: expose the four moves (transition stays put on each),
        # regardless of PLACE/STAY legality — never a forced self-cell PLACE.
        mask[:n_moves] = True

    return mask
