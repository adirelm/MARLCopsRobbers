"""Action space and per-agent legality mask for the CopsRobbersEnv (T1.2).

`Action` pins the joint move indices (UP..STAY); `DELTAS` maps each to a grid
step; `action_mask` returns the boolean legality vector the policy samples over.
Barrier placement is cop-only, budgeted, and only legal while the cop's CURRENT
cell is not already a barrier (mask agrees with the transition — codex W2 M1).
STAY is a reserved enum value that is NOT wired end-to-end (net heads and replay
masks are ``a_cop``-wide), so ``enable_stay: true`` fails LOUDLY (codex W2 M3).
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
    a reserved 5 that is NOT wired end-to-end: ``enable_stay: true`` makes
    ``action_mask`` raise loudly rather than emit a mask no net head can consume.
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

    Indices 0..3 are the directional moves and index 4 is PLACE_BARRIER, which is
    legal only while barrier budget remains AND the cop's current cell is not
    already a barrier — exactly the conditions the transition honors, so a
    mask-legal PLACE can never silently degrade to a free stay (codex W2 M1).
    For ``role == "thief"`` PLACE_BARRIER at index 4 is forced False. The mask is
    NEVER all-False over its move indices: a boxed-in actor falls back to
    all-True over the four move indices (transition no-ops any blocked move to a
    stay), REGARDLESS of whether PLACE is legal — a boxed-in cop with budget
    still gets the move no-op, not a forced self-cell PLACE.

    Args:
        state: The current global state (provides positions, barriers, budget).
        role: Either ``"cop"`` or ``"thief"``; only cops may place barriers.
        cfg: The loaded config dict (reads env.actions.* and game.max_barriers).
        idx: Which cop to mask when there are multiple cops (default 0).

    Returns:
        A boolean ``np.ndarray`` of length ``env.actions.a_cop``, True at every
        legal action index.

    Raises:
        ValueError: If ``role`` is not ``"cop"`` or ``"thief"``, if ``idx`` is
            out of range for ``state.cop_pos`` on the cop branch, or if
            ``env.actions.enable_stay`` is True — STAY is not wired end-to-end
            (net heads, replay masks and the shipped trained artifacts are all
            ``a_cop``-wide), so the toggle fails loudly instead of training a
            policy that can never select the action it exposes (codex W2 M3).
    """
    if role not in ("cop", "thief"):
        raise ValueError(f"unknown role {role!r} (expected 'cop' or 'thief')")
    actions_cfg = cfg["env"]["actions"]
    if actions_cfg["enable_stay"]:
        raise ValueError(
            "env.actions.enable_stay is not wired end-to-end: network heads, replay "
            "masks and the shipped trained artifacts are fixed at a_cop widths, so a "
            "STAY-widened mask can never be consumed. Keep enable_stay: false."
        )
    mask = np.zeros(actions_cfg["a_cop"], dtype=np.bool_)

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
        # The transition also refuses to place on a cell that is ALREADY a barrier
        # (the cop stands on its own placement after placing) — mirror that here so
        # the mask never advertises a PLACE that would resolve to a free stay.
        cell_placeable = pos not in state.barriers
        mask[int(Action.PLACE_BARRIER)] = actions_cfg["enable_barrier"] and budget_left and cell_placeable

    moves_legal = mask[:n_moves].any()
    if not moves_legal:
        # Boxed-in fallback: expose the four moves (transition stays put on each),
        # regardless of PLACE/STAY legality — never a forced self-cell PLACE.
        mask[:n_moves] = True

    return mask
