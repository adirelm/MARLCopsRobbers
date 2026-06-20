"""Pure simultaneous joint-action transition for the CopsRobbersEnv (T2.1).

`resolve_joint_action` is the deterministic, RNG-free dynamics kernel T(s'|s,a)
(BRIEF §2.1). Every agent's intended next cell is computed from the SAME pre-move
state (positions + barriers) via ``grid.resolve_move`` so the resolution is
order-independent (ADR-001 simultaneous; ``env.move_resolution``). Cop barrier
placement (``Action.PLACE_BARRIER``) drops the barrier on the cop's CURRENT cell
and the cop stays there; a TEAM budget cap (``game.max_barriers``) is iterated by
cop index, so once the cap is hit later cops' PLACE deterministically degrades to
a stay (codex F1). Capture (any cop on the thief OR a one-tick cop<->thief swap,
``env.capture_on_swap``) is checked BEFORE timeout, so a capture on the final
move beats the ``(step+1) >= game.max_moves`` timeout. Returns a NEW frozen
GlobalState at ``step+1`` — nothing is mutated (replay-deterministic).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.marl.env.actions import DELTAS, Action
from src.marl.env.grid import resolve_move
from src.marl.env.types import GlobalState, Pos


@dataclass(frozen=True)
class TransitionResult:
    """Immutable outcome of one simultaneous joint-action resolution.

    Attributes:
        next_state: The NEW frozen GlobalState at ``step+1`` with updated
            cop/thief positions, barriers, budget, and the ``terminal`` flag.
        winner: ``"cop"`` (capture), ``"thief"`` (timeout), or ``None`` (ongoing).
        capture: True iff a cop caught the thief this step (same-cell or swap).
    """

    next_state: GlobalState
    winner: str | None
    capture: bool


def _resolve_thief(state: GlobalState, action: Action) -> Pos:
    """Return the thief's post-move cell (OOB/barrier ⇒ stay), from pre-move state."""
    delta = DELTAS[action]
    return resolve_move(state.thief_pos, delta, state.thief_pos, state.barriers, state.h, state.w)


def _resolve_cops(
    state: GlobalState, joint_a: dict[str, Action], max_barriers: int
) -> tuple[tuple[Pos, ...], frozenset[Pos], int]:
    """Resolve every cop's move + barrier placement from the pre-move state.

    Iterates cops by index. A ``PLACE_BARRIER`` is honored only while
    ``barriers_used < max_barriers`` (TEAM cap, deterministic by index) AND the
    cop's cell is not ALREADY a barrier this tick: an honored placement drops a
    barrier on the cop's current cell, keeps the cop there, and increments the
    used count; once the cap is hit (or a co-located cop already placed on that
    exact cell) later cops' PLACE degrades to a stay. Keeping the increment tied
    to distinct placed cells means ``barriers_used == len(distinct placed cells)``
    so the per-barrier reward charge is applied exactly once. Non-PLACE moves
    resolve via ``grid.resolve_move`` against the original (pre-move) barrier set,
    so the resolution is order-independent.

    Args:
        state: The pre-move global state.
        joint_a: Per-agent action dict keyed ``cop_0..`` (+ ``thief``).
        max_barriers: The team barrier budget (``game.max_barriers``).

    Returns:
        A ``(new_cop_positions, new_barriers, new_barriers_used)`` triple.
    """
    barriers: frozenset[Pos] = state.barriers
    used = state.barriers_used
    new_positions: list[Pos] = []
    for i, cop in enumerate(state.cop_pos):
        action = joint_a[f"cop_{i}"]
        if action == Action.PLACE_BARRIER and used < max_barriers and cop not in barriers:
            barriers = barriers | {cop}
            used += 1
            new_positions.append(cop)  # placement IS the move ⇒ stay put
        else:
            # Over-budget PLACE or a directional move: resolve against pre-move
            # barriers (PLACE delta is (0,0) ⇒ also a stay).
            new_positions.append(resolve_move(cop, DELTAS[action], cop, state.barriers, state.h, state.w))
    return tuple(new_positions), barriers, used


def _detect_capture(
    old_cops: tuple[Pos, ...],
    new_cops: tuple[Pos, ...],
    old_thief: Pos,
    new_thief: Pos,
    capture_on_swap: bool,
) -> bool:
    """Return whether any cop captured the thief this tick (same-cell or swap)."""
    if any(cop == new_thief for cop in new_cops):
        return True
    if capture_on_swap:
        return any(
            old_cop == new_thief and new_cop == old_thief
            for old_cop, new_cop in zip(old_cops, new_cops, strict=True)
        )
    return False


def resolve_joint_action(state: GlobalState, joint_a: dict[str, Action], cfg: dict) -> TransitionResult:
    """Resolve a simultaneous joint action into a new state + outcome.

    All intended moves are computed from the SAME pre-move ``state`` (ADR-001
    simultaneous). Capture (any cop on the thief, or a cop<->thief swap when
    ``env.capture_on_swap``) is checked BEFORE timeout, so a capture on the final
    move beats the ``(step+1) >= game.max_moves`` timeout. The function is PURE
    (no RNG, no mutation) ⇒ replay-deterministic.

    Args:
        state: The pre-move global state.
        joint_a: Per-agent action dict keyed ``cop_0..cop_{n-1}`` plus ``thief``.
        cfg: The loaded config (reads ``env.move_resolution``,
            ``game.max_barriers``, ``game.max_moves``, and ``env.capture_on_swap``).

    Returns:
        A :class:`TransitionResult` with the new frozen state at ``step+1``, the
        winner (``"cop"`` / ``"thief"`` / ``None``), and the capture flag.

    Raises:
        NotImplementedError: If ``env.move_resolution`` is not ``"simultaneous"``
            (``cop_first`` / ``thief_first`` are an honest, unimplemented
            limitation rather than a silent fallback).
    """
    move_resolution = cfg["env"]["move_resolution"]
    if move_resolution != "simultaneous":
        raise NotImplementedError(
            f"env.move_resolution={move_resolution!r} is not supported; "
            "only 'simultaneous' is implemented (ADR-001)."
        )
    max_barriers = cfg["game"]["max_barriers"]
    max_moves = cfg["game"]["max_moves"]
    capture_on_swap = cfg["env"]["capture_on_swap"]

    new_cops, barriers, used = _resolve_cops(state, joint_a, max_barriers)
    new_thief = _resolve_thief(state, joint_a["thief"])

    capture = _detect_capture(state.cop_pos, new_cops, state.thief_pos, new_thief, capture_on_swap)
    step = state.step + 1
    if capture:
        winner: str | None = "cop"
    elif step >= max_moves:
        winner = "thief"
    else:
        winner = None
    terminal = winner is not None

    next_state = GlobalState(
        cop_pos=new_cops,
        thief_pos=new_thief,
        barriers=barriers,
        barriers_used=used,
        step=step,
        h=state.h,
        w=state.w,
        terminal=terminal,
    )
    return TransitionResult(next_state=next_state, winner=winner, capture=capture)
