"""Wire-payload -> local Observation reconstruction for the §9 bonus match.

The partner brief (docs/interfaces/partner_agent_brief.md §2) fixes a minimal
raw-position wire format: ``request_move`` carries ``your_pos``, ``opponent_pos``
(``null`` beyond the agreed masking radius, P5), the barrier cells within that
radius, ``barriers_left`` (both roles), and the 0-indexed ``tick``. Because the
env observation is radius-limited BY DESIGN, that payload is sufficient to
rebuild the EXACT env-emitted egocentric :class:`Observation` for the 1v1 match:
this module synthesizes a :class:`GlobalState` from the payload and routes
through the env's OWN builders (:func:`src.marl.env.observation.build_observation`
and :func:`src.marl.env.actions.action_mask`), so no geometry or scalar math is
re-implemented (DRY) and the served policy sees bit-identical inputs to the
evaluated one — proven per-tick in tests/unit/test_wire_obs.py. The
``steps_since_seen`` aliasing scalar is the one wire-absent field; it is
reproduced by applying the env's own visibility recurrence (reset-on-seen else
+1, updated BEFORE the obs is built, exactly like ``_env_helpers.update_memory``)
to a per-session counter that starts at 0 on ``new_sub_game``.
"""

from __future__ import annotations

import numpy as np

from src.marl.env.actions import action_mask
from src.marl.env.grid import manhattan
from src.marl.env.observation import VisibilityMemory, opponent_in_view, view_radius
from src.marl.env.observation import build_observation as _env_build_observation
from src.marl.env.types import GlobalState, Observation, Pos

_ROLES = ("cop", "thief")


def new_session(role: str, grid, max_moves: int, cfg: dict) -> dict:
    """Return the fresh per-sub-game wire-session state for ``role``.

    Called on every ``new_sub_game`` — including a void replay re-POST of the
    SAME session id — so the visibility counter always restarts at 0.

    Args:
        role: ``"cop"`` or ``"thief"`` (the brief's ``your_role``).
        grid: The ``[rows, cols]`` pair from the ``new_sub_game`` payload.
        max_moves: The sub-game move cap from the payload.
        cfg: The loaded config (reads ``game.*``, ``env.*``, ``mcp.observation``).

    Returns:
        The mutable session dict ``{"role", "grid", "steps_since_seen"}``.

    Raises:
        ValueError: On an unknown role; when ``max_moves`` differs from
            ``game.max_moves`` (the obs time scale would silently diverge); or
            when the wire masking radius is narrower than the env view radius
            (in-view cells would be missing -> NOT reconstructible, so refuse).
    """
    if role not in _ROLES:
        raise ValueError(f"unknown role {role!r} (expected one of {_ROLES})")
    h, w = int(grid[0]), int(grid[1])
    if int(max_moves) != int(cfg["game"]["max_moves"]):
        raise ValueError(f"max_moves {max_moves} != game.max_moves {cfg['game']['max_moves']}")
    wire_radius = int(cfg["mcp"]["observation"]["view_radius"])
    if view_radius(h, w, cfg) > wire_radius:
        raise ValueError(f"wire masking radius {wire_radius} < env view radius on {h}x{w}")
    return {"role": role, "grid": (h, w), "steps_since_seen": 0}


def _far_cell(center: Pos, h: int, w: int, radius: int) -> Pos:
    """Return a stand-in cell for an unseen opponent, guaranteed outside ``radius``.

    The farthest corner from ``center`` maximizes the Manhattan distance; if even
    that corner is within the view radius the whole board is visible and a
    ``null`` opponent cannot be represented — refuse rather than approximate.
    """
    far: Pos = (
        h - 1 if h - 1 - center[0] >= center[0] else 0,
        w - 1 if w - 1 - center[1] >= center[1] else 0,
    )
    if manhattan(far, center) <= radius:
        raise ValueError(f"no out-of-view stand-in cell on {h}x{w} at radius {radius}")
    return far


def _synth_state(session: dict, payload: dict, cfg: dict) -> GlobalState:
    """Synthesize the 1v1 :class:`GlobalState` the env builders consume.

    ``barriers_used`` is derived as ``max_barriers - barriers_left`` so the env's
    ``barriers_left_norm`` expression reproduces the wire value exactly; the
    radius-limited barrier subset is sufficient because the encoder never draws a
    barrier beyond the view disk and every mask-relevant (adjacent) cell lies
    within the wire masking radius.
    """
    h, w = session["grid"]
    you: Pos = (int(payload["your_pos"][0]), int(payload["your_pos"][1]))
    raw = payload["opponent_pos"]
    opp: Pos = _far_cell(you, h, w, view_radius(h, w, cfg)) if raw is None else (int(raw[0]), int(raw[1]))
    max_barriers = int(cfg["game"]["max_barriers"])
    left = int(payload["barriers_left"])
    if not 0 <= left <= max_barriers:
        raise ValueError(f"barriers_left {left} outside 0..{max_barriers}")
    barriers = frozenset((int(b[0]), int(b[1])) for b in payload["barriers"])
    cop, thief = (you, opp) if session["role"] == "cop" else (opp, you)
    return GlobalState(
        cop_pos=(cop,),
        thief_pos=thief,
        barriers=barriers,
        barriers_used=max_barriers - left,
        step=int(payload["tick"]),
        h=h,
        w=w,
    )


def build_observation(session: dict, payload: dict, cfg: dict) -> Observation:
    """Rebuild the env-emitted local :class:`Observation` from a wire payload.

    MUTATES ``session["steps_since_seen"]`` by the env's own update-then-build
    recurrence, so it must run exactly ONCE per (session, tick) — the wire
    agent's idempotency cache guarantees that ordering.

    Args:
        session: The :func:`new_session` state for this sub-game.
        payload: The brief's ``request_move`` body.
        cfg: The loaded config.

    Returns:
        An :class:`Observation` bit-identical to the env's for this role/state.
    """
    state = _synth_state(session, payload, cfg)
    role = session["role"]
    you, opp = (state.cop_pos[0], state.thief_pos) if role == "cop" else (state.thief_pos, state.cop_pos[0])
    seen = opponent_in_view(you, opp, view_radius(state.h, state.w, cfg))
    session["steps_since_seen"] = 0 if seen else session["steps_since_seen"] + 1
    key = "cop_0" if role == "cop" else "thief"
    memory = {key: VisibilityMemory(steps_since_seen=session["steps_since_seen"])}
    return _env_build_observation(state, key, memory, cfg)


def build_mask(session: dict, payload: dict, cfg: dict) -> np.ndarray:
    """Rebuild the env legality mask for this role from the same wire payload.

    PURE (no session mutation): every mask-relevant target cell is adjacent to
    ``your_pos`` (Manhattan 1), hence inside the radius-2 barrier report, so this
    equals the env's ``action_mask`` bit-for-bit (proven in test_wire_obs.py).
    """
    return action_mask(_synth_state(session, payload, cfg), session["role"], cfg, idx=0)
