"""Egocentric LOCAL observation builder for the CopsRobbersEnv (T2.2).

`view_radius` resolves the Manhattan view radius per grid size (config override
``env.view_radius_by_grid`` else the auto ``max(0, ceil(min(h,w)/2)-1)``).
`build_observation` assembles the EXEC-TIME LOCAL :class:`Observation` (egocentric
image via :mod:`observation_encoder` + the ``obs_scalars`` aliasing-memory hooks)
for one AGENT KEY (``cop_{i}`` or ``thief``). It is PURE given the env-owned
per-agent visibility ``memory`` keyed by agent key (T2.3). The thief sees the
WHOLE cop TEAM; each cop sees only the thief. NO absolute opponent position leaks:
an opponent is encoded only within the view radius, and on-board cells beyond the
radius are fogged (``out_of_bounds=1``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.marl.env.observation_encoder import encode_image, opponent_in_view
from src.marl.env.types import GlobalState, Observation, Pos

__all__ = ["VisibilityMemory", "build_observation", "opponent_in_view", "view_radius"]


@dataclass
class VisibilityMemory:
    """Env-owned per-agent visibility memory feeding the aliasing-memory scalars.

    Attributes:
        steps_since_seen: Steps since this agent last saw ANY opponent in view
            (reset to 0 when an opponent is in view, else incremented by the env).
    """

    steps_since_seen: int = 0


def view_radius(h: int, w: int, cfg: dict) -> int:
    """Return the Manhattan view radius for an ``h x w`` board.

    Uses the ``env.view_radius_by_grid`` override keyed by ``min(h, w)`` when
    present, otherwise the auto fallback ``max(0, ceil(min(h, w) / 2) - 1)``.

    Args:
        h: Board rows.
        w: Board columns.
        cfg: The loaded config (reads ``env.view_radius_by_grid``).

    Returns:
        The non-negative integer view radius.
    """
    size = min(h, w)
    by_grid = cfg["env"]["view_radius_by_grid"]
    if size in by_grid:
        return int(by_grid[size])
    return max(0, math.ceil(size / 2) - 1)


def _norm(value: int, span: int) -> float:
    """Normalize ``value`` by ``span`` (guarding a zero/degenerate span)."""
    return value / span if span > 0 else 0.0


def _resolve_view(state: GlobalState, agent_key: str) -> tuple[Pos, list[Pos]]:
    """Return ``(center, opponents)`` for ``agent_key``.

    A ``cop_{i}`` is centered on ``cop_pos[i]`` and sees only ``[thief_pos]``; the
    ``thief`` is centered on ``thief_pos`` and sees the WHOLE cop team.

    Args:
        state: The current global state.
        agent_key: ``"thief"`` or ``"cop_{i}"``.

    Returns:
        A ``(center, opponents)`` pair of the agent's cell and its opponent cells.
    """
    if agent_key == "thief":
        return state.thief_pos, list(state.cop_pos)
    idx = int(agent_key.split("_", 1)[1])
    return state.cop_pos[idx], [state.thief_pos]


def build_observation(state: GlobalState, agent_key: str, memory: dict, cfg: dict) -> Observation:
    """Build the egocentric LOCAL observation for ``agent_key``.

    The image is the egocentric ``(obs_channels, W_v, W_v)`` window centered on the
    agent; each on-board opponent is marked only within :func:`view_radius` and
    on-board cells beyond the radius are fogged. The scalars are the ``obs_scalars``
    aliasing-memory hooks, all normalized into ``[0, 1]``. PURE given ``memory``.

    Args:
        state: The current global state (train-time; only LOCAL fields are read).
        agent_key: ``"thief"`` or ``"cop_{i}"`` (replaces the old role+idx pair).
        memory: Env-owned dict ``{agent_key: VisibilityMemory}`` supplying
            ``steps_since_seen_norm`` for this agent.
        cfg: The loaded config (reads ``env.*`` and ``game.*``).

    Returns:
        An :class:`Observation` with ``image`` and ``scalars`` numpy arrays.
    """
    env_cfg, game_cfg = cfg["env"], cfg["game"]
    center, opponents = _resolve_view(state, agent_key)

    radius = view_radius(state.h, state.w, cfg)
    max_moves = game_cfg["max_moves"]
    time_norm = state.step / max_moves
    image = encode_image(
        center,
        opponents,
        state.barriers,
        state.h,
        state.w,
        radius,
        env_cfg["view_radius_max"],
        env_cfg["obs_channels"],
        time_norm,
    )

    seen_now = any(opponent_in_view(center, opp, radius) for opp in opponents)
    max_barriers = game_cfg["max_barriers"]
    steps_since_seen = memory[agent_key].steps_since_seen
    scalars = np.array(
        [
            _norm(center[0], state.h - 1),
            _norm(center[1], state.w - 1),
            time_norm,
            _norm(max_barriers - state.barriers_used, max_barriers),
            1.0 if seen_now else 0.0,
            min(steps_since_seen, max_moves) / max_moves,
        ],
        dtype=np.float32,
    )
    return Observation(image=image, scalars=scalars)
