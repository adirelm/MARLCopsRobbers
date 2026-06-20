"""Batching adapter + stage-invariant global-state encoder (T3.2).

Two PURE data encoders for the P3 data layer:

* :func:`encode_obs_batch` is a thin BATCHING ADAPTER over the env-built local
  :class:`~src.marl.env.types.Observation` arrays (already ``(C, 5, 5)`` via
  :mod:`src.marl.env.observation_encoder`). It ONLY stacks/concatenates — no
  geometry is re-implemented (DRY). The canonical training tensor is
  channels-first ``(B, C, 5, 5)`` to match the env ``Observation``.
* :func:`encode_state` produces the STAGE-INVARIANT 77-float global state so a
  centralized critic (P4) gets a fixed-size input across the 2x2->5x5 ladder.
  It is a pure data encoder (NOT a network).
"""

from __future__ import annotations

import numpy as np

from src.marl.env.types import GlobalState, Observation

# Plane indices inside the flattened (3, W, W) head of the encoded state.
_COP, _THIEF, _BARRIER = 0, 1, 2
_N_PLANES = 3


def encode_obs_batch(obs_list: list[Observation]) -> tuple[np.ndarray, np.ndarray]:
    """Stack a list of local observations into batched image/scalar tensors.

    Pure stacking adapter: the per-agent ``image`` arrays (already ``(C, 5, 5)``
    from the env encoder) become a channels-first ``(B, C, 5, 5)`` float32 batch
    and the ``scalars`` arrays become a ``(B, obs_scalars)`` float32 batch. No
    geometry or cropping is performed here (DRY over the env encoder).

    Args:
        obs_list: A non-empty list of env-built :class:`Observation` mappings.

    Returns:
        A ``(images, scalars)`` pair of float32 arrays with shapes
        ``(B, C, 5, 5)`` and ``(B, obs_scalars)``.
    """
    images = np.stack([obs["image"] for obs in obs_list]).astype(np.float32)
    scalars = np.stack([obs["scalars"] for obs in obs_list]).astype(np.float32)
    return images, scalars


def _plane_width(cfg: dict) -> int:
    """Return the stage-invariant plane width ``2 * view_radius_max + 1``."""
    return 2 * int(cfg["env"]["view_radius_max"]) + 1


def encode_state(state: GlobalState, cfg: dict) -> np.ndarray:
    """Encode a global state into the stage-invariant 77-float vector.

    Lays each cop, the thief, and every barrier onto its own ``W x W`` plane
    (``W = 2 * view_radius_max + 1 = 5``) anchored at the board's top-left so a
    smaller grid is padded (out-of-board cells stay 0). The three planes are
    flattened (cop, thief, barrier order) and the two normalized scalars
    ``[step_norm, barriers_left_norm]`` are appended, giving shape
    ``(3 * W * W + 2,) = (77,)``.

    Args:
        state: The current global state (train-time; full board knowledge).
        cfg: The loaded config (reads ``env.view_radius_max``, ``game.max_moves``,
            ``game.max_barriers``).

    Returns:
        A float32 ``np.ndarray`` of shape ``(77,)`` with every entry in ``[0, 1]``.
    """
    w = _plane_width(cfg)
    planes = np.zeros((_N_PLANES, w, w), dtype=np.float32)
    for cop in state.cop_pos:
        # Binary occupancy: two co-located cops collapse to a single 1.0
        # (idempotent set). Acceptable for P3; a P4 multi-cop centralized state
        # may need a count or per-cop plane to disambiguate overlapping cops.
        if 0 <= cop[0] < w and 0 <= cop[1] < w:
            planes[_COP, cop[0], cop[1]] = 1.0
    thief = state.thief_pos
    if 0 <= thief[0] < w and 0 <= thief[1] < w:
        planes[_THIEF, thief[0], thief[1]] = 1.0
    for barrier in state.barriers:
        if 0 <= barrier[0] < w and 0 <= barrier[1] < w:
            planes[_BARRIER, barrier[0], barrier[1]] = 1.0

    max_moves = cfg["game"]["max_moves"]
    max_barriers = cfg["game"]["max_barriers"]
    step_norm = state.step / max_moves if max_moves > 0 else 0.0
    barriers_left = max_barriers - state.barriers_used
    barriers_left_norm = barriers_left / max_barriers if max_barriers > 0 else 0.0
    scalars = np.array([step_norm, barriers_left_norm], dtype=np.float32)
    return np.concatenate([planes.reshape(-1), scalars]).astype(np.float32)
