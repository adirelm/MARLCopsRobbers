"""Config-derived network dimensions (P4a Stage 1, T4.0b).

The SINGLE source for every net/mixer dimension so no module hardcodes a
shape (CLAUDE.md §4). Each helper reads the loaded config and mirrors a P3
data encoder so the nets stay consistent with the data layer:

* :func:`obs_dim` matches the flattened local observation
  (``obs_channels * (2*view_radius_max+1)**2 + obs_scalars`` = 131).
* :func:`state_dim` matches :func:`src.marl.data.obs_encoder.encode_state`
  (``3 * (2*view_radius_max+1)**2 + 2`` = 77).
"""

from __future__ import annotations

# Plane count baked into encode_state (cop, thief, barrier) and the 2 scalars
# it appends. These mirror the P3 encoder contract, not tunable knobs.
_STATE_PLANES = 3
_STATE_SCALARS = 2


def _window(cfg: dict) -> int:
    """Return the stage-invariant window width ``2 * view_radius_max + 1``."""
    return 2 * int(cfg["env"]["view_radius_max"]) + 1


def obs_dim(cfg: dict) -> int:
    """Return the flattened local-observation dimension (=131 for the 5x5 stage).

    Equals ``obs_channels * window**2 + obs_scalars`` where
    ``window = 2 * view_radius_max + 1``, matching the flattened P3 local obs
    (image channels + scalar hooks).

    Args:
        cfg: The loaded config (reads ``env.obs_channels``,
            ``env.view_radius_max``, ``env.obs_scalars``).

    Returns:
        The integer input width of the agent-net encoder (before agent-id).
    """
    env = cfg["env"]
    return int(env["obs_channels"]) * _window(cfg) ** 2 + int(env["obs_scalars"])


def hidden_dim(cfg: dict) -> int:
    """Return the GRU hidden width ``nets.hidden_dim`` (=64).

    Args:
        cfg: The loaded config (reads ``nets.hidden_dim``).

    Returns:
        The recurrent hidden-state width H.
    """
    return int(cfg["nets"]["hidden_dim"])


def action_dim(cfg: dict, role: str) -> int:
    """Return the role action-space size (cop=``a_cop``=5, thief=``a_thief``=4).

    Args:
        cfg: The loaded config (reads ``env.actions.a_cop`` / ``a_thief``).
        role: Either ``"cop"`` or ``"thief"``.

    Returns:
        The number of discrete actions for ``role``.

    Raises:
        ValueError: If ``role`` is not ``"cop"`` or ``"thief"``.
    """
    actions = cfg["env"]["actions"]
    if role == "cop":
        return int(actions["a_cop"])
    if role == "thief":
        return int(actions["a_thief"])
    raise ValueError(f"unknown role {role!r}: expected 'cop' or 'thief'")


def state_dim(cfg: dict) -> int:
    """Return the global-state dimension (=77), matching ``encode_state``.

    Equals ``3 * window**2 + 2`` where ``window = 2 * view_radius_max + 1`` —
    the cop/thief/barrier planes plus ``[step_norm, barriers_left_norm]``.

    Args:
        cfg: The loaded config (reads ``env.view_radius_max``).

    Returns:
        The flattened global-state width consumed by the QMIX mixer.
    """
    return _STATE_PLANES * _window(cfg) ** 2 + _STATE_SCALARS
