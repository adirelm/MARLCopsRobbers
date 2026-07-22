"""Shared scripted fixtures for the foreign-cop battery tests (DRY seam).

Mirrors the `_buffer_fixtures` pattern: tiny obs/mask builders + a stationary
scripted thief, imported by `test_foreign_cops.py` and
`test_foreign_cops_battery.py` so neither test file duplicates them.
"""

from __future__ import annotations

import numpy as np

from src.marl.env.actions import Action

STAGE = (5, 5, 1)
MOVES_ONLY = [True, True, True, True, False]
WITH_PLACE = [True, True, True, True, True]


def make_obs(cfg: dict, thief_rel: tuple[int, int] | None = None) -> dict:
    """Egocentric obs with an optionally visible thief at a (drow, dcol) offset.

    Args:
        cfg: The loaded config (obs geometry: view_radius_max/channels/scalars).
        thief_rel: Window offset of the visible thief, or None for no sighting.

    Returns:
        An Observation-shaped dict (image + scalars) for one cop.
    """
    center = int(cfg["env"]["view_radius_max"])
    side = 2 * center + 1
    image = np.zeros((int(cfg["env"]["obs_channels"]), side, side), dtype=np.float32)
    image[0, center, center] = 1.0
    if thief_rel is not None:
        image[1, center + thief_rel[0], center + thief_rel[1]] = 1.0
    return {"image": image, "scalars": np.zeros(int(cfg["env"]["obs_scalars"]), dtype=np.float32)}


class UpThief:
    """Scripted thief: always UP — pins to the top wall, then stays put (stationary)."""

    def reset(self) -> None:
        """No state."""

    def act(self, _obs: list, _masks: list, _eps: float, _rng: object, state: object = None) -> list:
        """Always head UP (the wall no-ops it into a stay)."""
        return [Action.UP]
