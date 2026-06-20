"""Unit tests for src/marl/env/observation.py — P2-fix partial-observability cases.

Split out of test_observation.py to keep both files within the 150-LOC gate.
Covers the non-square board footprint, in-window-but-beyond-radius fog for the
``other_visible``/barrier channels, and the thief role seeing the WHOLE cop team
(within the egocentric Manhattan radius). build_observation is PURE given the
env-owned per-agent visibility memory; cfg/make_state come from conftest.
"""

from __future__ import annotations

from src.marl.env.observation import (
    VisibilityMemory,
    build_observation,
    view_radius,
)

# Channel order is PINNED by the P2 spec / Observation TypedDict.
SELF, OTHER, BARRIER, OOB, TIME = 0, 1, 2, 3, 4
CENTER = 2  # window center index = view_radius_max (2)


def _mem(
    steps_since_seen: int = 999, keys: tuple[str, ...] = ("cop_0", "thief")
) -> dict[str, VisibilityMemory]:
    """Build a fresh per-agent visibility-memory dict for build_observation."""
    return {key: VisibilityMemory(steps_since_seen=steps_since_seen) for key in keys}


def test_image_shape_is_5x5x5_on_non_square_board(cfg, make_state):
    # A non-square 3x5 board still pads to the 5x5 footprint; self centered.
    channels = cfg["env"]["obs_channels"]
    state = make_state(cop_pos=(1, 2), thief_pos=(0, 0), h=3, w=5)
    obs = build_observation(state, "cop_0", _mem(), cfg)
    assert obs["image"].shape == (channels, 5, 5) == (5, 5, 5)
    assert obs["image"][SELF, CENTER, CENTER] == 1.0
    assert obs["image"][SELF].sum() == 1.0


def test_other_visible_zero_in_window_but_beyond_radius(cfg, make_state):
    # cop(2,2), thief(0,0) on 5x5: thief IS inside the 5x5 window (offset -2,-2)
    # but manhattan == 4 > radius 2 -> partial-obs fog -> NOT marked.
    r = view_radius(5, 5, cfg)
    cop = (2, 2)
    thief = (0, 0)
    assert abs(0 - 2) + abs(0 - 2) > r  # in window, out of radius
    state = make_state(cop_pos=cop, thief_pos=thief, h=5, w=5)
    obs = build_observation(state, "cop_0", _mem(), cfg)
    assert obs["image"][OTHER].sum() == 0.0
    assert obs["scalars"][4] == 0.0  # other_seen_now


def test_off_window_barrier_not_marked(cfg, make_state):
    # A barrier in-window but beyond the radius (offset -2,-2; manhattan 4 > 2)
    # is FOGGED -> the barrier channel must stay empty there.
    cop = (2, 2)
    state = make_state(cop_pos=cop, thief_pos=(4, 4), barriers=[(0, 0)], barriers_used=1, h=5, w=5)
    obs = build_observation(state, "cop_0", _mem(), cfg)
    assert obs["image"][BARRIER].sum() == 0.0
    # And the fogged corner carries OOB, not barrier.
    assert obs["image"][OOB, 0, 0] == 1.0


def test_thief_role_sees_cop_within_radius(cfg, make_state):
    # The thief's obs is egocentric on the thief; a cop is the "other".
    r = view_radius(5, 5, cfg)
    thief = (2, 2)
    cop = (2, 2 - r)
    state = make_state(cop_pos=cop, thief_pos=thief, h=5, w=5)
    obs = build_observation(state, "thief", _mem(), cfg)
    img = obs["image"]
    assert img[SELF, CENTER, CENTER] == 1.0
    assert img[OTHER, CENTER, CENTER - r] == 1.0


def test_thief_sees_both_cops_when_both_in_radius(cfg, make_state):
    # 2-cop case: the thief sees the WHOLE team -> both in-radius cops marked.
    r = view_radius(5, 5, cfg)
    thief = (2, 2)
    cop_a = (2, 2 - r)  # offset (0, -r)
    cop_b = (2 + r, 2)  # offset (+r, 0)
    state = make_state(cop_pos=[cop_a, cop_b], thief_pos=thief, h=5, w=5)
    obs = build_observation(state, "thief", _mem(keys=("cop_0", "cop_1", "thief")), cfg)
    img = obs["image"]
    assert img[OTHER, CENTER, CENTER - r] == 1.0
    assert img[OTHER, CENTER + r, CENTER] == 1.0
    assert img[OTHER].sum() == 2.0
    assert obs["scalars"][4] == 1.0  # other_seen_now (any cop in view)
