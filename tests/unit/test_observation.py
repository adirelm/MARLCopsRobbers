"""Unit tests for src/marl/env/observation.py (T2.2) — egocentric LOCAL obs.

Covers view_radius (config override + auto fallback) and build_observation keyed
by AGENT KEY (``cop_{i}`` / ``thief``): the PINNED image shape (obs_channels, 5, 5)
across square 2x2/3x3/4x4/5x5 AND a non-square 3x5 board, partial-observability FOG
(an on-board cell beyond the Manhattan radius is out_of_bounds — a 2x2 radius-0
window is near-blind, only the self cell), self=1 at the window center,
other_visible=1 WITHIN the radius and 0 BEYOND it (even when in-window),
off-window barriers are NOT marked, the visible-barrier channel, the time_norm
broadcast, and every scalar staying in [0, 1]. The thief sees the WHOLE cop team.
build_observation is PURE given the env-owned per-agent visibility memory.
"""

from __future__ import annotations

import math

import numpy as np

from src.marl.env.observation import (
    VisibilityMemory,
    build_observation,
    opponent_in_view,
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


def test_view_radius_uses_config_override(cfg):
    by_grid = cfg["env"]["view_radius_by_grid"]
    assert view_radius(5, 5, cfg) == by_grid[5]
    assert view_radius(4, 4, cfg) == by_grid[4]
    assert view_radius(3, 3, cfg) == by_grid[3]
    assert view_radius(2, 2, cfg) == by_grid[2]


def test_view_radius_auto_fallback_when_no_override(cfg):
    # A 7x7 grid has no by-grid entry -> auto = max(0, ceil(min(h,w)/2)-1).
    assert 7 not in cfg["env"]["view_radius_by_grid"]
    assert view_radius(7, 7, cfg) == max(0, math.ceil(7 / 2) - 1)


def test_opponent_in_view_predicate():
    # The shared DRY helper: visible iff Manhattan distance <= radius.
    assert opponent_in_view((2, 2), (2, 2), 0) is True
    assert opponent_in_view((2, 2), (2, 3), 0) is False
    assert opponent_in_view((2, 2), (0, 2), 2) is True
    assert opponent_in_view((0, 0), (4, 4), 2) is False


def test_image_shape_is_5x5x5_across_grid_sizes(cfg, make_state):
    channels = cfg["env"]["obs_channels"]
    wv = 2 * cfg["env"]["view_radius_max"] + 1
    for n in (2, 3, 4, 5):
        thief = (n - 1, n - 1)
        state = make_state(cop_pos=(0, 0), thief_pos=thief, h=n, w=n)
        obs = build_observation(state, "cop_0", _mem(), cfg)
        assert obs["image"].shape == (channels, wv, wv) == (5, 5, 5)
        assert obs["image"].dtype == np.float32


def test_self_marked_at_window_center(cfg, make_state):
    state = make_state(cop_pos=(1, 1), thief_pos=(4, 4), h=5, w=5)
    obs = build_observation(state, "cop_0", _mem(), cfg)
    img = obs["image"]
    assert img[SELF, CENTER, CENTER] == 1.0
    # self channel is hot ONLY at the center.
    assert img[SELF].sum() == 1.0


def test_radius0_window_is_near_blind_on_2x2(cfg, make_state):
    # On a 2x2 board view_radius==0: the visible region is the Manhattan disk of
    # radius 0 = the self cell ONLY. Every other window cell is fogged (OOB=1):
    # off-board cells AND the on-board neighbours that lie beyond the radius.
    assert view_radius(2, 2, cfg) == 0
    state = make_state(cop_pos=(0, 0), thief_pos=(1, 1), h=2, w=2)
    obs = build_observation(state, "cop_0", _mem(), cfg)
    oob = obs["image"][OOB]
    # Only the self cell (window center) is visible -> not OOB; all 24 others are.
    assert oob[CENTER, CENTER] == 0.0
    assert oob.sum() == 24.0
    # The thief is on-board but beyond the radius -> NOT marked anywhere.
    assert obs["image"][OTHER].sum() == 0.0


def test_other_visible_within_radius(cfg, make_state):
    # 5x5 radius=2: place the thief at Manhattan distance == radius -> visible.
    r = view_radius(5, 5, cfg)
    cop = (2, 2)
    thief = (2, 2 + r)  # manhattan == r
    state = make_state(cop_pos=cop, thief_pos=thief, h=5, w=5)
    obs = build_observation(state, "cop_0", _mem(), cfg)
    img = obs["image"]
    # other appears at the window offset (dr, dc) = (0, r) from center.
    assert img[OTHER, CENTER, CENTER + r] == 1.0
    assert img[OTHER].sum() == 1.0
    assert obs["scalars"][4] == 1.0  # other_seen_now


def test_other_visible_zero_beyond_window(cfg, make_state):
    cop = (0, 0)
    thief = (4, 4)  # manhattan == 8 > r and outside the window -> NOT visible
    r = view_radius(5, 5, cfg)
    assert (abs(4 - 0) + abs(4 - 0)) > r
    state = make_state(cop_pos=cop, thief_pos=thief, h=5, w=5)
    obs = build_observation(state, "cop_0", _mem(), cfg)
    assert obs["image"][OTHER].sum() == 0.0
    assert obs["scalars"][4] == 0.0  # other_seen_now


def test_barrier_channel_marks_visible_barrier(cfg, make_state):
    # A barrier one cell to the cop's right is inside the radius -> marked.
    cop = (2, 2)
    state = make_state(cop_pos=cop, thief_pos=(4, 4), barriers=[(2, 3)], barriers_used=1, h=5, w=5)
    obs = build_observation(state, "cop_0", _mem(), cfg)
    img = obs["image"]
    assert img[BARRIER, CENTER, CENTER + 1] == 1.0
    assert img[BARRIER].sum() == 1.0


def test_time_norm_channel_broadcast(cfg, make_state):
    max_moves = cfg["game"]["max_moves"]
    step = 5
    state = make_state(cop_pos=(2, 2), thief_pos=(4, 4), step=step, h=5, w=5)
    obs = build_observation(state, "cop_0", _mem(), cfg)
    expected = step / max_moves
    assert np.allclose(obs["image"][TIME], expected)


def test_scalars_in_unit_interval(cfg, make_state):
    state = make_state(
        cop_pos=(3, 4),
        thief_pos=(3, 3),
        barriers_used=2,
        step=10,
        h=5,
        w=5,
    )
    obs = build_observation(state, "cop_0", _mem(steps_since_seen=3), cfg)
    scalars = obs["scalars"]
    assert scalars.shape == (cfg["env"]["obs_scalars"],) == (6,)
    assert scalars.dtype == np.float32
    assert np.all(scalars >= 0.0) and np.all(scalars <= 1.0)


def test_steps_since_seen_norm_clamped(cfg, make_state):
    # A huge steps_since_seen clamps to max_moves -> norm == 1.0.
    state = make_state(cop_pos=(0, 0), thief_pos=(4, 4), h=5, w=5)
    obs = build_observation(state, "cop_0", _mem(steps_since_seen=10_000), cfg)
    assert obs["scalars"][5] == 1.0
