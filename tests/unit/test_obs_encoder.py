"""Unit tests for src.marl.data.obs_encoder (T3.2).

Covers the batching adapter encode_obs_batch (shapes/dtype, pure stacking) and
the stage-invariant encode_state 77-float encoder (shape, value range, planes).
"""

from __future__ import annotations

import numpy as np

from src.marl.data.obs_encoder import encode_obs_batch, encode_state
from src.marl.env.observation import VisibilityMemory, build_observation

# Plane layout inside the flattened (3, 5, 5) head of encode_state.
_W = 5
_PLANE = _W * _W
_COP, _THIEF, _BARRIER = 0, 1, 2
_STATE_DIM = 3 * _PLANE + 2


def _obs_list(make_state, cfg, n: int = 3):
    state = make_state(cop_pos=(0, 0), thief_pos=(4, 4))
    mem = {"thief": VisibilityMemory(), "cop_0": VisibilityMemory()}
    return [build_observation(state, "cop_0", mem, cfg) for _ in range(n)]


def test_encode_obs_batch_shapes_and_dtype(make_state, cfg):
    """encode_obs_batch stacks to (B,C,5,5) images and (B,obs_scalars) scalars."""
    obs_list = _obs_list(make_state, cfg, n=3)
    images, scalars = encode_obs_batch(obs_list)
    channels = cfg["env"]["obs_channels"]
    n_scalars = cfg["env"]["obs_scalars"]
    assert images.shape == (3, channels, _W, _W)
    assert scalars.shape == (3, n_scalars)
    assert images.dtype == np.float32
    assert scalars.dtype == np.float32


def test_encode_obs_batch_is_pure_stack(make_state, cfg):
    """encode_obs_batch only stacks: row i equals the i-th Observation arrays."""
    obs_list = _obs_list(make_state, cfg, n=2)
    images, scalars = encode_obs_batch(obs_list)
    np.testing.assert_array_equal(images[0], obs_list[0]["image"])
    np.testing.assert_array_equal(scalars[1], obs_list[1]["scalars"])


def _distinct_obs(channels: int, n_scalars: int, tag: int):
    """Build a synthetic Observation whose every cell encodes (tag, c, r, col).

    The image is filled so image[c, r, col] = tag*1000 + c*100 + r*10 + col,
    making every (item, channel, row, col) value unique — any reordering or a
    channels-last layout would scramble these probes detectably.
    """
    image = np.empty((channels, _W, _W), dtype=np.float32)
    for c in range(channels):
        for r in range(_W):
            for col in range(_W):
                image[c, r, col] = tag * 1000 + c * 100 + r * 10 + col
    scalars = np.arange(n_scalars, dtype=np.float32) + tag * 100
    return {"image": image, "scalars": scalars}


def test_encode_obs_batch_is_order_preserving_and_channels_first(cfg):
    """DISTINCT per-item probes prove order-preservation + a channels-first layout.

    With unique per-item/channel/cell values, item 0 must remain row 0 (ordering)
    and image[i, c, r, col] must equal the source's [c, r, col] — i.e. the batch
    is unambiguously ``(B, C, 5, 5)`` channels-first, not channels-last.
    """
    channels = cfg["env"]["obs_channels"]
    n_scalars = cfg["env"]["obs_scalars"]
    items = [_distinct_obs(channels, n_scalars, tag=t) for t in (1, 2, 3)]
    images, scalars = encode_obs_batch(items)
    assert images.shape == (3, channels, _W, _W)
    # Order preserved: each batch row equals its source image (no reshuffle).
    for i, item in enumerate(items):
        np.testing.assert_array_equal(images[i], item["image"])
        np.testing.assert_array_equal(scalars[i], item["scalars"])
    # Channels-first probe: axis 1 indexes the channel, axis 2 the row.
    assert images[0, 0, 0, 0] == np.float32(1000)  # tag=1, c=0, r=0, col=0
    assert images[2, 1, 0, 0] == np.float32(3 * 1000 + 100)  # tag=3, c=1
    assert images[1, 0, 1, 0] == np.float32(2 * 1000 + 10)  # tag=2, r=1


def test_encode_state_shape_and_range(make_state, cfg):
    """encode_state returns the 77-float vector with all entries in [0, 1]."""
    state = make_state(cop_pos=(1, 2), thief_pos=(3, 4), barriers_used=2, step=5)
    vec = encode_state(state, cfg)
    assert vec.shape == (_STATE_DIM,)
    assert vec.dtype == np.float32
    assert float(vec.min()) >= 0.0
    assert float(vec.max()) <= 1.0


def test_encode_state_marks_positions_on_planes(make_state, cfg):
    """Cop and thief positions light up their own planes at the right cell."""
    state = make_state(cop_pos=(1, 2), thief_pos=(3, 4))
    vec = encode_state(state, cfg)
    cop_plane = vec[_COP * _PLANE : (_COP + 1) * _PLANE].reshape(_W, _W)
    thief_plane = vec[_THIEF * _PLANE : (_THIEF + 1) * _PLANE].reshape(_W, _W)
    assert cop_plane[1, 2] == 1.0
    assert thief_plane[3, 4] == 1.0
    assert cop_plane.sum() == 1.0
    assert thief_plane.sum() == 1.0


def test_encode_state_marks_two_cops_on_cop_plane(make_state, cfg):
    """With two cops (4x4 stage) BOTH light up the single cop plane (B5)."""
    state = make_state(cop_pos=[(0, 0), (2, 3)], thief_pos=(3, 4))
    vec = encode_state(state, cfg)
    cop_plane = vec[_COP * _PLANE : (_COP + 1) * _PLANE].reshape(_W, _W)
    assert cop_plane[0, 0] == 1.0
    assert cop_plane[2, 3] == 1.0
    assert cop_plane.sum() == 2.0  # two distinct cells set


def test_encode_state_colocated_cops_collapse_to_one(make_state, cfg):
    """Two co-located cops collapse to a single 1.0 (idempotent set; P4 note, B5).

    The P3 cop plane is a binary occupancy mask, so two cops on the SAME cell sum
    to 1.0, not 2.0 — acceptable for P3. A P4 multi-cop centralized state may need
    a per-cop or count plane to disambiguate; see the encode_state docstring.
    """
    state = make_state(cop_pos=[(1, 1), (1, 1)], thief_pos=(3, 4))
    vec = encode_state(state, cfg)
    cop_plane = vec[_COP * _PLANE : (_COP + 1) * _PLANE].reshape(_W, _W)
    assert cop_plane[1, 1] == 1.0
    assert cop_plane.sum() == 1.0  # collapsed, NOT 2.0


def test_encode_state_barrier_shows_on_barrier_plane(make_state, cfg):
    """A placed barrier lights up the barrier plane at its cell."""
    state = make_state(cop_pos=(0, 0), thief_pos=(4, 4), barriers=[(2, 2)])
    vec = encode_state(state, cfg)
    barrier_plane = vec[_BARRIER * _PLANE : (_BARRIER + 1) * _PLANE].reshape(_W, _W)
    assert barrier_plane[2, 2] == 1.0
    assert barrier_plane.sum() == 1.0


def test_encode_state_scalars_tail(make_state, cfg):
    """The two trailing scalars are step_norm and barriers_left_norm in [0, 1]."""
    max_moves = cfg["game"]["max_moves"]
    max_barriers = cfg["game"]["max_barriers"]
    state = make_state(cop_pos=(0, 0), thief_pos=(4, 4), barriers_used=2, step=5)
    vec = encode_state(state, cfg)
    assert vec[-2] == np.float32(5 / max_moves)
    assert vec[-1] == np.float32((max_barriers - 2) / max_barriers)


def test_encode_state_pads_small_grid_within_5x5(make_state, cfg):
    """On a 2x2 board positions stay within the 5x5 plane (no out-of-range cell)."""
    state = make_state(cop_pos=(0, 0), thief_pos=(1, 1), h=2, w=2)
    vec = encode_state(state, cfg)
    cop_plane = vec[_COP * _PLANE : (_COP + 1) * _PLANE].reshape(_W, _W)
    thief_plane = vec[_THIEF * _PLANE : (_THIEF + 1) * _PLANE].reshape(_W, _W)
    assert cop_plane[0, 0] == 1.0
    assert thief_plane[1, 1] == 1.0
