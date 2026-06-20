"""Unit tests for src.marl.data.bc_dataset (T3.2 npz IO).

Pins the shapes of a built (obs, scalars, action) dataset, manifest provenance
fields, seeded reproducibility, and the npz save/load round-trip equality.
"""

from __future__ import annotations

from random import Random

import numpy as np
import pytest

from src.marl.data.bc_dataset import build_bc_dataset, load_npz, save_npz
from src.marl.data.heuristics import cop_expert, thief_expert
from src.marl.data.schemas import DatasetManifest
from src.marl.env.cops_robbers_env import CopsRobbersEnv


def test_build_yields_requested_n_pairs_shapes(cfg):
    """build_bc_dataset returns (n_pairs, C, 5, 5) obs + (n_pairs, scalars) + (n_pairs,)."""
    n_pairs = 12
    obs, scalars, actions, _manifest = build_bc_dataset(cfg, (3, 3), n_pairs, seed=7)
    channels = cfg["env"]["obs_channels"]
    n_scalars = cfg["env"]["obs_scalars"]
    w_v = 2 * cfg["env"]["view_radius_max"] + 1
    assert obs.shape == (n_pairs, channels, w_v, w_v)
    assert scalars.shape == (n_pairs, n_scalars)
    assert actions.shape == (n_pairs,)
    assert obs.dtype == np.float32
    assert scalars.dtype == np.float32


def test_actions_are_valid_indices(cfg):
    """Every label is a non-negative action index below the cop action count."""
    _obs, _sc, actions, _m = build_bc_dataset(cfg, (3, 3), 16, seed=1)
    assert actions.min() >= 0
    assert actions.max() < cfg["env"]["actions"]["a_cop"]


def test_manifest_fields(cfg):
    """The manifest carries grid/n_pairs/seed/source + schema dims from config."""
    n_pairs = 8
    _obs, _sc, _a, manifest = build_bc_dataset(cfg, (3, 3), n_pairs, seed=42)
    assert isinstance(manifest, DatasetManifest)
    assert manifest.grid == (3, 3)
    assert manifest.n_pairs == n_pairs
    assert manifest.seed == 42
    assert manifest.source == "expert"
    assert manifest.obs_channels == cfg["env"]["obs_channels"]
    assert manifest.obs_scalars == cfg["env"]["obs_scalars"]
    assert manifest.w_v == 2 * cfg["env"]["view_radius_max"] + 1


def test_build_is_seeded_reproducible(cfg):
    """The same seed reproduces identical obs/scalars/actions arrays."""
    a = build_bc_dataset(cfg, (3, 3), 20, seed=99)
    b = build_bc_dataset(cfg, (3, 3), 20, seed=99)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(a[2], b[2])


def _replay_greedy_labels(cfg, grid, n_pairs, seed):
    """Mirror bc_dataset._collect_pairs's exact RNG order, recomputing labels.

    Reproduces the builder's seeded trajectory (same Random(seed), same env-seed
    draws, same epsilon-diversified joint actions) and returns the GREEDY
    (epsilon=0) cop expert action for every visited state, in collection order.
    """
    rng = Random(seed)
    epsilon = float(cfg["bc"]["epsilon"])
    h, w = grid
    env = CopsRobbersEnv(cfg, h=h, w=w, num_cops=1)
    labels: list[int] = []
    while len(labels) < n_pairs:
        _obs, _info = env.reset(seed=rng.randrange(2**31))
        terminated = False
        while not terminated and len(labels) < n_pairs:
            state = env.state()
            labels.append(int(cop_expert(state, cfg, idx=0)))  # greedy, no rng
            joint = {
                "cop_0": cop_expert(state, cfg, idx=0, rng=rng, epsilon=epsilon),
                "thief": thief_expert(state, cfg, rng=rng, epsilon=epsilon),
            }
            _obs, _r, terminated, _info = env.step(joint)
    return labels


def test_label_is_greedy_expert_action_not_collection_epsilon(cfg):
    """The recorded LABEL is the greedy (epsilon=0) cop action, not bc.epsilon (B3).

    Collection diversifies with a NON-ZERO ``bc.epsilon``, but every stored label
    must equal the epsilon=0 cop_expert action for the visited state. We replay
    the exact seeded trajectory and require an element-wise match — had the
    builder recorded the epsilon-greedy (collection) action instead, the noisy
    explore steps would diverge and this assertion would fail (mutation-proof).
    """
    cfg = dict(cfg)
    cfg["bc"] = dict(cfg["bc"], epsilon=0.5)  # heavy collection noise
    grid, n_pairs = (3, 3), 40
    _obs, _sc, actions, _m = build_bc_dataset(cfg, grid, n_pairs, seed=5)
    expected = _replay_greedy_labels(cfg, grid, n_pairs, seed=5)
    assert [int(a) for a in actions] == expected


def test_build_empty_n_pairs_raises_value_error(cfg):
    """n_pairs=0 raises a clear ValueError, not np.stack's empty-array leak (B4)."""
    with pytest.raises(ValueError, match="n_pairs"):
        build_bc_dataset(cfg, (3, 3), 0, seed=1)


def test_build_negative_n_pairs_raises_value_error(cfg):
    """A negative n_pairs is rejected up front with a clear ValueError (B4)."""
    with pytest.raises(ValueError, match="n_pairs"):
        build_bc_dataset(cfg, (3, 3), -1, seed=1)


def test_npz_round_trip_equal(cfg, tmp_path):
    """save_npz then load_npz returns arrays + a manifest equal to the originals."""
    obs, scalars, actions, manifest = build_bc_dataset(cfg, (3, 3), 10, seed=3)
    path = tmp_path / "bc.npz"
    save_npz(path, obs, scalars, actions, manifest)
    r_obs, r_sc, r_actions, r_manifest = load_npz(path)
    np.testing.assert_array_equal(obs, r_obs)
    np.testing.assert_array_equal(scalars, r_sc)
    np.testing.assert_array_equal(actions, r_actions)
    assert r_manifest == manifest
