"""Seeded behavior-cloning dataset builder + npz IO (T3.2).

Generates ``n_pairs`` supervised ``(local_obs, expert_action)`` records for the
COP by rolling out the Manhattan-heuristic experts (cop=oracle, thief=oracle)
through the P2 :class:`~src.marl.env.cops_robbers_env.CopsRobbersEnv`. Collection
is epsilon-diversified (``bc.epsilon``) so the visited states vary, while the
recorded LABEL is always the greedy (``epsilon=0``) cop expert action. The whole
pipeline is seeded for reproducibility and round-trips through ``np.savez``.

Privileged-expert vs local-obs imitation gap (B3): the LABEL comes from a
PRIVILEGED expert reading the full :class:`~src.marl.env.types.GlobalState` (it
always knows the thief's cell), while the stored INPUT is the cop's LOCAL
:class:`~src.marl.env.types.Observation`. When the thief is beyond the view
radius the local input cannot, in principle, determine the privileged label — an
inherent BC realizability gap, partially mitigated by the aliasing-memory scalars
(``other_seen_now`` / ``steps_since_seen_norm``) and revisited under the
recurrent CTDE learner in P4/T4.4 (the BC val-acc gate lives there, NOT at P3).

Stacking reuses :func:`src.marl.data.obs_encoder.encode_obs_batch` (DRY over the
env encoder); no geometry is re-implemented here.
"""

from __future__ import annotations

from pathlib import Path
from random import Random

import numpy as np

from src.marl.data.heuristics import cop_expert, thief_expert
from src.marl.data.obs_encoder import encode_obs_batch
from src.marl.data.schemas import DatasetManifest, SourceTag
from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.marl.env.types import Observation

_SCHEMA_VERSION = "p3-bc-v1"


def _collect_pairs(
    cfg: dict, grid: tuple[int, int], n_pairs: int, seed: int
) -> tuple[list[Observation], list[int]]:
    """Roll experts through the env until ``n_pairs`` cop records are collected.

    Args:
        cfg: The loaded config (reads ``bc.epsilon`` for collection diversity).
        grid: The ``(h, w)`` board size to generate on.
        n_pairs: Number of (obs, expert-action) cop records to collect.
        seed: The master seed driving episode seeds + the epsilon RNG.

    Returns:
        A ``(obs_list, action_list)`` pair of the cop's local observations and
        the greedy cop expert action LABELS, length ``n_pairs`` each.
    """
    rng = Random(seed)
    epsilon = float(cfg["bc"]["epsilon"])
    h, w = grid
    env = CopsRobbersEnv(cfg, h=h, w=w, num_cops=1)
    obs_list: list[Observation] = []
    actions: list[int] = []
    while len(obs_list) < n_pairs:
        obs, _info = env.reset(seed=rng.randrange(2**31))
        terminated = False
        while not terminated and len(obs_list) < n_pairs:
            state = env.state()
            label = cop_expert(state, cfg, idx=0)  # greedy LABEL (epsilon=0)
            obs_list.append(obs["cop_0"])
            actions.append(int(label))
            joint = {
                "cop_0": cop_expert(state, cfg, idx=0, rng=rng, epsilon=epsilon),
                "thief": thief_expert(state, cfg, rng=rng, epsilon=epsilon),
            }
            obs, _r, terminated, _info = env.step(joint)
    return obs_list, actions


def build_bc_dataset(
    cfg: dict, grid: tuple[int, int], n_pairs: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, DatasetManifest]:
    """Build a seeded BC dataset of ``(obs, scalars, action)`` cop records.

    Args:
        cfg: The loaded config (reads ``env.*`` schema dims + ``bc.epsilon``).
        grid: The ``(h, w)`` board size to generate on.
        n_pairs: Number of supervised records to produce.
        seed: The reproducibility seed.

    Returns:
        ``(obs, scalars, actions, manifest)``: ``obs`` ``(n_pairs, C, 5, 5)`` f32,
        ``scalars`` ``(n_pairs, obs_scalars)`` f32, ``actions`` ``(n_pairs,)`` i64,
        and the :class:`DatasetManifest` provenance record.

    Raises:
        ValueError: If ``n_pairs`` is not a positive integer (an empty dataset
            would otherwise leak ``np.stack``'s "need at least one array" error).
    """
    if n_pairs <= 0:
        raise ValueError(f"n_pairs must be a positive integer, got {n_pairs}")
    obs_list, action_list = _collect_pairs(cfg, grid, n_pairs, seed)
    obs, scalars = encode_obs_batch(obs_list)
    actions = np.asarray(action_list, dtype=np.int64)
    env_cfg = cfg["env"]
    manifest = DatasetManifest(
        grid=(grid[0], grid[1]),
        n_pairs=n_pairs,
        seed=seed,
        source=str(SourceTag.EXPERT),
        obs_channels=int(env_cfg["obs_channels"]),
        obs_scalars=int(env_cfg["obs_scalars"]),
        w_v=2 * int(env_cfg["view_radius_max"]) + 1,
        schema_version=_SCHEMA_VERSION,
    )
    return obs, scalars, actions, manifest


def save_npz(
    path: str | Path,
    obs: np.ndarray,
    scalars: np.ndarray,
    actions: np.ndarray,
    manifest: DatasetManifest,
) -> None:
    """Persist a BC dataset + manifest to a single ``.npz`` archive.

    Args:
        path: Destination ``.npz`` path (parent dirs are created).
        obs: The ``(n_pairs, C, 5, 5)`` float32 observation images.
        scalars: The ``(n_pairs, obs_scalars)`` float32 scalar hooks.
        actions: The ``(n_pairs,)`` int64 expert action labels.
        manifest: The provenance :class:`DatasetManifest`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        obs=obs,
        scalars=scalars,
        actions=actions,
        grid=np.asarray(manifest.grid, dtype=np.int64),
        n_pairs=np.int64(manifest.n_pairs),
        seed=np.int64(manifest.seed),
        source=manifest.source,
        obs_channels=np.int64(manifest.obs_channels),
        obs_scalars=np.int64(manifest.obs_scalars),
        w_v=np.int64(manifest.w_v),
        schema_version=manifest.schema_version,
    )


def load_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, DatasetManifest]:
    """Load a BC dataset + manifest saved by :func:`save_npz`.

    Args:
        path: The ``.npz`` archive path.

    Returns:
        ``(obs, scalars, actions, manifest)`` reconstructed from the archive,
        equal to the originals passed to :func:`save_npz`.
    """
    with np.load(Path(path), allow_pickle=False) as data:
        obs = data["obs"]
        scalars = data["scalars"]
        actions = data["actions"]
        grid = tuple(int(x) for x in data["grid"])
        manifest = DatasetManifest(
            grid=(grid[0], grid[1]),
            n_pairs=int(data["n_pairs"]),
            seed=int(data["seed"]),
            source=str(data["source"]),
            obs_channels=int(data["obs_channels"]),
            obs_scalars=int(data["obs_scalars"]),
            w_v=int(data["w_v"]),
            schema_version=str(data["schema_version"]),
        )
    return obs, scalars, actions, manifest
