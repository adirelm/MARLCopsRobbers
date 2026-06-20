"""``.npz`` persistence for the seeded BC datasets (T3.2 / T4.4).

A single archive carries the role records (``obs`` / ``scalars`` / ``actions``),
the per-record ``episode_ids`` (so an episode-level split survives a reload), and
the flattened :class:`~src.marl.data.schemas.DatasetManifest` provenance fields.
Split out of ``bc_dataset`` to keep each module under the 150-line gate (DRY:
the builder re-exports these two functions).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.marl.data.schemas import DatasetManifest


def save_npz(  # noqa: PLR0913 — one positional per dataset column is intentional
    path: str | Path,
    obs: np.ndarray,
    scalars: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    manifest: DatasetManifest,
) -> None:
    """Persist a BC dataset + episode ids + manifest to one ``.npz`` archive.

    Args:
        path: Destination ``.npz`` path (parent dirs are created).
        obs: The ``(n, C, 5, 5)`` float32 observation images.
        scalars: The ``(n, obs_scalars)`` float32 scalar hooks.
        actions: The ``(n,)`` int64 expert action labels.
        episode_ids: The ``(n,)`` int64 per-record episode ids.
        manifest: The provenance :class:`DatasetManifest`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        obs=obs,
        scalars=scalars,
        actions=actions,
        episode_ids=episode_ids,
        grid=np.asarray(manifest.grid, dtype=np.int64),
        n_pairs=np.int64(manifest.n_pairs),
        seed=np.int64(manifest.seed),
        source=manifest.source,
        obs_channels=np.int64(manifest.obs_channels),
        obs_scalars=np.int64(manifest.obs_scalars),
        w_v=np.int64(manifest.w_v),
        schema_version=manifest.schema_version,
    )


def load_npz(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DatasetManifest]:
    """Load a BC dataset + episode ids + manifest saved by :func:`save_npz`.

    Args:
        path: The ``.npz`` archive path.

    Returns:
        ``(obs, scalars, actions, episode_ids, manifest)`` reconstructed from the
        archive, equal to the originals passed to :func:`save_npz`.
    """
    with np.load(Path(path), allow_pickle=False) as data:
        obs = data["obs"]
        scalars = data["scalars"]
        actions = data["actions"]
        episode_ids = data["episode_ids"]
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
    return obs, scalars, actions, episode_ids, manifest
