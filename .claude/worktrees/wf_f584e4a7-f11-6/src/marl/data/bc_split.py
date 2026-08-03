"""Deterministic EPISODE-level train/val split for BC datasets (T4.4).

Splitting BY EPISODE (not by record) is the honest choice: adjacent BC records
are temporally correlated, so a per-record split leaks near-duplicate states into
validation and inflates val-acc. Keying the partition on episode id keeps every
episode's records on ONE side. Split out of ``bc_dataset`` to hold the 150-line
gate; the builder re-exports :func:`episode_split`.
"""

from __future__ import annotations

import numpy as np
from numpy.random import default_rng

# A non-empty episode-level split needs both sides occupied -> at least 2 episodes.
_MIN_EPISODES = 2


def episode_split(episode_ids: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically hold out WHOLE episodes for a BC train/val split.

    Shuffles the UNIQUE episode ids with ``bc.split_seed`` and assigns the
    leading ``round(bc.val_fraction * n_episodes)`` episodes (clamped to
    ``[1, n_episodes - 1]``) to validation, so BOTH sides are always non-empty,
    returning the record indices for each side.

    Args:
        episode_ids: The ``(n,)`` per-record episode ids from the builder.
        cfg: The loaded config (reads ``bc.val_fraction`` and ``bc.split_seed``).

    Returns:
        A ``(train_idx, val_idx)`` pair of int64 record-index arrays whose union
        is all records and whose episode sets are disjoint.

    Raises:
        ValueError: If there are fewer than 2 episodes (a split cannot keep both
            train and val non-empty — an empty train side yields NaN BC loss).
    """
    bc = cfg["bc"]
    val_fraction = float(bc["val_fraction"])
    rng = default_rng(int(bc["split_seed"]))
    unique = np.unique(episode_ids)
    if len(unique) < _MIN_EPISODES:
        raise ValueError(f"episode_split needs >=2 episodes for a non-empty split, got {len(unique)}")
    rng.shuffle(unique)
    n_val = min(len(unique) - 1, max(1, round(val_fraction * len(unique))))
    val_episodes = set(unique[:n_val].tolist())
    is_val = np.isin(episode_ids, list(val_episodes))
    all_idx = np.arange(len(episode_ids), dtype=np.int64)
    return all_idx[~is_val], all_idx[is_val]
