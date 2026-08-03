"""Seeded role-parameterized behavior-cloning dataset builder (T3.2 / T4.4).

Generates ``n_pairs`` supervised ``(local_obs, expert_action)`` records for the
chosen ROLE (``"cop"`` -> ``obs["cop_0"]`` + :func:`cop_expert`; ``"thief"`` ->
``obs["thief"]`` + :func:`thief_expert`) by rolling the Manhattan-heuristic
experts through the P2 :class:`~src.marl.env.cops_robbers_env.CopsRobbersEnv`.
Collection is epsilon-diversified (``bc.epsilon``) so visited states vary, while
the recorded LABEL is always the role's GREEDY (``epsilon=0``) expert action.
The pipeline is seeded and each record carries its EPISODE id so
:func:`episode_split` can hold out WHOLE episodes (never adjacent records).

Privileged-expert vs local-obs imitation gap (B3): the LABEL comes from a
PRIVILEGED expert reading the full :class:`~src.marl.env.types.GlobalState`
while the stored INPUT is the role's LOCAL
:class:`~src.marl.env.types.Observation`; beyond the view radius the local input
cannot determine the label, capping val-acc below 1.0 (the per-grid Bayes
ceilings in docs/ANALYSIS.md §0 justify ``bc.val_acc_gate_by_grid``). Stacking
reuses :func:`src.marl.data.obs_encoder.encode_obs_batch` (DRY); no geometry.
"""

from __future__ import annotations

from random import Random

import numpy as np

from src.marl.data.bc_npz import load_npz, save_npz
from src.marl.data.bc_split import episode_split
from src.marl.data.heuristics import cop_expert, thief_expert
from src.marl.data.obs_encoder import encode_obs_batch
from src.marl.data.schemas import DatasetManifest, SourceTag
from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.marl.env.types import GlobalState, Observation

__all__ = ["build_bc_dataset", "episode_split", "load_npz", "save_npz"]

_SCHEMA_VERSION = "p3-bc-v1"

# Per-role (obs key, greedy expert) selection — the single role-parameter switch.
_ROLE_OBS_KEY = {"cop": "cop_0", "thief": "thief"}


def _greedy_label(state: GlobalState, cfg: dict, role: str) -> int:
    """Return the role's greedy (epsilon=0) expert action LABEL for ``state``."""
    if role == "cop":
        return int(cop_expert(state, cfg, idx=0))
    return int(thief_expert(state, cfg))


def _joint_action(state: GlobalState, cfg: dict, rng, epsilon: float) -> dict:
    """Return the epsilon-diversified joint action driving the collection roll."""
    return {
        "cop_0": cop_expert(state, cfg, idx=0, rng=rng, epsilon=epsilon),
        "thief": thief_expert(state, cfg, rng=rng, epsilon=epsilon),
    }


def _collect_pairs(
    cfg: dict, grid: tuple[int, int], n_pairs: int, seed: int, role: str
) -> tuple[list[Observation], list[int], list[int]]:
    """Roll experts through the env until ``n_pairs`` role records are collected.

    Args:
        cfg: The loaded config (reads ``bc.epsilon`` for collection diversity).
        grid: The ``(h, w)`` board size to generate on.
        n_pairs: Number of (obs, expert-action) role records to collect.
        seed: The master seed driving episode seeds + the epsilon RNG.
        role: ``"cop"`` or ``"thief"`` — selects the recorded obs key + expert.

    Returns:
        A ``(obs_list, action_list, episode_ids)`` triple: the role's local
        observations, greedy role-expert LABELS, and the 0-based episode id of
        each record (contiguous, non-decreasing runs).
    """
    rng = Random(seed)
    epsilon = float(cfg["bc"]["epsilon"])
    obs_key = _ROLE_OBS_KEY[role]
    h, w = grid
    env = CopsRobbersEnv(cfg, h=h, w=w, num_cops=1)
    obs_list: list[Observation] = []
    actions: list[int] = []
    episode_ids: list[int] = []
    episode = 0
    while len(obs_list) < n_pairs:
        obs, _info = env.reset(seed=rng.randrange(2**31))
        terminated = False
        while not terminated and len(obs_list) < n_pairs:
            state = env.state()
            obs_list.append(obs[obs_key])
            actions.append(_greedy_label(state, cfg, role))
            episode_ids.append(episode)
            obs, _r, terminated, _info = env.step(_joint_action(state, cfg, rng, epsilon))
        episode += 1
    return obs_list, actions, episode_ids


def build_bc_dataset(
    cfg: dict, grid: tuple[int, int], n_pairs: int, seed: int, role: str = "cop"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DatasetManifest]:
    """Build a seeded BC dataset of role ``(obs, scalars, action, episode)`` records.

    Args:
        cfg: The loaded config (reads ``env.*`` schema dims + ``bc.epsilon``).
        grid: The ``(h, w)`` board size to generate on.
        n_pairs: Number of supervised records to produce.
        seed: The reproducibility seed.
        role: ``"cop"`` (default, backward-compatible) or ``"thief"``.

    Returns:
        ``(obs, scalars, actions, episode_ids, manifest)`` — ``obs`` ``(n,C,5,5)``
        f32, ``scalars`` ``(n,obs_scalars)`` f32, ``actions`` / ``episode_ids``
        ``(n,)`` i64, and the :class:`DatasetManifest`.

    Raises:
        ValueError: If ``n_pairs`` is not positive, or ``role`` is unknown.
    """
    if n_pairs <= 0:
        raise ValueError(f"n_pairs must be a positive integer, got {n_pairs}")
    if role not in _ROLE_OBS_KEY:
        raise ValueError(f"unknown role {role!r}: expected 'cop' or 'thief'")
    obs_list, action_list, episode_list = _collect_pairs(cfg, grid, n_pairs, seed, role)
    obs, scalars = encode_obs_batch(obs_list)
    actions = np.asarray(action_list, dtype=np.int64)
    episode_ids = np.asarray(episode_list, dtype=np.int64)
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
    return obs, scalars, actions, episode_ids, manifest
