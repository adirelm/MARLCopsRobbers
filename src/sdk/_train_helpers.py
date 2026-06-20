"""Training-entry wiring for ``MarlSDK.train`` (keeps sdk.py / _helpers.py <=150).

Maps the public algorithm name to the config knobs the self-play trainer reads,
and resolves a curriculum stage index to its ``(h, w, num_cops)``. ``qmix`` is the
graded PRIMARY (CopLearner + QmixMixer), ``vdn`` the ablation arm (CopLearner +
VdnMixer), and ``iql`` the §7.2 baseline (IqlLearner, no mixer). All values are
config-derived; the algorithm switch only flips ``algo.name`` / ``algo.mixer.type``.
"""

from __future__ import annotations

import copy

# Public algorithm -> (algo.name, algo.mixer.type). vdn keeps algo.name "qmix" so
# CopLearner is used but with the VDN mixer; iql routes to the no-mixer baseline.
_ALGO_SPECS: dict[str, tuple[str, str]] = {
    "qmix": ("qmix", "qmix"),
    "vdn": ("qmix", "vdn"),
    "iql": ("iql", "qmix"),
}


def cfg_for_algo(cfg: dict, algorithm: str) -> dict:
    """Return a deep-copied config with ``algo.name`` / ``algo.mixer.type`` set.

    Args:
        cfg: The loaded base config (never mutated — a deep copy is returned).
        algorithm: ``"qmix"`` | ``"vdn"`` | ``"iql"``.

    Returns:
        A new config selecting the requested algorithm.

    Raises:
        ValueError: If ``algorithm`` is not a known arm.
    """
    if algorithm not in _ALGO_SPECS:
        raise ValueError(f"unknown algorithm {algorithm!r}: expected one of {sorted(_ALGO_SPECS)}")
    name, mixer = _ALGO_SPECS[algorithm]
    out = copy.deepcopy(cfg)
    out["algo"]["name"] = name
    out["algo"]["mixer"]["type"] = mixer
    return out


def stage_params(cfg: dict, stage_idx: int) -> tuple[int, int, int]:
    """Return ``(h, w, num_cops)`` for curriculum stage ``stage_idx``.

    Args:
        cfg: The loaded config (reads ``env.curriculum.stages`` /
            ``num_cops_by_stage``).
        stage_idx: Index into the curriculum ladder (0 == the 2x2 stage).

    Returns:
        The board height, width, and real cop count for the stage.
    """
    curriculum = cfg["env"]["curriculum"]
    h, w = curriculum["stages"][stage_idx]
    return int(h), int(w), int(curriculum["num_cops_by_stage"][stage_idx])
