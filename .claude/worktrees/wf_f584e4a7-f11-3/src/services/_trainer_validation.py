"""§16 validation guards for :class:`~src.services.trainer.SelfPlayTrainer`.

Split out of trainer.py to keep it under the 150-LOC gate (same convention as
``src/sdk/_validation.py``).

Input: the loaded config + the trainer's int-valued public arguments.
Output: the validated config (returned so ``__init__`` binds it in one statement).
Setup: none — pure functions, no I/O.
"""

from __future__ import annotations

# section -> the positive-int keys this trainer reads out of it.
_REQUIRED_KNOBS = {
    "algo": ("batch_episodes",),
    "selfplay": ("episodes_per_round", "update_ratio", "window_k", "rounds"),
    "replay": ("min_replay_episodes",),
    "compute": ("num_threads",),
}


def _validate_config(cfg: dict) -> dict:
    """Return ``cfg`` after checking every knob the self-play loop reads.

    Raises:
        ValueError: If a required section/key is missing or is not a positive int
            (``replay.min_replay_episodes`` may be 0 — no warmup).
    """
    if "env" not in cfg or "actions" not in cfg.get("env", {}):
        raise ValueError("config is missing required section env.actions")
    for section, keys in _REQUIRED_KNOBS.items():
        if section not in cfg:
            raise ValueError(f"config is missing required section {section!r}")
        for key in keys:
            value = cfg[section].get(key)
            floor = 0 if key == "min_replay_episodes" else 1
            if not isinstance(value, int) or isinstance(value, bool) or value < floor:
                raise ValueError(f"config {section}.{key} must be an int >= {floor}, got {value!r}")
    return cfg


def _validate_input(**ints: object) -> None:
    """Type-check the trainer's int-valued public arguments (``None`` = use the default).

    Raises:
        TypeError: If a named argument is neither ``None`` nor an ``int``.
    """
    for name, value in ints.items():
        if value is None or (isinstance(value, int) and not isinstance(value, bool)):
            continue
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
