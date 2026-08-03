"""§16 validation guards for the SDK facade (kept out of sdk.py to hold it <=150 LOC).

Input: the loaded config dict, and the public ``MarlSDK.train`` arguments.
Output: the validated config (``_validate_config`` returns it so ``__init__`` can bind
the result in one statement); ``None`` from ``_validate_input``.
Setup: none — pure functions with no I/O, imported by :mod:`src.sdk.sdk`.
"""

from __future__ import annotations

# The config sections the facade (or the services it routes to) actually reads:
# compute -> apply_compute_limits, algo -> cfg_for_algo, env/game -> build_env,
# selfplay/training -> the trainer + sweeps, gmail/project -> send_final_report.
_REQUIRED_SECTIONS = ("compute", "algo", "env", "game", "selfplay", "training", "gmail", "project")


def _validate_config(cfg: dict) -> dict:
    """Return ``cfg`` after checking the sections/knobs the SDK reads.

    Args:
        cfg: The loaded project config.

    Returns:
        The same ``cfg`` object (so callers can bind it in one statement).

    Raises:
        ValueError: If ``cfg`` is not a dict, a required section is missing, or
            ``compute.num_threads`` is not a positive int.
    """
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a dict, got {type(cfg).__name__}")
    missing = [key for key in _REQUIRED_SECTIONS if key not in cfg]
    if missing:
        raise ValueError(f"config is missing required section(s): {missing}")
    threads = cfg["compute"].get("num_threads")
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError(f"config compute.num_threads must be a positive int, got {threads!r}")
    return cfg


def _validate_input(algorithm: object, seed: object, stage_idx: object) -> None:
    """Type-check the public ``MarlSDK.train`` arguments.

    Args:
        algorithm: Must be a ``str`` (the arm name; membership is checked downstream
            by :func:`src.sdk._train_helpers.cfg_for_algo`, which raises ValueError).
        seed: Must be an ``int``.
        stage_idx: Must be an ``int``.

    Raises:
        TypeError: If any argument has the wrong type.
    """
    if not isinstance(algorithm, str):
        raise TypeError(f"algorithm must be a str, got {type(algorithm).__name__}")
    for name, value in (("seed", seed), ("stage_idx", stage_idx)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
