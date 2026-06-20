"""Single-source config loader for MARL Cops & Robbers (T0.3).

Reads `config/config.yaml`, expands ${VAR} tokens from the environment, and
validates the version pin plus the required top-level sections. This is the
ONLY sanctioned entry point for algorithm-relevant parameters (CLAUDE.md §4).
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

EXPECTED_VERSION = "1.0.0"

# `version` is a scalar string; every other required section is a YAML mapping.
VERSION_KEY = "version"

# The 19 frozen top-level sections (P1 spec line 19 + the P3 `p3_smoke` block).
# Order is documentation-only.
REQUIRED_SECTIONS: tuple[str, ...] = (
    VERSION_KEY,
    "project",
    "game",
    "env",
    "algo",
    "nets",
    "olora",
    "bc",
    "replay",
    "selfplay",
    "training",
    "reward",
    "mcp",
    "cloud",
    "gmail",
    "gui",
    "paths",
    "logging",
    "p3_smoke",
)

# The mapping sections that must each parse to a (possibly empty) dict, never None.
MAPPING_SECTIONS: tuple[str, ...] = tuple(s for s in REQUIRED_SECTIONS if s != VERSION_KEY)

# config/config.yaml resolved relative to this file (package-relative, repo root).
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"

# Single-element cache box so the default-path load happens once without a
# module-level `global` rebinding (ruff PLW0603-clean).
_CACHE: dict[str, dict[str, Any]] = {}


def _interpolate_env(obj: Any) -> Any:
    """Recursively expand ${VAR} tokens in string scalars from os.environ.

    Recurses into dict values and list elements so a ${VAR} token expands
    whether it sits at a mapping value or inside a list. Non-string scalars
    (int, float, bool, None) are returned unchanged. An UNDEFINED ${VAR} is
    left as the literal token (os.path.expandvars never raises and does not
    substitute unknown names) — callers must treat a surviving ${...} as a
    deliberately unset value, not an error.

    Args:
        obj: A nested structure of dicts, lists, and scalars from yaml.

    Returns:
        The same structure with every string scalar passed through
        os.path.expandvars.
    """
    if isinstance(obj, dict):
        return {key: _interpolate_env(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_interpolate_env(item) for item in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


def _validate_config(cfg: dict) -> None:
    """Validate the version pin and required top-level sections.

    A section that is present but empty (an empty ``gui:`` YAML block parses
    to ``None``) is rejected, not silently accepted: every mapping section
    must be a dict and no required section may be ``None``.

    Args:
        cfg: The parsed, interpolated config dict.

    Raises:
        ValueError: If the config root is not a mapping, the version is not
            EXPECTED_VERSION, a required section is missing or ``None``, or a
            mapping section is not a dict.
    """
    if not isinstance(cfg, dict):
        raise ValueError(f"config root must be a mapping, got {type(cfg).__name__}")
    version = cfg.get(VERSION_KEY)
    if version != EXPECTED_VERSION:
        raise ValueError(f"config version {version!r} != required {EXPECTED_VERSION!r}")
    missing = [section for section in REQUIRED_SECTIONS if section not in cfg]
    if missing:
        raise ValueError(f"config missing required top-level section(s): {missing}")
    empty = [section for section in REQUIRED_SECTIONS if cfg.get(section) is None]
    if empty:
        raise ValueError(f"config has present-but-empty required section(s): {empty}")
    not_mapping = [s for s in MAPPING_SECTIONS if not isinstance(cfg[s], dict)]
    if not_mapping:
        raise ValueError(f"config section(s) must be a mapping: {not_mapping}")


def load_config(path: str | Path | None = None) -> dict:
    """Load, interpolate, validate, and return the config as a nested dict.

    Every call returns a fresh ``copy.deepcopy`` — including cache hits — so a
    caller that mutates the returned dict can never pollute the cached config
    or any other caller's copy. Undefined ${VAR} tokens are left as their
    literal ``${...}`` text (see :func:`_interpolate_env`); they are a valid
    "unset" value and do NOT raise.

    Args:
        path: Optional override path; defaults to config/config.yaml. When the
            default path is used the validated config is cached, but each call
            still returns an independent deep copy.

    Returns:
        A fresh, validated, env-interpolated config dictionary.

    Raises:
        ValueError: On a bad version, a missing section, or a present-but-empty
            (``None`` / non-mapping) required section.
    """
    use_default = path is None
    if use_default and "default" in _CACHE:
        return copy.deepcopy(_CACHE["default"])
    target = DEFAULT_PATH if use_default else Path(path)
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    cfg = _interpolate_env(raw or {})
    _validate_config(cfg)
    if use_default:
        _CACHE["default"] = cfg
    return copy.deepcopy(cfg)
