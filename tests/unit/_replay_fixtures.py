"""Shared helpers for the §9.3 wire-replay tests — committed rehearsal artifacts only.

The tests run against the COMMITTED dress-rehearsal log + records (skip-if-absent, the
project's artifact-test convention: never assert a git-ignored/optional artifact exists).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.gui.spectator import SpectatorFrame
from src.utils.config_loader import load_config

_ROOT = Path(__file__).resolve().parents[2]


def rehearsal_paths() -> tuple[Path, Path]:
    """Return (newest timestamped committed wire log, rehearsal records); skip when absent."""
    cfg = load_config()
    logs = sorted((_ROOT / cfg["wire_match"]["log_dir"]).glob("wire_log_[0-9]*.jsonl"))
    records = _ROOT / cfg["wire_match"]["rehearsal"]["records"]
    if not logs or not records.exists():
        pytest.skip("committed rehearsal wire log / records not present")
    return logs[-1], records


def frame(**over):
    """Build a minimal SpectatorFrame for mid-frame-pick tests (override fields as needed)."""
    base = {
        "grid": (5, 5),
        "cop_positions": ((0, 0),),
        "thief_position": (4, 4),
        "barriers": (),
        "view_radius": 2,
        "move": 0,
        "max_moves": 25,
        "sub_game": 1,
        "num_games": 6,
        "scores": {"cop": 0, "thief": 0},
        "totals": {"cop": 0, "thief": 0},
        "winner": None,
        "last_action": None,
    }
    return SpectatorFrame(**{**base, **over})
