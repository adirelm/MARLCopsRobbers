"""Shared helpers for the §9.3 wire-replay tests — committed rehearsal artifacts only.

The tests run against the COMMITTED dress-rehearsal log + records (skip-if-absent, the
project's artifact-test convention: never assert a git-ignored/optional artifact exists).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.gui.spectator import SpectatorFrame
from src.utils.config_loader import load_config

_ROOT = Path(__file__).resolve().parents[2]


def rehearsal_paths() -> tuple[Path, Path]:
    """Return (the committed log matching the rehearsal records, those records); skip when absent.

    Selected by MATCHING the records, not by taking the newest file. The newest-wins version
    silently swapped to the real §9 match log the moment that match was played, then replayed
    it against the rehearsal records — six tests failed for a reason that had nothing to do
    with what they assert. A log and its records are one artifact pair; pick them as a pair.
    """
    cfg = load_config()
    records = _ROOT / cfg["wire_match"]["rehearsal"]["records"]
    logs = sorted((_ROOT / cfg["wire_match"]["log_dir"]).glob("wire_log_[0-9]*.jsonl"))
    if not logs or not records.exists():
        pytest.skip("committed rehearsal wire log / records not present")
    want = [(g["id"], g["winner"], g["moves"]) for g in json.loads(records.read_text("utf-8"))["sub_games"]]
    for log in logs:
        results = [
            json.loads(line)["sub_game"]
            for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line).get("direction") == "result"
        ]
        if [(r["id"], r["winner"], r["moves"]) for r in results] == want:
            return log, records
    pytest.skip("no committed wire log matches the rehearsal records")


REHEARSAL_SEEDS = [101, 202, 303, 404, 505, 606]  # the list the committed rehearsal log was played under


def rehearsal_cfg() -> dict:
    """Config pinned to the seeds the COMMITTED rehearsal log was recorded under.

    The live ``wire_match.seeds`` tracks whatever match is current — after the real §9 match
    against biu-azri it is their frozen list. A test that replays a FIXED artifact must not
    depend on that: the artifact's seeds are a property of the artifact, not of today's config.

    The GROUP NAMES are pinned for the same reason. The replay now binds §9.1 role alternation
    to ``wire_match.groups.*.name`` — the frozen agreement — instead of trusting the body's own
    ``groups`` block, and the rehearsal body predates the real partner, so it still carries the
    ``<PARTNER GROUP CODE>`` placeholder. Pinning the fixture's config to the artifact keeps
    that check ON for the rehearsal instead of loosening it for everyone.
    """
    cfg = load_config()
    cfg["wire_match"] = {
        **cfg["wire_match"],
        "seeds": list(REHEARSAL_SEEDS),
        "groups": {
            "group_1": {**cfg["wire_match"]["groups"]["group_1"], "name": "adrl-001"},
            "group_2": {**cfg["wire_match"]["groups"]["group_2"], "name": "<PARTNER GROUP CODE>"},
        },
    }
    return cfg


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
