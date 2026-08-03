"""The replay log and its records must be chosen as a MATCHING PAIR.

Shipped bug this pins: the log default took the newest timestamped file while the records
default fell back to the committed rehearsal records whenever the git-ignored real draft was
absent. On any fresh clone that paired the REAL §9 match log with REHEARSAL records, so the
README's documented §9.3 reproduction command died with ReplayMismatchError before printing
a line — on a public repo a grader would hit it first thing.
"""

from __future__ import annotations

import json

import pytest

from src.mcp._replay_log import select_log_and_records
from src.utils.config_loader import load_config


@pytest.fixture
def cfg_():
    return load_config()


def _summaries(path) -> list[tuple]:
    body = json.loads(path.read_text(encoding="utf-8"))
    return [(g["id"], g["winner"], g["moves"]) for g in body["sub_games"]]


def _log_results(path) -> list[tuple]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("direction") == "result":
            g = entry["sub_game"]
            out.append((g["id"], g["winner"], g["moves"]))
    return out


def test_selected_pair_actually_describes_the_same_match(cfg_) -> None:
    """The returned log and records must agree sub-game for sub-game."""
    log, records = select_log_and_records(cfg_)
    assert _log_results(log) == _summaries(records), f"{log.name} does not match {records.name}"


def test_a_fresh_clone_resolves_to_tracked_records(cfg_, tmp_path) -> None:
    """REGRESSION: with the git-ignored draft absent, the pair must still be consistent.

    Simulates a fresh clone by pointing draft_report at a path that does not exist — the
    exact condition under which the old defaults produced a mismatched pair.
    """
    cfg = load_config()
    cfg["wire_match"]["draft_report"] = str(tmp_path / "absent.json")
    log, records = select_log_and_records(cfg)
    assert records.exists(), "fresh clone resolved to a records file that is not committed"
    assert _log_results(log) == _summaries(records)


def test_no_matching_pair_fails_loudly(cfg_, tmp_path) -> None:
    """A missing pair must exit with guidance, never replay a mismatched one."""
    cfg = load_config()
    cfg["wire_match"]["draft_report"] = str(tmp_path / "absent.json")
    cfg["wire_match"]["redacted_records"] = str(tmp_path / "absent2.json")
    cfg["wire_match"]["rehearsal"] = {"records": str(tmp_path / "absent3.json")}
    with pytest.raises(SystemExit, match="no log under"):
        select_log_and_records(cfg)
