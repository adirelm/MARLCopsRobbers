"""Tests for the §3.5 MatchReport assembly + validation (T6.3).

The report DERIVES totals from the per-sub-game scores and validates against
docs/schema/report.schema.json. Student PII is a PLACEHOLDER only (never real).
"""

from __future__ import annotations

import pytest

from src.reporting.schema import Student, build_report, validate

_RESULTS = [
    {
        "game_id": "sg-0",
        "grid": [5, 5],
        "winner": "cop",
        "capture": True,
        "steps": 3,
        "scores": {"cop": 20, "thief": 5},
        "seed": 7,
    },
    {
        "game_id": "sg-1",
        "grid": [5, 5],
        "winner": "thief",
        "capture": False,
        "steps": 25,
        "scores": {"cop": 5, "thief": 10},
        "seed": 17,
    },
]
_STUDENTS = [Student("Placeholder Student", "000000000")]  # PII PLACEHOLDER only


def test_build_report_derives_totals_and_validates():
    """A built report has derived totals + num_games and validates vs the §3.5 schema."""
    body = build_report("adrl-001", _STUDENTS, _RESULTS).to_dict()
    assert body["num_games"] == 2
    assert body["totals"] == {"cop": 25, "thief": 15}
    validate(body)


def test_totals_equal_sum_of_sub_game_scores():
    """totals == Σ per-sub-game scores (the §3.5 / P6 exit invariant)."""
    body = build_report("adrl-001", _STUDENTS, _RESULTS).to_dict()
    assert body["totals"]["cop"] == sum(g["scores"]["cop"] for g in body["sub_games"])
    assert body["totals"]["thief"] == sum(g["scores"]["thief"] for g in body["sub_games"])


def test_validate_rejects_bad_winner_enum():
    """An out-of-enum winner is rejected."""
    body = build_report("adrl-001", _STUDENTS, _RESULTS).to_dict()
    body["sub_games"][0]["winner"] = "burglar"
    with pytest.raises(ValueError, match="enum"):
        validate(body)


def test_validate_rejects_missing_required_key():
    """A body missing a required top-level key is rejected."""
    with pytest.raises(ValueError, match="required"):
        validate({"group": "adrl-001"})


def test_validate_rejects_extra_top_level_key():
    """An unexpected top-level key (additionalProperties:false) is rejected."""
    body = build_report("adrl-001", _STUDENTS, _RESULTS).to_dict()
    body["leak"] = 1
    with pytest.raises(ValueError, match="unexpected"):
        validate(body)


def test_validate_rejects_inconsistent_num_games():
    """A schema-valid body whose num_games lies about sub_games is rejected (semantic)."""
    body = build_report("adrl-001", _STUDENTS, _RESULTS).to_dict()
    body["num_games"] = 999
    with pytest.raises(ValueError, match="num_games"):
        validate(body)


def test_validate_rejects_totals_not_equal_to_sum():
    """A body whose totals disagree with Σ sub-game scores is rejected (totals are derived)."""
    body = build_report("adrl-001", _STUDENTS, _RESULTS).to_dict()
    body["totals"]["cop"] = 0
    with pytest.raises(ValueError, match="totals"):
        validate(body)
