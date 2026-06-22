"""Tests for the §3.5 MatchReport assembly + validation (T6.3).

The report MATCHES the BRIEF §3.5 structure (group_name / students[role,name,id] /
github_repo / timezone / sub_games[id,start,end,moves,winner,scores] / derived totals)
and validates against docs/schema/report.schema.json. Student PII is a PLACEHOLDER only.
"""

from __future__ import annotations

import pytest

from src.reporting.schema import Student, build_report, validate

_RESULTS = [
    {
        "start": "2026-06-17T18:00:05.000+03:00",
        "end": "2026-06-17T18:02:40.000+03:00",
        "moves": 3,
        "winner": "cop",
        "scores": {"cop": 20, "thief": 5},
    },
    {
        "start": "2026-06-17T18:03:10.000+03:00",
        "end": "2026-06-17T18:06:02.000+03:00",
        "moves": 25,
        "winner": "thief",
        "scores": {"cop": 5, "thief": 10},
    },
]
_STUDENTS = [Student("A", "Placeholder Student", "000000000")]  # PII PLACEHOLDER only
_REPO = "https://github.com/example/marl-cop-thief"
_TZ = "Asia/Jerusalem"


def _body() -> dict:
    return build_report("adrl-001", _STUDENTS, _REPO, _TZ, _RESULTS).to_dict()


def test_build_report_matches_brief_structure_and_validates():
    """A built report carries the brief fields + derived totals and validates vs the schema."""
    body = _body()
    assert body["group_name"] == "adrl-001" and body["github_repo"] == _REPO and body["timezone"] == _TZ
    assert body["students"] == [{"role": "A", "full_name": "Placeholder Student", "id": "000000000"}]
    assert body["totals"] == {"cop": 25, "thief": 15}
    first = body["sub_games"][0]
    assert first["id"] == 1 and first["moves"] == 3 and first["start"].endswith("+03:00")
    assert set(first) == {"id", "start", "end", "moves", "winner", "scores"}
    validate(body)


def test_totals_equal_sum_of_sub_game_scores():
    """totals == Σ per-sub-game scores (the §3.5 / P6 exit invariant)."""
    body = _body()
    assert body["totals"]["cop"] == sum(g["scores"]["cop"] for g in body["sub_games"])
    assert body["totals"]["thief"] == sum(g["scores"]["thief"] for g in body["sub_games"])


def test_validate_rejects_bad_winner_enum():
    """An out-of-enum winner is rejected."""
    body = _body()
    body["sub_games"][0]["winner"] = "burglar"
    with pytest.raises(ValueError, match="enum"):
        validate(body)


def test_validate_rejects_missing_required_key():
    """A body missing a required top-level key is rejected."""
    with pytest.raises(ValueError, match="required"):
        validate({"group_name": "adrl-001"})


def test_validate_rejects_extra_top_level_key():
    """An unexpected top-level key (additionalProperties:false) is rejected."""
    body = _body()
    body["leak"] = 1
    with pytest.raises(ValueError, match="unexpected"):
        validate(body)


def test_validate_rejects_wrong_game_count_when_expected_given():
    """validate(expected_games=6) rejects a body that doesn't have exactly 6 sub-games."""
    with pytest.raises(ValueError, match="exactly 6"):
        validate(_body(), expected_games=6)  # _body has 2 sub-games


def test_validate_rejects_totals_not_equal_to_sum():
    """A body whose totals disagree with Σ sub-game scores is rejected (totals are derived)."""
    body = _body()
    body["totals"]["cop"] = 0
    with pytest.raises(ValueError, match="totals"):
        validate(body)
