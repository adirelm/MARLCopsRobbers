"""RED->GREEN tests for the §9 bonus-game report assembly + validation (FR-ANL-9, built).

Placeholder identities ONLY (PII stays in git-ignored *.local.yaml files). The §9.1
role alternation (games 1-3: group_1 = cop; 4-6 swapped), the derived totals_by_group,
and the §9.2 bonus_claim (10/7 win-lose, 5/5 tie) are the semantic invariants.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.reporting.bonus import (
    build_bonus_report,
    derive_bonus_claim,
    derive_totals_by_group,
    validate_bonus,
)

_G1, _G2 = "adrl-001", "team-beta"
_STUDENTS_1 = [{"role": "A", "full_name": "Placeholder One", "id": "12345"}]
_STUDENTS_2 = [{"role": "A", "full_name": "Placeholder Two", "id": "67890"}]
_REPOS = ("https://github.com/example/ours", "https://github.com/example/theirs")


def _result(winner: str) -> dict:
    scores = {"cop": 20, "thief": 5} if winner == "cop" else {"cop": 5, "thief": 10}
    return {
        "start": "2026-07-04T18:00:00+03:00",
        "end": "2026-07-04T18:02:00+03:00",
        "moves": 17,
        "winner": winner,
        "scores": scores,
    }


def _report(winners: list[str], mutual_agreement: bool = True) -> dict:
    return build_bonus_report(
        groups=(_G1, _G2),
        repos=_REPOS,
        students=(_STUDENTS_1, _STUDENTS_2),
        timezone="Asia/Jerusalem",
        results=[_result(w) for w in winners],
        mutual_agreement=mutual_agreement,
    )


def test_role_alternation_and_kept_fields():
    """Games 1-3: group_1 is cop, 4-6 swapped; §3.5 fields (start/end/moves) are KEPT."""
    report = _report(["cop", "thief", "cop", "thief", "thief", "cop"])
    assert report["report_type"] == "bonus_game"
    for game in report["sub_games"][:3]:
        assert game["cop_group"] == _G1 and game["thief_group"] == _G2
    for game in report["sub_games"][3:]:
        assert game["cop_group"] == _G2 and game["thief_group"] == _G1
    assert all(g["start"] and g["end"] and g["moves"] == 17 for g in report["sub_games"])
    assert [g["id"] for g in report["sub_games"]] == [1, 2, 3, 4, 5, 6]


def test_totals_by_group_and_bonus_claim_win_lose():
    """We capture 2 of 3 as cop + escape 3 of 3 as thief -> we total higher -> claim 10/7."""
    report = _report(["cop", "cop", "thief", "thief", "thief", "thief"])
    # ours: 20+20+5 (cop games) + 10+10+10 (thief games) = 75; theirs: 5+5+10 + 5+5+5 = 35
    assert report["totals_by_group"] == {_G1: 75, _G2: 35}
    assert report["bonus_claim"] == {_G1: 10, _G2: 7}


def test_bonus_claim_tie_is_5_5():
    """3-3 (mirror outcomes across the swap) totals equal -> both claim 5 (§9.2)."""
    report = _report(["cop", "thief", "thief", "cop", "thief", "thief"])
    totals = report["totals_by_group"]
    assert totals[_G1] == totals[_G2]
    assert report["bonus_claim"] == {_G1: 5, _G2: 5}


def test_validate_bonus_accepts_the_built_report():
    validate_bonus(_report(["cop", "thief", "cop", "thief", "thief", "cop"]))


def test_validate_bonus_rejects_wrong_alternation():
    """A sub-game whose cop_group violates the §9.1 1-3/4-6 alternation raises."""
    report = _report(["cop"] * 6)
    report["sub_games"][0]["cop_group"] = _G2
    report["sub_games"][0]["thief_group"] = _G1
    with pytest.raises(ValueError, match="alternation"):
        validate_bonus(report)


def test_validate_bonus_rejects_tampered_totals_and_claim():
    """totals_by_group and bonus_claim are DERIVED, never trusted."""
    report = _report(["cop"] * 6)
    report["totals_by_group"][_G1] += 5
    with pytest.raises(ValueError, match="totals_by_group"):
        validate_bonus(report)
    report = _report(["cop"] * 6)
    report["bonus_claim"] = {_G1: 10, _G2: 10}
    with pytest.raises(ValueError, match="bonus_claim"):
        validate_bonus(report)


def test_validate_bonus_requires_six_games_and_schema_shape():
    """5 results refuse to build (§9.1: exactly 6); a missing required key fails the schema."""
    with pytest.raises(ValueError, match="6"):
        _report(["cop"] * 5)
    report = _report(["cop"] * 6)
    del report["mutual_agreement"]
    with pytest.raises(ValueError):
        validate_bonus(report)


def test_build_bonus_report_rejects_non_bool_mutual_agreement():
    """A truthy NON-bool ('false'!) must never be coerced into a §9.3 agreement claim."""
    with pytest.raises(TypeError, match="mutual_agreement"):
        _report(["cop"] * 6, mutual_agreement="false")  # bool('false') is True — refuse the coercion
    with pytest.raises(TypeError, match="mutual_agreement"):
        _report(["cop"] * 6, mutual_agreement=1)


_SCORING = {"cop_win": 20, "thief_win": 10, "cop_loss": 5, "thief_loss": 5}  # §3.4 Table 1


def test_validate_bonus_rejects_scores_not_matching_table1():
    """With `scoring`, a winner-cop sub-game carrying non-Table-1 scores is fraud -> raises."""
    report = _report(["cop"] * 6)
    validate_bonus(report, scoring=_SCORING)  # honest Table-1 scores pass
    report["sub_games"][0]["scores"] = {"cop": 0, "thief": 999}
    totals = derive_totals_by_group(report["sub_games"], _G1, _G2)  # keep totals/claim consistent
    report["totals_by_group"], report["bonus_claim"] = totals, derive_bonus_claim(totals)
    validate_bonus(report)  # without scoring the fraud is invisible (draft-stage structural check)
    with pytest.raises(ValueError, match="Table-1"):
        validate_bonus(report, scoring=_SCORING)


def test_bonus_schema_bounds_reject_out_of_range_moves_and_bad_timestamps():
    """The tightened schema rejects moves outside 1..25 and non-ISO-8601 start/end."""
    report = _report(["cop"] * 6)
    report["sub_games"][0]["moves"] = 0
    with pytest.raises(ValueError, match="minimum"):
        validate_bonus(report)
    report = _report(["cop"] * 6)
    report["sub_games"][0]["start"] = "not-a-timestamp"
    with pytest.raises(ValueError, match="pattern"):
        validate_bonus(report)


def test_committed_rehearsal_draft_still_validates_after_schema_tightening():
    """F5 regression: the tracked §9.4 rehearsal placeholder passes the tightened schema."""
    path = (
        Path(__file__).resolve().parents[2] / "results" / "reports" / "bonus_draft.rehearsal.placeholder.json"
    )
    if not path.exists():
        pytest.skip("rehearsal placeholder artifact not present")  # never assert untracked artifacts
    validate_bonus(json.loads(path.read_text(encoding="utf-8")))
