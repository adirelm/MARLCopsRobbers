"""The §9.4 body's IDENTITY and OUTCOME fields — the second layer of records binding.

Split from ``test_replay_records_bind.py`` at the 150-LOC cap. That file covers the match
SHAPE (id set, role alternation, Table-1 scores); this one covers the fields a second
adversarial pass showed could still be edited freely: the ``groups`` block itself, and the
published ``totals_by_group`` / ``bonus_claim`` that the whole bonus is about.

Every test here reproduces an artifact that passed a full ``replay_match`` before the fix.
"""

from __future__ import annotations

import pytest

from src.mcp._replay_log import ReplayMismatchError
from src.mcp.wire_replay import replay_match


def test_deleting_the_groups_block_does_not_disable_the_role_check(real_match, write_body) -> None:
    """One deleted key used to switch the whole alternation fix off.

    The first version treated a missing ``groups`` as "a redacted copy carries no identity"
    and returned early. That rationale was empirically wrong — redaction strips the student
    blocks and repo URLs and always keeps ``groups`` — and it meant deleting one key plus
    swapping the roles published a 40-60 loss as a 60-40 win, against an untouched log.
    """
    cfg, log, body = real_match
    body.pop("groups")
    for game in body["sub_games"]:
        game["cop_group"], game["thief_group"] = game["thief_group"], game["cop_group"]
    with pytest.raises(ReplayMismatchError, match="no 'groups' block"):
        replay_match(cfg, log, write_body(body))


def test_deleting_the_role_fields_does_not_satisfy_the_role_check(real_match, write_body) -> None:
    """``record.get(field, expected)`` made absent compare EQUAL to correct — vacuous."""
    cfg, log, body = real_match
    for game in body["sub_games"]:
        game.pop("cop_group")
        game.pop("thief_group")
    with pytest.raises(ReplayMismatchError, match="alternation"):
        replay_match(cfg, log, write_body(body))


def test_the_groups_block_is_bound_to_the_agreement_not_to_itself(real_match, write_body) -> None:
    """Swapping ``groups`` AND every role together is internally consistent — and a forgery.

    The names now come from the frozen agreement in config, so §9.1 is checked against what
    the two groups agreed, not against a mapping the body declares about itself. On a lost
    match this edit would otherwise manufacture a win.
    """
    cfg, log, body = real_match
    body["groups"] = {"group_1": body["groups"]["group_2"], "group_2": body["groups"]["group_1"]}
    for game in body["sub_games"]:
        game["cop_group"], game["thief_group"] = game["thief_group"], game["cop_group"]
    with pytest.raises(ReplayMismatchError, match="jointly frozen agreement"):
        replay_match(cfg, log, write_body(body))


def test_fabricated_totals_are_rejected(real_match, write_body) -> None:
    """``totals_by_group`` is the published margin and was verified by NOTHING here.

    The SEND path re-derived it, but the replay is the command offered to a grader and the
    opposing group as §9.3 evidence, and it is what runs against a fresh clone's artifact.
    """
    cfg, log, body = real_match
    body["totals_by_group"] = dict.fromkeys(body["totals_by_group"], 0)
    with pytest.raises(ReplayMismatchError, match="totals_by_group"):
        replay_match(cfg, log, write_body(body))


def test_a_claim_that_contradicts_its_own_totals_is_rejected(real_match, write_body) -> None:
    """Publishing the loser's totals with the winner's claim must not survive."""
    cfg, log, body = real_match
    ours, theirs = body["groups"]["group_1"], body["groups"]["group_2"]
    body["totals_by_group"] = {ours: 40, theirs: 60}
    with pytest.raises(ReplayMismatchError):
        replay_match(cfg, log, write_body(body))


def test_the_claim_is_rejected_even_when_the_totals_are_honest(real_match, write_body) -> None:
    """The §9.2 claim needs its OWN test — mutating totals never reaches the claim check.

    Found by mutation testing: neutering the ``bonus_claim`` comparison left every existing
    test green, because the only test touching the claim also broke the TOTALS, and the
    totals check fires first. So the claim guard was live code with no test behind it.
    Here the totals stay exactly right and only the claim is flipped to the loser's side.
    """
    cfg, log, body = real_match
    ours, theirs = body["groups"]["group_1"], body["groups"]["group_2"]
    claim = cfg["game"]["bonus_claim"]
    body["bonus_claim"] = {ours: int(claim["loser"]), theirs: int(claim["winner"])}
    with pytest.raises(ReplayMismatchError, match="bonus_claim"):
        replay_match(cfg, log, write_body(body))


def test_a_body_that_is_not_a_bonus_game_is_rejected(real_match, write_body) -> None:
    """Envelope fields that carry meaning but no arithmetic were unchecked on this path."""
    cfg, log, body = real_match
    body["report_type"] = "not_a_bonus_game"
    with pytest.raises(ReplayMismatchError, match="report_type"):
        replay_match(cfg, log, write_body(body))


def test_a_sub_game_that_ends_before_it_starts_is_rejected(real_match, write_body) -> None:
    """Its own test, not a second block: the fixture body is ONE object per test, so a
    second scenario in the same function would inherit the first one's mutation and pass
    for the wrong reason."""
    cfg, log, body = real_match
    body["sub_games"][0]["start"] = "2027-01-01T00:00:00+02:00"
    body["sub_games"][0]["end"] = "2019-01-01T00:00:00+02:00"
    with pytest.raises(ReplayMismatchError, match="precedes start"):
        replay_match(cfg, log, write_body(body))
