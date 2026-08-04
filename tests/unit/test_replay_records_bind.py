"""The §9.4 RECORDS must be bound to the log, not merely replayable alongside it.

An adversarial pass found the replay verifying every MOVE while never checking the fields
that decide the outcome. Each test below reproduces a tampered artifact that passed a full
``replay_match`` before the fix — verified against the REAL match, not a synthetic one.

Scope, stated because it bounds what a green replay means: nothing here proves the partner's
server said what the log claims — no signature binds a logged reply to its author, so a
fully self-consistent fabricated match passes by construction. The counter-signature is
§9.3's independent byte-compare of both groups' drafts. What the replay proves is that ONE
group's log and records describe the SAME self-consistent match under the frozen seeds; the
gaps closed here are the ones where the records could drift from the log unilaterally.
"""

from __future__ import annotations

import json

import pytest

from src.mcp._replay_log import ReplayMismatchError, select_log_and_records
from src.mcp.wire_replay import replay_match
from src.utils.config_loader import load_config


@pytest.fixture
def real_match():
    """The committed §9 match (log + records) — skipped if this checkout lacks them."""
    cfg = load_config()
    try:
        log, records = select_log_and_records(cfg)
    except SystemExit:  # pragma: no cover - only on a checkout without the match artifacts
        pytest.skip("no committed wire match in this checkout")
    return cfg, log, json.loads(records.read_text(encoding="utf-8"))


def _write(tmp_path, body: dict):
    """Write a tampered §9.4 body and return its path."""
    path = tmp_path / "tampered_records.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_the_real_match_still_replays(real_match, tmp_path) -> None:
    """Positive control: the honest artifacts must keep passing, or the rest proves nothing."""
    cfg, log, body = real_match
    assert len(replay_match(cfg, log, _write(tmp_path, body))) == int(cfg["game"]["num_games"])


def test_swapping_who_played_cop_is_rejected(real_match, tmp_path) -> None:
    """THE headline hole: a role swap flipped our 60-40 win to 45-55 and replayed clean.

    §9.1 fixes the alternation — group_1 is cop in the first half, group_2 in the second —
    so ``cop_group``/``thief_group`` are DERIVED, never free fields. The replay checked that
    mirror halves shared a seed but never who played which side, and the margin is computed
    entirely from that assignment.
    """
    cfg, log, body = real_match
    for game in body["sub_games"]:
        if int(game["id"]) > int(cfg["game"]["num_games"]) // 2:
            game["cop_group"], game["thief_group"] = game["thief_group"], game["cop_group"]
    with pytest.raises(ReplayMismatchError, match="alternation"):
        replay_match(cfg, log, _write(tmp_path, body))


def test_dropping_a_lost_sub_game_is_rejected(real_match, tmp_path) -> None:
    """Deleting a loss from BOTH log and records left a 5-game 'match' that verified.

    Dropped from both, which is the real attack: removing it from the records alone was
    already caught by the log-vs-records comparison, so only the consistent deletion probes
    whether anything pins the match to §9.1's exactly-``num_games`` shape. Nothing did.
    """
    cfg, log, body = real_match
    body["sub_games"] = [g for g in body["sub_games"] if int(g["id"]) != 2]
    pruned = tmp_path / "pruned_log.jsonl"
    kept = [
        line
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.strip() and '"sg-1"' not in line  # sg-1 is the session id of sub-game 2
    ]
    pruned.write_text("\n".join(kept) + "\n", encoding="utf-8")
    with pytest.raises(ReplayMismatchError, match="sub-game ids"):
        replay_match(cfg, pruned, _write(tmp_path, body))


def test_an_extra_sub_game_is_rejected(real_match, tmp_path) -> None:
    """A 7th sub-game cloned from a win passed: gid 7 maps to pair 1 and is never examined."""
    cfg, log, body = real_match
    extra = json.loads(json.dumps(body["sub_games"][0]))
    extra["id"] = int(cfg["game"]["num_games"]) + 1
    body["sub_games"].append(extra)
    with pytest.raises(ReplayMismatchError, match="sub-game ids"):
        replay_match(cfg, log, _write(tmp_path, body))


def test_scores_that_do_not_follow_from_the_winner_are_rejected(real_match, tmp_path) -> None:
    """The §3.4 table decides the scores; a record may not invent its own."""
    cfg, log, body = real_match
    body["sub_games"][0]["scores"] = {"cop": 99, "thief": 0}
    with pytest.raises(ReplayMismatchError):
        replay_match(cfg, log, _write(tmp_path, body))


def test_deleting_the_groups_block_does_not_disable_the_role_check(real_match, tmp_path) -> None:
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
        replay_match(cfg, log, _write(tmp_path, body))


def test_deleting_the_role_fields_does_not_satisfy_the_role_check(real_match, tmp_path) -> None:
    """``record.get(field, expected)`` made absent compare EQUAL to correct — vacuous."""
    cfg, log, body = real_match
    for game in body["sub_games"]:
        game.pop("cop_group")
        game.pop("thief_group")
    with pytest.raises(ReplayMismatchError, match="alternation"):
        replay_match(cfg, log, _write(tmp_path, body))


def test_the_groups_block_is_bound_to_the_agreement_not_to_itself(real_match, tmp_path) -> None:
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
        replay_match(cfg, log, _write(tmp_path, body))


def test_fabricated_totals_are_rejected(real_match, tmp_path) -> None:
    """``totals_by_group`` is the published margin and was verified by NOTHING here.

    The SEND path re-derived it, but the replay is the command offered to a grader and the
    opposing group as §9.3 evidence, and it is what runs against a fresh clone's artifact.
    """
    cfg, log, body = real_match
    body["totals_by_group"] = dict.fromkeys(body["totals_by_group"], 0)
    with pytest.raises(ReplayMismatchError, match="totals_by_group"):
        replay_match(cfg, log, _write(tmp_path, body))


def test_a_claim_that_contradicts_its_own_totals_is_rejected(real_match, tmp_path) -> None:
    """Publishing the loser's totals with the winner's claim must not survive."""
    cfg, log, body = real_match
    ours, theirs = body["groups"]["group_1"], body["groups"]["group_2"]
    body["totals_by_group"] = {ours: 40, theirs: 60}
    with pytest.raises(ReplayMismatchError):
        replay_match(cfg, log, _write(tmp_path, body))


def test_a_body_that_is_not_a_bonus_game_is_rejected(real_match, tmp_path) -> None:
    """Envelope fields that carry meaning but no arithmetic were unchecked on this path."""
    cfg, log, body = real_match
    body["report_type"] = "not_a_bonus_game"
    with pytest.raises(ReplayMismatchError, match="report_type"):
        replay_match(cfg, log, _write(tmp_path, body))


def test_a_sub_game_that_ends_before_it_starts_is_rejected(real_match, tmp_path) -> None:
    """Its own test, not a second block: the fixture body is ONE object per test, so a
    second scenario in the same function would inherit the first one's mutation and pass
    for the wrong reason."""
    cfg, log, body = real_match
    body["sub_games"][0]["start"] = "2027-01-01T00:00:00+02:00"
    body["sub_games"][0]["end"] = "2019-01-01T00:00:00+02:00"
    with pytest.raises(ReplayMismatchError, match="precedes start"):
        replay_match(cfg, log, _write(tmp_path, body))
