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

from src.mcp._replay_log import ReplayMismatchError
from src.mcp.wire_replay import replay_match


def test_the_real_match_still_replays(real_match, write_body) -> None:
    """Positive control: the honest artifacts must keep passing, or the rest proves nothing."""
    cfg, log, body = real_match
    assert len(replay_match(cfg, log, write_body(body))) == int(cfg["game"]["num_games"])


def test_swapping_who_played_cop_is_rejected(real_match, write_body) -> None:
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
        replay_match(cfg, log, write_body(body))


def test_dropping_a_lost_sub_game_is_rejected(real_match, write_body, tmp_path) -> None:
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
        replay_match(cfg, pruned, write_body(body))


def test_an_extra_sub_game_is_rejected(real_match, write_body) -> None:
    """A 7th sub-game cloned from a win passed: gid 7 maps to pair 1 and is never examined."""
    cfg, log, body = real_match
    extra = json.loads(json.dumps(body["sub_games"][0]))
    extra["id"] = int(cfg["game"]["num_games"]) + 1
    body["sub_games"].append(extra)
    with pytest.raises(ReplayMismatchError, match="sub-game ids"):
        replay_match(cfg, log, write_body(body))


def test_scores_that_do_not_follow_from_the_winner_are_rejected(real_match, write_body) -> None:
    """The §3.4 table decides the scores; a record may not invent its own."""
    cfg, log, body = real_match
    body["sub_games"][0]["scores"] = {"cop": 99, "thief": 0}
    with pytest.raises(ReplayMismatchError):
        replay_match(cfg, log, write_body(body))
