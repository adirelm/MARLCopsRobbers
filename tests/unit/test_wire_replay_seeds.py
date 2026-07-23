"""§9.3 replay seed resolution — a legitimately ESCALATED match must replay (P7 amendment).

After 3 consecutive voids the next unused spare seed replaces s_k for the pair k/k+3, so
the log's final runs were played under a SPARE seed. Spawn verification therefore tries
s_k first and then every configured spare in order; the accepted seed is reported in the
summary, mirror games must resolve to ONE seed, and no-match still raises loudly.
All logs here are synthetic FULL games driven through the real env (_synthetic_wire).
"""

from __future__ import annotations

import pytest

from src.mcp.wire_replay import ReplayMismatchError, replay_match
from tests.unit._synthetic_wire import write_match

_SEEDS = [101, 202, 303, 404, 505, 606]  # first 3 = pair seeds, rest = spares (P7 order)


def test_escalated_pair_replays_under_the_matching_spare_seed(cfg, tmp_path):
    cfg["wire_match"]["seeds"] = list(_SEEDS)
    # the pair (1, 4) escalated: both mirror games were ultimately played under spare 404
    log, records = write_match(tmp_path, cfg, [("sg-0", 404), ("sg-3", 404)])
    replays = replay_match(cfg, log, records)
    assert [g["gid"] for g in replays] == [1, 4]
    assert [g["seed"] for g in replays] == [404, 404]  # the SPARE, reported in the summary


def test_unescalated_games_still_resolve_to_their_pair_seed(cfg, tmp_path):
    cfg["wire_match"]["seeds"] = list(_SEEDS)
    log, records = write_match(tmp_path, cfg, [("sg-1", 202), ("sg-4", 202)])
    assert [g["seed"] for g in replay_match(cfg, log, records)] == [202, 202]


def test_spawns_matching_no_configured_seed_raise_loudly(cfg, tmp_path):
    cfg["wire_match"]["seeds"] = list(_SEEDS)
    log, records = write_match(tmp_path, cfg, [("sg-0", 12345)])  # a seed NOT in the schedule
    with pytest.raises(ReplayMismatchError, match="spare"):
        replay_match(cfg, log, records)


def test_mirror_pair_resolving_to_different_seeds_raises(cfg, tmp_path):
    cfg["wire_match"]["seeds"] = list(_SEEDS)
    # both games individually valid, but game 1 under s_1 and game 4 under a spare:
    # an impossible escalation history (the spare replaces the seed for the WHOLE pair)
    log, records = write_match(tmp_path, cfg, [("sg-0", 101), ("sg-3", 404)])
    with pytest.raises(ReplayMismatchError, match="mirror pair"):
        replay_match(cfg, log, records)
