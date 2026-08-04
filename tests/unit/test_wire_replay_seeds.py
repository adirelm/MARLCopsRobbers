"""§9.3 replay seed resolution — a legitimately ESCALATED match must replay (P7 amendment).

PRIMARY seed source: the referee's logged ``result`` event (exact seed + session_id),
spawn-verified as the tamper cross-check; distinct seeds CAN collide on spawns (505 and
1073 both give cop=(2,0)/thief=(4,2) on the 5x5 board), so spawn-matching alone can pick
a decoy. FALLBACK for logs predating seed events: s_k then every configured spare in
order. The accepted seed is reported in the summary, mirror games must resolve to ONE
seed, no spare may serve two pairs, and no-match still raises loudly.
All logs here are synthetic FULL games driven through the real env (_synthetic_wire).
"""

from __future__ import annotations

import pytest

from src.mcp.wire_replay import ReplayMismatchError, replay_match
from tests.unit._synthetic_wire import write_match

# These logs are deliberate SUBSETS (one mirror pair) so seed resolution can be probed
# in isolation, so they pass full_match=False; the §9.1 completeness check itself is
# covered against the REAL match in test_replay_records_bind.py.
_SEEDS = [101, 202, 303, 404, 505, 606]  # first 3 = pair seeds, rest = spares (P7 order)


def test_escalated_pair_replays_under_the_matching_spare_seed(cfg, tmp_path):
    cfg["wire_match"]["seeds"] = list(_SEEDS)
    # the pair (1, 4) escalated: both mirror games were ultimately played under spare 404,
    # so the log must also CARRY the escalation that bought 404 (P7 charges for spares)
    log, records = write_match(
        tmp_path, cfg, [("sg-0", 404), ("sg-3", 404)], voids=int(cfg["wire_match"]["max_void_replays"])
    )
    replays = replay_match(cfg, log, records, full_match=False)
    assert [g["gid"] for g in replays] == [1, 4]
    assert [g["seed"] for g in replays] == [404, 404]  # the SPARE, reported in the summary


def test_unescalated_games_still_resolve_to_their_pair_seed(cfg, tmp_path):
    cfg["wire_match"]["seeds"] = list(_SEEDS)
    log, records = write_match(tmp_path, cfg, [("sg-1", 202), ("sg-4", 202)])
    assert [g["seed"] for g in replay_match(cfg, log, records, full_match=False)] == [202, 202]


def test_spawns_matching_no_configured_seed_raise_loudly(cfg, tmp_path):
    cfg["wire_match"]["seeds"] = list(_SEEDS)
    log, records = write_match(tmp_path, cfg, [("sg-0", 12345)])  # a seed NOT in the schedule
    with pytest.raises(ReplayMismatchError, match="spare"):
        replay_match(cfg, log, records, full_match=False)


def test_mirror_pair_resolving_to_different_seeds_raises(cfg, tmp_path):
    cfg["wire_match"]["seeds"] = list(_SEEDS)
    # both games individually valid, but game 1 under s_1 and game 4 under a spare:
    # an impossible escalation history (the spare replaces the seed for the WHOLE pair)
    log, records = write_match(tmp_path, cfg, [("sg-0", 101), ("sg-3", 404)])
    with pytest.raises(ReplayMismatchError, match="mirror pair"):
        replay_match(cfg, log, records, full_match=False)


def test_recorded_seed_beats_a_colliding_decoy_spare(cfg, tmp_path):
    # seeds 1073 and 505 COLLIDE on spawns (cop=(2,0), thief=(4,2)); the decoy sits FIRST
    # in the spare order, so spawn-matching alone silently mis-replays under 1073 ...
    cfg["wire_match"]["seeds"] = [101, 202, 303, 1073, 505, 606]
    games = [("sg-0", 505), ("sg-3", 505)]
    # 505 is a SPARE, so the log must carry P7's escalation or the replay rejects it before
    # the seed-resolution ambiguity this test exists to probe.
    voids = int(cfg["wire_match"]["max_void_replays"])
    log, records = write_match(tmp_path, cfg, games, voids=voids)
    assert [g["seed"] for g in replay_match(cfg, log, records, full_match=False)] == [
        1073,
        1073,
    ]  # the ambiguity trap
    # ... but with the referee's result events the RECORDED seed wins, spawn-verified:
    log, records = write_match(tmp_path, cfg, games, result_events=True, voids=voids)
    assert [g["seed"] for g in replay_match(cfg, log, records, full_match=False)] == [505, 505]


def test_recorded_seed_outside_the_schedule_raises(cfg, tmp_path):
    cfg["wire_match"]["seeds"] = [101, 202, 303, 404, 606, 707]  # 505 is NOT in the schedule
    log, records = write_match(tmp_path, cfg, [("sg-0", 505)], result_events=True)
    with pytest.raises(ReplayMismatchError, match="recorded result seed"):
        replay_match(cfg, log, records, full_match=False)


def test_recorded_seed_whose_spawns_mismatch_raises(cfg, tmp_path):
    cfg["wire_match"]["seeds"] = list(_SEEDS)
    log, records = write_match(
        tmp_path, cfg, [("sg-0", 505)], result_events=True, voids=int(cfg["wire_match"]["max_void_replays"])
    )
    tampered = log.read_text(encoding="utf-8").replace('"seed": 505', '"seed": 404')
    log.write_text(tampered, encoding="utf-8")  # the result event now LIES about the seed
    with pytest.raises(ReplayMismatchError, match="recorded seed 404"):
        replay_match(cfg, log, records, full_match=False)


def test_one_spare_serving_two_pairs_raises(cfg, tmp_path):
    cfg["wire_match"]["seeds"] = list(_SEEDS)
    # pairs 1/4 AND 2/5 all "escalated" onto spare 404 — each mirror pair agrees internally,
    # but SeedSchedule consumes every spare at most ONCE, so this schedule is impossible
    games = [("sg-0", 404), ("sg-3", 404), ("sg-1", 404), ("sg-4", 404)]
    log, records = write_match(tmp_path, cfg, games)
    with pytest.raises(ReplayMismatchError, match="spare seed 404"):
        replay_match(cfg, log, records, full_match=False)
