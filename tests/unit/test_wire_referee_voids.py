"""RED->GREEN tests for the P7 seed schedule with the agreed VOID AMENDMENT.

The amendment (decided before implementation): a technical void REPLAYS the same
sub-game with the SAME seed; only after 3 CONSECUTIVE voids of one sub-game does the
next unused spare seed replace s_k for the whole pair k/k+3 — replaying the pair's
base game if it was already played. Game 5 voiding must NOT replay game 2 unless the
3-void escalation fires.
"""

from __future__ import annotations

import pytest

from src.mcp.wire_referee import SeedSchedule, WireReferee
from tests.unit._wire_fixtures import SEEDS, StubAgent, stub_clients, wire_cfg


def test_schedule_base_order_and_mirrored_seeds():
    sched = SeedSchedule(SEEDS, 6, 3)
    played = []
    while (head := sched.next_game()) is not None:
        played.append(head)
        sched.record_result(head[0])
    assert played == [(1, 1), (2, 2), (3, 3), (4, 1), (5, 2), (6, 3)]


def _after_four_results() -> SeedSchedule:
    sched = SeedSchedule(SEEDS, 6, 3)
    for gid in (1, 2, 3, 4):
        sched.record_result(gid)
    return sched


def test_schedule_void_replays_same_game_same_seed():
    sched = _after_four_results()
    assert sched.next_game() == (5, 2)
    assert sched.record_void(5) == []  # void 1 -> replay game 5, SAME seed
    assert sched.next_game() == (5, 2)
    assert sched.record_void(5) == []  # void 2 -> still the same seed, game 2 untouched
    assert sched.next_game() == (5, 2)


def test_schedule_three_voids_escalate_to_spare_and_replay_base():
    sched = _after_four_results()
    sched.record_void(5)
    sched.record_void(5)
    assert sched.record_void(5) == [2]  # 3rd consecutive void -> s_2 replaced for the PAIR
    assert sched.next_game() == (2, SEEDS[3])  # base replays FIRST, on the spare seed
    sched.record_result(2)
    assert sched.next_game() == (5, SEEDS[3])  # the mirror uses the SAME replacement seed
    sched.record_result(5)
    assert sched.next_game() == (6, SEEDS[2])  # pair 3 is untouched


def test_schedule_escalation_on_a_base_game_requeues_nothing():
    sched = SeedSchedule(SEEDS, 6, 3)
    sched.record_result(1)
    assert sched.record_void(2) == []
    assert sched.record_void(2) == []
    assert sched.record_void(2) == []  # base game: no already-played mirror to invalidate
    assert sched.next_game() == (2, SEEDS[3])


def test_schedule_void_counter_resets_on_a_valid_result():
    sched = SeedSchedule(SEEDS, 6, 3)
    sched.record_void(1)
    sched.record_void(1)
    sched.record_result(1)  # a valid game breaks the consecutive-void run
    sched.record_void(2)
    assert sched.record_void(2) == []  # only 2 consecutive for game 2 -> no escalation
    assert sched.next_game() == (2, SEEDS[1])


def test_schedule_requires_the_agreed_minimum_of_seeds():
    with pytest.raises(ValueError):
        SeedSchedule([1, 2, 3], 6, 3)


def test_schedule_spare_exhaustion_raises_runtime_error():
    sched = SeedSchedule([1, 2, 3, 4, 5, 6], 6, 3)  # exactly three spares
    for spare in (4, 5, 6):  # burn every spare on game 1
        sched.record_void(1)
        sched.record_void(1)
        assert sched.record_void(1) == []
        assert sched.next_game() == (1, spare)
    sched.record_void(1)
    sched.record_void(1)
    with pytest.raises(RuntimeError):
        sched.record_void(1)


def test_match_void_replays_same_sub_game_same_seed(tmp_path):
    referee = WireReferee(wire_cfg(), tmp_path / "log.jsonl")
    g1, g2 = StubAgent(), StubAgent(fail_first={"sg-4": 2})
    result = referee.play_match(stub_clients(g1, g2))
    assert [g["id"] for g in result["sub_games"]] == [1, 2, 3, 4, 5, 6]
    sg4_new = [c for c in g2.new_calls if c["session_id"] == "sg-4"]
    assert len(sg4_new) == 3  # two voided attempts + the valid replay
    assert all(c["your_pos"] == sg4_new[0]["your_pos"] for c in sg4_new)  # SAME seed replayed
    assert len([c for c in g1.new_calls if c["session_id"] == "sg-1"]) == 1  # game 2 NOT replayed


def test_match_three_voids_escalate_and_replay_the_base(tmp_path):
    referee = WireReferee(wire_cfg(), tmp_path / "log.jsonl")
    g1, g2 = StubAgent(), StubAgent(fail_first={"sg-4": 3})
    result = referee.play_match(stub_clients(g1, g2))
    assert [g["id"] for g in result["sub_games"]] == [1, 2, 3, 4, 5, 6]
    sg1_new = [c for c in g1.new_calls if c["session_id"] == "sg-1"]  # base of the voided pair
    assert len(sg1_new) == 2  # played, invalidated by the escalation, replayed
    assert sg1_new[0]["your_pos"] != sg1_new[1]["your_pos"]  # replacement seed -> new layout
    sg4_new = [c for c in g2.new_calls if c["session_id"] == "sg-4"]
    assert len(sg4_new) == 4  # three voided attempts + one valid game on the spare seed
    assert sg1_new[1]["your_pos"] == sg4_new[3]["your_pos"]  # mirrors stay on identical layouts
