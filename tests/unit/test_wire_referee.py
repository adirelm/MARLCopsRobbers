"""RED->GREEN tests for the §9 wire referee — clean-match behavior vs the partner brief."""

from __future__ import annotations

import json
from datetime import datetime

from src.mcp.wire_referee import WireReferee, build_draft_report, mask_payload
from src.reporting.bonus import build_bonus_report, validate_bonus
from tests.unit._wire_fixtures import REPOS, STUDENTS, StubAgent, stub_clients, wire_cfg


def _run(tmp_path):
    cfg = wire_cfg()
    referee = WireReferee(cfg, tmp_path / "wire_log.jsonl")
    g1, g2 = StubAgent(), StubAgent()
    return referee.play_match(stub_clients(g1, g2)), g1, g2, cfg


def test_match_plays_six_recorded_sub_games(tmp_path):
    result, _g1, _g2, cfg = _run(tmp_path)
    games = result["sub_games"]
    assert [g["id"] for g in games] == [1, 2, 3, 4, 5, 6]
    assert all(g["winner"] == "thief" and g["moves"] == 25 for g in games)
    assert all(g["scores"] == {"cop": 5, "thief": 10} for g in games)
    names = [cfg["wire_match"]["groups"][k]["name"] for k in ("group_1", "group_2")]
    assert [g["cop_group"] for g in games] == [names[0]] * 3 + [names[1]] * 3
    assert [g["thief_group"] for g in games] == [names[1]] * 3 + [names[0]] * 3
    assert result["totals_by_group"] == {names[0]: 45, names[1]: 45}


def test_timestamps_are_iso8601_milliseconds_jerusalem(tmp_path):
    result, *_ = _run(tmp_path)
    for game in result["sub_games"]:
        for key in ("start", "end"):
            stamp = datetime.fromisoformat(game[key])  # parses -> valid ISO-8601 with offset
            assert stamp.utcoffset() is not None
            assert "." in game[key]  # millisecond precision kept (byte-compare canon)


def test_session_ids_and_role_alternation(tmp_path):
    _result, g1, g2, _cfg = _run(tmp_path)
    assert [c["session_id"] for c in g1.new_calls] == [f"sg-{i}" for i in range(6)]
    assert [g1.roles[f"sg-{i}"] for i in range(6)] == ["cop"] * 3 + ["thief"] * 3
    assert [g2.roles[f"sg-{i}"] for i in range(6)] == ["thief"] * 3 + ["cop"] * 3


def test_mirror_sub_games_share_the_seed_layout(tmp_path):
    _result, g1, g2, _cfg = _run(tmp_path)
    for k in range(3):  # P7: sub-game k and k+3 use s_k -> identical spawns, roles swapped
        cop_base = next(c for c in g1.new_calls if c["session_id"] == f"sg-{k}")
        cop_mirror = next(c for c in g2.new_calls if c["session_id"] == f"sg-{k + 3}")
        assert cop_base["your_pos"] == cop_mirror["your_pos"]


def test_new_sub_game_payload_matches_brief(tmp_path):
    _result, g1, _g2, cfg = _run(tmp_path)
    call = g1.new_calls[0]
    grid = cfg["game"]["grid_size"]
    assert set(call) == {"session_id", "grid", "your_role", "your_pos", "max_moves"}
    assert call["grid"] == [grid, grid] and call["max_moves"] == cfg["game"]["max_moves"]


def test_request_move_payload_shape_and_zero_indexed_ticks(tmp_path):
    _result, g1, _g2, _cfg = _run(tmp_path)
    keys = {"session_id", "tick", "your_pos", "opponent_pos", "barriers", "barriers_left"}
    assert g1.move_calls and all(set(c) == keys for c in g1.move_calls)
    ticks = [c["tick"] for c in g1.move_calls if c["session_id"] == "sg-0"]
    assert ticks == list(range(25))  # 0-indexed, one per move, capped at max_moves
    assert all(c["barriers_left"] == 5 and c["barriers"] == [] for c in g1.move_calls)


def test_masking_opponent_pos_null_iff_outside_radius_2(tmp_path):
    _result, g1, g2, _cfg = _run(tmp_path)
    mine = {(c["session_id"], c["tick"]): c for c in g1.move_calls}
    theirs = {(c["session_id"], c["tick"]): c for c in g2.move_calls}
    assert mine and set(mine) == set(theirs)
    for key, ours in mine.items():
        other = theirs[key]
        dist = abs(ours["your_pos"][0] - other["your_pos"][0]) + abs(
            ours["your_pos"][1] - other["your_pos"][1]
        )
        assert ours["opponent_pos"] == (other["your_pos"] if dist <= 2 else None)
        assert other["opponent_pos"] == (ours["your_pos"] if dist <= 2 else None)


def test_tick_zero_opponent_is_always_null(tmp_path):
    _result, g1, _g2, _cfg = _run(tmp_path)
    assert all(c["opponent_pos"] is None for c in g1.move_calls if c["tick"] == 0)  # P6 spawns


def test_mask_payload_visibility_and_barrier_filtering():
    payload = mask_payload("sg-0", 4, (2, 2), (3, 3), frozenset({(2, 1), (0, 0)}), 3, 2)
    assert payload["opponent_pos"] == [3, 3]  # manhattan 2 -> visible
    assert payload["barriers"] == [[2, 1]]  # (0,0) is manhattan 4 away -> filtered out
    assert payload["barriers_left"] == 3 and payload["tick"] == 4
    far = mask_payload("sg-0", 0, (0, 0), (2, 1), frozenset(), 5, 2)
    assert far["opponent_pos"] is None  # manhattan 3 -> null


def test_result_log_is_timestamped_parseable_jsonl(tmp_path):
    result, *_ = _run(tmp_path)
    lines = (tmp_path / "wire_log.jsonl").read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in lines]
    assert entries and all("ts" in e and "direction" in e for e in entries)
    results = [e for e in entries if e["direction"] == "result"]
    assert [e["sub_game"]["id"] for e in results] == [1, 2, 3, 4, 5, 6]
    assert result["log_path"] == str(tmp_path / "wire_log.jsonl")


def test_records_feed_the_real_validate_bonus(tmp_path):
    result, _g1, _g2, cfg = _run(tmp_path)
    names = tuple(cfg["wire_match"]["groups"][k]["name"] for k in ("group_1", "group_2"))
    report = build_bonus_report(
        groups=names,
        repos=REPOS,
        students=(STUDENTS, STUDENTS),
        timezone=cfg["project"]["timezone"],
        results=result["sub_games"],
        game=cfg["game"],
        mutual_agreement=True,
    )
    validate_bonus(report, cfg["game"])  # the REAL §9 validator accepts the assembled body
    assert report["totals_by_group"] == result["totals_by_group"]


def test_build_draft_report_with_placeholder_players(tmp_path, monkeypatch):
    result, *_ = _run(tmp_path)

    def fake_players(path=None):
        side = "one" if path is None else "two"
        return {"group_name": f"grp-{side}", "students": STUDENTS, "github_repo": REPOS[0]}

    monkeypatch.setattr("src.mcp.wire_referee.load_players", fake_players)
    draft = build_draft_report(wire_cfg(), result["sub_games"])
    assert draft["report_type"] == "bonus_game" and draft["mutual_agreement"] is False
    assert draft["groups"] == {"group_1": "grp-one", "group_2": "grp-two"}
    assert [g["id"] for g in draft["sub_games"]] == [1, 2, 3, 4, 5, 6]
