"""§9.3 wire-replay tests — parse, deterministic re-run, records cross-check, loud failures.

Acceptance (vs the COMMITTED rehearsal artifacts, skip-if-absent): exactly 6 sessions
parse; spawn verification passes; per-game moves/winner/scores match the records; and
every tamper (action, seed schedule, records row) raises ReplayMismatchError LOUDLY.
"""

from __future__ import annotations

import json

import pytest

from src.mcp.wire_replay import ReplayMismatchError, parse_wire_log, replay_match, replay_sub_game
from tests.unit._replay_fixtures import rehearsal_cfg, rehearsal_paths

_ZERO = {"cop": 0, "thief": 0}


def _first_record(records_path) -> dict:
    return json.loads(records_path.read_text(encoding="utf-8"))["sub_games"][0]


def test_parse_finds_exactly_six_sessions_with_spawns_and_contiguous_ticks():
    log, _records = rehearsal_paths()
    sessions = parse_wire_log(log)
    assert sorted(sessions) == [f"sg-{i}" for i in range(6)]
    for sess in sessions.values():
        assert set(sess["spawns"]) == {"cop", "thief"}
        assert sorted(sess["actions"]) == list(range(len(sess["actions"])))
        assert all(set(pair) == {"cop", "thief"} for pair in sess["actions"].values())


def test_replay_match_verifies_all_six_games_against_records():
    cfg = rehearsal_cfg()
    log, records_path = rehearsal_paths()
    replays = replay_match(cfg, log, records_path)
    records = {r["id"]: r for r in json.loads(records_path.read_text(encoding="utf-8"))["sub_games"]}
    assert [g["gid"] for g in replays] == [1, 2, 3, 4, 5, 6]
    for game in replays:
        want = records[game["gid"]]
        assert (game["moves"], game["winner"]) == (want["moves"], want["winner"])
        assert game["scores"] == want["scores"]
        assert len(game["frames"]) == game["moves"] + 1  # spawn frame + one per tick


def test_replay_frames_carry_spawn_and_terminal_hud():
    cfg = rehearsal_cfg()
    log, records_path = rehearsal_paths()
    game = replay_match(cfg, log, records_path)[0]
    first, last = game["frames"][0], game["frames"][-1]
    assert (first.move, first.winner, first.last_action) == (0, None, None)
    assert last.move == game["moves"] and last.winner == game["winner"]
    assert set(last.last_action) == {"cop_0", "thief"}


def test_p7_pair_mapping_shares_seed_between_k_and_k_plus_3():
    cfg = rehearsal_cfg()
    log, records_path = rehearsal_paths()
    replays = replay_match(cfg, log, records_path)
    for k in range(3):
        assert replays[k]["seed"] == replays[k + 3]["seed"] == cfg["wire_match"]["seeds"][k]


def test_tampered_action_raises_loudly(cfg):
    log, records_path = rehearsal_paths()
    sess = parse_wire_log(log)["sg-0"]
    sess["actions"][2]["cop"] = "down" if sess["actions"][2]["cop"] != "down" else "up"
    with pytest.raises(ReplayMismatchError):
        replay_sub_game(cfg, "sg-0", sess, _first_record(records_path), dict(_ZERO))


def test_wrong_seed_schedule_fails_spawn_verification(cfg):
    log, records_path = rehearsal_paths()
    # a WHOLLY wrong schedule: the true seed appears neither as s_k nor as any spare —
    # the referee-recorded seed (result event) is rejected against it before any spawn try
    cfg["wire_match"]["seeds"] = [s + 1 for s in cfg["wire_match"]["seeds"]]
    sess = parse_wire_log(log)["sg-0"]
    with pytest.raises(ReplayMismatchError, match="neither s_k nor"):
        replay_sub_game(cfg, "sg-0", sess, _first_record(records_path), dict(_ZERO))


def test_tampered_record_winner_raises(cfg):
    log, records_path = rehearsal_paths()
    sess = parse_wire_log(log)["sg-0"]
    record = {**_first_record(records_path), "winner": "thief"}
    with pytest.raises(ReplayMismatchError, match="record"):
        replay_sub_game(cfg, "sg-0", sess, record, dict(_ZERO))


def test_illegal_thief_action_and_gapped_ticks_raise():
    cfg = rehearsal_cfg()
    log, records_path = rehearsal_paths()
    record = _first_record(records_path)
    sess = parse_wire_log(log)["sg-0"]
    sess["actions"][0]["thief"] = "place_barrier"  # cop-only action on the thief side
    with pytest.raises(ReplayMismatchError, match="illegal"):
        replay_sub_game(cfg, "sg-0", sess, record, dict(_ZERO))
    gapped = parse_wire_log(log)["sg-0"]
    del gapped["actions"][1]
    with pytest.raises(ReplayMismatchError, match="non-contiguous"):
        replay_sub_game(cfg, "sg-0", gapped, record, dict(_ZERO))
    partial = parse_wire_log(log)["sg-0"]
    del partial["actions"][0]["thief"]
    with pytest.raises(ReplayMismatchError, match="roles"):
        replay_sub_game(cfg, "sg-0", partial, record, dict(_ZERO))


def test_records_session_count_mismatch_raises(tmp_path):
    """A records file short of the §9.1 match size is rejected before anything is replayed.

    Now caught by the completeness check rather than the log-vs-records comparison, which is
    the more specific answer: dropping a record makes the id set wrong, and that is true of
    the records alone — no log needed to see it.
    """
    cfg = rehearsal_cfg()
    log, records_path = rehearsal_paths()
    body = json.loads(records_path.read_text(encoding="utf-8"))
    body["sub_games"] = body["sub_games"][:5]  # drop one record: 6 sessions vs 5 records
    short = tmp_path / "short.json"
    short.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ReplayMismatchError, match="sub-game ids"):
        replay_match(cfg, log, short)


def test_committed_void_path_log_replays_via_last_hello():
    cfg = rehearsal_cfg()
    """The REAL referee void log (a voided sub-game 2 + its same-seed replay) verifies cleanly."""
    log, records_path = rehearsal_paths()
    void_log = log.parent / "wire_log_voidtest.jsonl"
    if not void_log.exists():
        pytest.skip("committed void-path rehearsal log not present")
    replays = replay_match(cfg, void_log, records_path)
    assert [g["winner"] for g in replays] == ["cop", "cop", "thief", "cop", "cop", "thief"]


def test_void_re_hello_supersedes_earlier_run(tmp_path):
    """A second hello pair for a session wipes the voided first run (last hello wins)."""

    def _req(url: str, payload: dict) -> str:
        return json.dumps({"direction": "request", "label": "g-cop", "url": url, "payload": payload})

    def _resp(body: dict) -> str:
        return json.dumps({"direction": "response", "label": "g-cop", "response": body})

    hello = {"session_id": "sg-0", "your_role": "cop", "your_pos": [3, 3]}
    # full wire payload per the brief — the parser now RELIES on the contract fields
    move = {
        "session_id": "sg-0",
        "tick": 0,
        "your_pos": [3, 3],
        "opponent_pos": None,
        "barriers": [],
        "barriers_left": 5,
    }
    lines = [
        _req("http://x/new_sub_game", hello),
        _resp({"ok": True}),
        _req("http://x/request_move", move),
        _resp({"action": "up"}),  # voided first run: this action must be wiped
        _req("http://x/new_sub_game", {**hello, "your_pos": [1, 4]}),
        _resp({"ok": True}),
    ]
    log = tmp_path / "log.jsonl"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sess = parse_wire_log(log)["sg-0"]
    assert sess["spawns"] == {"cop": (1, 4)}  # the LAST hello's spawn
    assert sess["actions"] == {}  # the voided run's moves are gone


def test_midgame_tamper_in_25_move_game_is_caught_at_the_tick():
    cfg = rehearsal_cfg()
    """The silent-divergence hole, pinned: flip ONE mid-game action of the 25-move sg-2.

    Terminal-summary checking alone can miss a divergence that wanders back to the same
    outcome; the per-tick ground-truth check (each role's logged ``your_pos`` +
    ``barriers_left``) must catch it AT the tick after the tamper, mid-game.
    """
    log, records_path = rehearsal_paths()
    sessions = parse_wire_log(log)
    sess = sessions["sg-2"]  # the 25-move thief-win game
    assert len(sess["actions"]) == 25
    tampered = sess["actions"][12]["thief"]
    sess["actions"][12]["thief"] = "down" if tampered != "down" else "up"
    record = json.loads(records_path.read_text(encoding="utf-8"))["sub_games"][2]
    with pytest.raises(ReplayMismatchError, match=r"tick 1[23]"):  # caught mid-game, not at move 25
        replay_sub_game(cfg, "sg-2", sess, record, dict(_ZERO))
