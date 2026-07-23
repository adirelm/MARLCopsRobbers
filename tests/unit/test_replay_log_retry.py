"""Wire-log parsing: a P8 TRANSPORT RETRY of a hello is an idempotent no-op, NOT a void.

The void heuristic ("role already in spawns -> wipe the run") used to misfire on a
legitimately retried hello: the second role's identical re-POST wiped the FIRST role's
spawn, making a VALID log unreplayable. The rule now wipes only when the position
changed OR moves (actions/states) were recorded since — an identical re-hello with
nothing in between is the transport retry the brief's idempotency contract allows.
"""

from __future__ import annotations

import json

from src.mcp._replay_log import parse_wire_log

_HELLO_URL, _MOVE_URL = "http://x/new_sub_game", "http://x/request_move"


def _req(label, url, payload):
    return json.dumps({"direction": "request", "label": label, "url": url, "payload": payload})


def _resp(label, body):
    return json.dumps({"direction": "response", "label": label, "response": body})


def _hello(role, pos):
    return {"session_id": "sg-0", "your_role": role, "your_pos": pos}


def _move(tick, pos):
    return {
        "session_id": "sg-0",
        "tick": tick,
        "your_pos": pos,
        "opponent_pos": None,
        "barriers": [],
        "barriers_left": 5,
    }


def _parse(tmp_path, lines):
    log = tmp_path / "log.jsonl"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return parse_wire_log(log)["sg-0"]


def test_identical_second_role_hello_retry_keeps_the_first_roles_spawn(tmp_path):
    sess = _parse(
        tmp_path,
        [
            _req("g-cop", _HELLO_URL, _hello("cop", [3, 3])),
            _resp("g-cop", {"ok": True}),
            _req("g-thief", _HELLO_URL, _hello("thief", [0, 1])),  # attempt 0: reply lost (P8 fault)
            _req("g-thief", _HELLO_URL, _hello("thief", [0, 1])),  # attempt 1: IDENTICAL retry
            _resp("g-thief", {"ok": True}),
        ],
    )
    assert sess["spawns"] == {"cop": (3, 3), "thief": (0, 1)}  # nothing wiped


def test_same_pos_re_hello_with_recorded_actions_still_wipes(tmp_path):
    sess = _parse(
        tmp_path,
        [
            _req("g-cop", _HELLO_URL, _hello("cop", [3, 3])),
            _resp("g-cop", {"ok": True}),
            _req("g-cop", _MOVE_URL, _move(0, [3, 3])),
            _resp("g-cop", {"action": "up"}),
            _req("g-cop", _HELLO_URL, _hello("cop", [3, 3])),  # void replay: SAME seed => SAME pos
            _resp("g-cop", {"ok": True}),
        ],
    )
    assert sess["spawns"] == {"cop": (3, 3)} and sess["actions"] == {} and sess["states"] == {}


def test_same_pos_re_hello_after_a_faulted_move_request_wipes_states(tmp_path):
    # the COMMON real void: a move request was logged (states) but its reply never came
    sess = _parse(
        tmp_path,
        [
            _req("g-cop", _HELLO_URL, _hello("cop", [3, 3])),
            _resp("g-cop", {"ok": True}),
            _req("g-cop", _MOVE_URL, _move(0, [3, 3])),  # both attempts timed out -> void
            _req("g-cop", _HELLO_URL, _hello("cop", [3, 3])),
            _resp("g-cop", {"ok": True}),
        ],
    )
    assert sess["spawns"] == {"cop": (3, 3)} and sess["states"] == {}  # stale tick-0 truth wiped
