"""A logged VOID must be evidenced, not merely asserted — both directions.

Split from ``test_replay_spare_seed.py`` at the 150-LOC cap. That file asks whether the
escalation BUDGET adds up; this one asks whether the voids paying into it are real.

Each test reproduces an attack that passed a full replay before the fix, using the project's
own frozen config rather than a convenient synthetic one.
"""

from __future__ import annotations

import json

import pytest

from src.mcp._replay_log import ReplayMismatchError, parse_wire_log
from src.mcp.wire_replay import replay_match
from src.utils.config_loader import load_config
from tests.unit._synthetic_wire import records_body, synth_session
from tests.unit.test_replay_spare_seed import _seeds


def _write(tmp_path, name, lines, records, cfg=None):
    """Write a synthetic (log, records) pair and return both paths."""
    log = tmp_path / f"wire_log_{name}.jsonl"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    recs = tmp_path / f"records_{name}.json"
    recs.write_text(json.dumps(records_body(cfg or load_config(), records)), encoding="utf-8")
    return log, recs


def _bare_hello(sid: str, role: str, pos: tuple[int, int], cfg) -> str:
    """A ``new_sub_game`` REQUEST line with no response, no move, no fault behind it."""
    return json.dumps(
        {
            "direction": "request",
            "label": f"g-{role}",
            "url": "http://x/new_sub_game",
            "payload": {
                "session_id": sid,
                "grid": [int(cfg["game"]["grid_size"])] * 2,
                "max_moves": int(cfg["game"]["max_moves"]),
                "your_role": role,
                "your_pos": list(pos),
            },
        }
    )


def test_bare_hello_lines_cannot_mint_voids(tmp_path) -> None:
    """The escalation budget must not be payable in counterfeit.

    Verified exploitable before the fix, against the real config: prefixing a match with
    THREE bare hello lines — no response, no move, no fault of any kind — made the parser
    count three voids and unlocked a spare seed, so a referee could shop the frozen list for
    a favourable layout at the cost of typing three lines. A void now requires the superseded
    attempt to have logged at least one request_move, which is a payload that has to survive
    verify_tick against the seeded env. Absence of evidence cannot be the evidence.
    """
    cfg = load_config()
    _pair, spares, needed = _seeds(cfg)
    lines, records = [], []
    for sid in ("sg-0", "sg-3"):  # a mirror pair played on a spare it never earned
        game_lines, record = synth_session(cfg, sid, spares[0], result_event=True, voids=0)
        lines += game_lines
        records.append(record)
    forged = [_bare_hello("sg-0", "cop", (i % 2, 0), cfg) for i in range(needed)] + lines
    log, recs = _write(tmp_path, "forged", forged, records)

    assert parse_wire_log(log)["sg-0"]["voids"] == 0, "bare hellos must not count as voids"
    with pytest.raises(ReplayMismatchError, match="never paid for"):
        replay_match(cfg, log, recs, full_match=False)


def test_unlimited_same_seed_re_rolls_are_rejected(tmp_path) -> None:
    """A floor alone lets a referee replay one layout until the result is favourable.

    The live SeedSchedule forces escalation on the n-th consecutive void and resets its
    counter, so a sub-game can carry at most ``needed - 1`` voids while staying on its base
    seed. A log claiming many more, with no spare resolved, describes a match the referee
    could not have played.

    Caught by the PER-SESSION rule rather than the match-wide ceiling, which is the whole
    point: the ceiling is a match total and could be spent entirely on one sub-game.
    """
    cfg = load_config()
    pair, _spares, needed = _seeds(cfg)
    lines, records = [], []
    for sid in ("sg-0", "sg-3"):
        game_lines, record = synth_session(cfg, sid, pair[0], result_event=True, voids=needed + 2)
        lines += game_lines
        records.append(record)
    log, recs = _write(tmp_path, "reroll", lines, records)
    with pytest.raises(ReplayMismatchError, match="BASE seed"):
        replay_match(cfg, log, recs, full_match=False)


def test_a_result_event_before_any_request_still_counts_voids(tmp_path) -> None:
    """The two session accumulators must share ONE shape.

    The ``result`` branch built its dict without ``voids``, so a log whose result event
    was flushed before its requests — any reordered or streamed log — raised KeyError on
    the first void instead of verifying.
    """
    cfg = load_config()
    _pair, spares, needed = _seeds(cfg)
    lines, _record = synth_session(cfg, "sg-0", spares[0], result_event=True, voids=needed)
    reordered = [ln for ln in lines if '"result"' in ln] + [ln for ln in lines if '"result"' not in ln]
    log = tmp_path / "wire_log_reordered.jsonl"
    log.write_text("\n".join(reordered) + "\n", encoding="utf-8")
    assert parse_wire_log(log)["sg-0"]["voids"] == needed


def test_a_garbage_request_move_cannot_mint_a_void(tmp_path) -> None:
    """The first void fix was COSMETIC — its own docstring claimed more than it did.

    It required a request_move in the superseded attempt and said the payload "must survive
    verify_tick against the seeded env". It did not: the states were discarded on the very
    next line, so an off-board ``your_pos`` with a negative barrier count and NO P5 masking
    fields still minted a void, at three lines of text apiece. The attempt is now retained
    and re-seeded by spawn match before the void is spent.
    """
    cfg = load_config()
    _pair, spares, needed = _seeds(cfg)
    lines, records = [], []
    for sid in ("sg-0", "sg-3"):
        game_lines, record = synth_session(cfg, sid, spares[0], result_event=True, voids=0)
        lines += game_lines
        records.append(record)
    forged = []
    for i in range(needed):
        forged += [
            _bare_hello("sg-0", "cop", (i % 2, 0), cfg),
            _bare_hello("sg-0", "thief", (4, 4), cfg),
            json.dumps(
                {
                    "direction": "request",
                    "label": "g-cop",
                    "url": "http://x/request_move",
                    "payload": {"session_id": "sg-0", "tick": 0, "your_pos": [99, 99], "barriers_left": -1},
                }
            ),
        ]
    log, recs = _write(tmp_path, "garbage", forged + lines, records)
    with pytest.raises(ReplayMismatchError, match="matching no P7 seed"):
        replay_match(cfg, log, recs, full_match=False)


def test_a_void_attempt_with_a_wrong_tick_zero_payload_is_rejected(tmp_path) -> None:
    """Correct spawns are not enough — the attempt's tick 0 must verify against the env.

    Found by mutation testing: neutering the ``verify_tick`` call inside the void path left
    every test green, because the spawn match alone already rejected the garbage payload.
    So the deeper check was live code with nothing behind it. Here the spawns are GENUINE
    (copied from a real seeded opening) and only the P5 masking is wrong — which is what an
    attacker who bothered to read the config would produce.
    """
    cfg = load_config()
    _pair, spares, needed = _seeds(cfg)
    lines, records = [], []
    for sid in ("sg-0", "sg-3"):
        game_lines, record = synth_session(cfg, sid, spares[0], result_event=True, voids=needed)
        lines += game_lines
        records.append(record)
    # Corrupt the P5 masking on the FIRST voided attempt's tick 0 only: opponent_pos is
    # disclosed when radius-2 masking says it must be withheld.
    for index, line in enumerate(lines):
        if '"request_move"' in line or "request_move" in line:
            entry = json.loads(line)
            if entry.get("url", "").endswith("/request_move"):
                entry["payload"]["opponent_pos"] = [0, 0]
                lines[index] = json.dumps(entry)
                break
    log, recs = _write(tmp_path, "voidmask", lines, records)
    with pytest.raises(ReplayMismatchError, match="void#0"):
        replay_match(cfg, log, recs, full_match=False)
