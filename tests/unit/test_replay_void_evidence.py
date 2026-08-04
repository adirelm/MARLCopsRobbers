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
from tests.unit._synthetic_wire import synth_session
from tests.unit.test_replay_spare_seed import _seeds


def _write(tmp_path, name, lines, records):
    """Write a synthetic (log, records) pair and return both paths."""
    log = tmp_path / f"wire_log_{name}.jsonl"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    recs = tmp_path / f"records_{name}.json"
    recs.write_text(json.dumps({"sub_games": records}), encoding="utf-8")
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
    """
    cfg = load_config()
    pair, _spares, needed = _seeds(cfg)
    lines, records = [], []
    for sid in ("sg-0", "sg-3"):
        game_lines, record = synth_session(cfg, sid, pair[0], result_event=True, voids=needed + 2)
        lines += game_lines
        records.append(record)
    log, recs = _write(tmp_path, "reroll", lines, records)
    with pytest.raises(ReplayMismatchError, match="never escalated"):
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
