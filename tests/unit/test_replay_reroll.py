"""Re-rolling a sub-game must COST something — the three ways it was free.

A third adversarial pass showed the void machinery could be bypassed or bought cheaply in
three independent ways, one of which needed no void at all. Each test reproduces an exhibit
that passed a full ``replay_match`` before the fix.

The threat these all serve is seed shopping: replay the deciding sub-game, or shop the
frozen seed list, until the layout or the result is favourable.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.mcp._replay_log import ReplayMismatchError, parse_wire_log
from src.mcp._replay_pair import select_log_and_records
from src.mcp._replay_verify import verify_escalation_budget, verify_session_voids
from src.utils.config_loader import load_config


def _seed_sets(cfg):
    """Return (base seeds, spare seeds, voids per escalation)."""
    pairs = int(cfg["game"]["num_games"]) // 2
    seeds = [int(s) for s in cfg["wire_match"]["seeds"]]
    return seeds[:pairs], seeds[pairs:], int(cfg["wire_match"]["max_void_replays"])


def test_a_spare_seed_session_is_still_bounded() -> None:
    """H1: exempting spare sessions re-opened the hole the per-session cap was written for.

    A sub-game the attacker escalated FIRST then had no per-session bound at all, so 27
    undisclosed replays of the deciding sub-game passed. Reaching spare #n costs n
    escalations plus at most ``needed - 1`` further voids before the run that succeeded.
    """
    cfg = load_config()
    _base, spares, needed = _seed_sets(cfg)
    cap = needed * 1 + (needed - 1)  # spare #1
    assert verify_session_voids(cfg, "sg-0", cap, spares[0]) is None
    for voids in (cap + 1, 12, 27):
        with pytest.raises(ReplayMismatchError, match="re-rolling one layout"):
            verify_session_voids(cfg, "sg-0", voids, spares[0])


def test_reaching_a_later_spare_costs_more_than_the_first() -> None:
    """H3: billing per DISTINCT spare priced every layout at one escalation.

    SeedSchedule consumes spares strictly IN ORDER, so spare #2 costs two escalations and
    #3 costs three. Charging one for all of them handed the referee a free CHOICE of layout
    for the price of the cheapest — which is seed shopping exactly.
    """
    cfg = load_config()
    base, spares, needed = _seed_sets(cfg)
    for index, spare in enumerate(spares):
        played = [spare, base[1], base[2], spare, base[1], base[2]]
        owed = needed * (index + 1)
        assert verify_escalation_budget(cfg, played, owed) == 1
        if index:  # the first spare legitimately costs one escalation
            with pytest.raises(ReplayMismatchError, match="never paid for"):
                verify_escalation_budget(cfg, played, owed - 1)


def test_the_honest_traces_are_still_accepted() -> None:
    """Both tightenings must leave every legal SeedSchedule trace alone."""
    cfg = load_config()
    base, spares, _needed = _seed_sets(cfg)
    assert verify_escalation_budget(cfg, base * 2, 0) == 0
    honest_17 = [spares[0], base[1], base[2], base[0], base[1], base[2]]
    assert verify_escalation_budget(cfg, honest_17, 17) == 1


def test_a_second_attempt_at_one_tick_cannot_hide_as_a_p8_retry() -> None:
    """H2 — the severe one: it needs NO void at all.

    ``states``/``actions`` were last-write-wins per (tick, role), so a referee could re-roll
    a sub-game 21 times without emitting a single hello: every abandoned attempt collapsed
    into the surviving one, no void was claimed, nothing was counted, and the replay
    reported a clean match. P8 permits re-POSTING the SAME body, never a second attempt.
    """
    cfg = load_config()
    log, _records = select_log_and_records(cfg)
    lines = log.read_text(encoding="utf-8").splitlines()
    request = next(
        entry
        for entry in map(json.loads, lines)
        if entry.get("direction") == "request"
        and entry.get("url", "").endswith("/request_move")
        and entry["payload"]["session_id"] == "sg-0"
        and int(entry["payload"]["tick"]) == 0
    )
    reply = next(
        entry
        for entry in map(json.loads, lines)
        if entry.get("direction") == "response" and entry.get("label") == request["label"]
    )
    first = {**reply, "response": {"action": "down"}}
    second = {**reply, "response": {"action": "up"}}  # a DIFFERENT action, same tick
    forged = [json.dumps(request), json.dumps(first), json.dumps(request), json.dumps(second)]
    path = Path(tempfile.mkdtemp()) / "reattempt.jsonl"
    path.write_text("\n".join(forged + lines) + "\n", encoding="utf-8")
    with pytest.raises(ReplayMismatchError, match=r"re-attempt"):
        parse_wire_log(path, int(cfg["wire_match"]["retries"]))


def test_the_committed_logs_still_parse() -> None:
    """Positive control: a GENUINE P8 retry (identical re-POST) must remain legal.

    The committed void-drill log contains one, so a naive "any repeated tick is a
    re-attempt" rule would false-positive it — which is why the rule compares bodies.
    """
    cfg = load_config()
    retries = int(cfg["wire_match"]["retries"])
    for log in sorted(Path(cfg["wire_match"]["log_dir"]).glob("*.jsonl")):
        assert parse_wire_log(log, retries), f"{log.name} parsed empty"
