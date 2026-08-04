"""P7 escalation must be PAID FOR in the log — a spare seed needs its voids.

The verifier accepted `s_k` or ANY spare with nothing tying a spare to the escalation that
earns it. Spawn-matching does not close that: a referee shopping for a favourable layout
plays the spare for real, so the spawns legitimately match it, and simply logs no voids.

The budget is deliberately MATCH-wide. The first fix billed each session for its own
escalation, which is tighter and WRONG — see
:func:`test_the_budget_is_match_wide_because_escalation_re_seeds_the_mirror`.
"""

from __future__ import annotations

import json

import pytest

from src.mcp._replay_log import ReplayMismatchError, parse_wire_log
from src.mcp._replay_verify import verify_escalation_budget
from src.mcp.wire_replay import replay_match
from src.utils.config_loader import load_config
from tests.unit._synthetic_wire import synth_session


def _seeds(cfg) -> tuple[list[int], list[int], int]:
    """Return (pair seeds, spare seeds, the voids one escalation costs)."""
    pairs = int(cfg["game"]["num_games"]) // 2
    all_seeds = [int(s) for s in cfg["wire_match"]["seeds"]]
    return all_seeds[:pairs], all_seeds[pairs:], int(cfg["wire_match"]["max_void_replays"])


def test_a_clean_match_owes_nothing() -> None:
    """Baseline: no spare resolved, no voids logged, no complaint."""
    cfg = load_config()
    pair, _spares, _needed = _seeds(cfg)
    assert verify_escalation_budget(cfg, pair, 0) == 0


def test_a_spare_with_no_voids_is_rejected() -> None:
    """The seed-shopping case: the spare was really played, but nothing earned it."""
    cfg = load_config()
    pair, spares, _needed = _seeds(cfg)
    with pytest.raises(ReplayMismatchError, match="never paid for"):
        verify_escalation_budget(cfg, [*pair[1:], spares[0]], 0)


def test_a_spare_with_too_few_voids_is_rejected() -> None:
    """One short of the threshold is not an escalation — the bound is exact."""
    cfg = load_config()
    pair, spares, needed = _seeds(cfg)
    with pytest.raises(ReplayMismatchError, match="void re-hello"):
        verify_escalation_budget(cfg, [*pair[1:], spares[0]], needed - 1)


def test_a_spare_backed_by_its_escalation_is_accepted() -> None:
    """A genuine P7 escalation must still replay — the guard blocks fraud, not the rule."""
    cfg = load_config()
    pair, spares, needed = _seeds(cfg)
    assert verify_escalation_budget(cfg, [*pair[1:], spares[0]], needed) == 1


def test_each_spare_is_billed_separately() -> None:
    """Two spares cost two escalations; one escalation's worth of voids does not cover both."""
    cfg = load_config()
    pair, spares, needed = _seeds(cfg)
    played = [pair[2], spares[0], spares[1]]
    with pytest.raises(ReplayMismatchError, match="2 SPARE seed"):
        verify_escalation_budget(cfg, played, 2 * needed - 1)
    assert verify_escalation_budget(cfg, played, 2 * needed) == 2


def test_the_same_spare_twice_is_billed_once() -> None:
    """A spare re-seeds the whole PAIR, so both mirror halves share ONE escalation."""
    cfg = load_config()
    _pair, spares, needed = _seeds(cfg)
    assert verify_escalation_budget(cfg, [spares[0], spares[0]], needed) == 1


def test_the_budget_is_match_wide_because_escalation_re_seeds_the_mirror(tmp_path) -> None:
    """REGRESSION: a per-session rule rejects an HONEST log, and the first fix shipped one.

    P7 escalation re-seeds the pair k/k+3 and re-queues an already-played base game under
    the new seed. That re-queued session shows a spare seed with a single re-hello of its
    own, while the three voids that bought the spare sit in its SIBLING. Billing each
    session for its own escalation fails a match no honest referee could have played
    differently — so the budget is summed across the log.
    """
    cfg = load_config()
    _pair, spares, needed = _seeds(cfg)
    lines, records = [], []
    for sid, voids in (("sg-0", needed), ("sg-3", 0)):  # gid 1 and its mirror gid 4
        game_lines, record = synth_session(cfg, sid, spares[0], result_event=True, voids=voids)
        lines += game_lines
        records.append(record)
    log = tmp_path / "wire_log_mirror.jsonl"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    recs = tmp_path / "records_mirror.json"
    recs.write_text(json.dumps({"sub_games": records}), encoding="utf-8")

    replays = replay_match(cfg, log, recs, full_match=False)  # must NOT raise
    assert [r["seed"] for r in replays] == [spares[0], spares[0]]


def test_the_parser_counts_void_re_hellos(tmp_path) -> None:
    """The count the budget spends must come from the log, not be assumed."""
    cfg = load_config()
    _pair, spares, needed = _seeds(cfg)
    lines, _record = synth_session(cfg, "sg-0", spares[0], result_event=True, voids=needed)
    log = tmp_path / "wire_log_voids.jsonl"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert parse_wire_log(log)["sg-0"]["voids"] == needed
