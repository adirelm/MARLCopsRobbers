"""The §9.3 replay must prove the referee WITHHELD what P5 says it must.

Gap this closes: the verifier compared each tick's `your_pos` and `barriers_left` but
ignored `opponent_pos` and the masked `barriers`. A log in which the referee fed its own
agent full board visibility — the exact cheat the shared log exists to make impossible —
replayed perfectly clean. We had told the partner the log makes every transition
independently re-derivable, so the gap sat under a fairness claim we had already made.
"""

from __future__ import annotations

import json

import pytest

from src.mcp._replay_log import ReplayMismatchError, select_log_and_records
from src.mcp.wire_replay import replay_match
from src.utils.config_loader import load_config


@pytest.fixture
def real_pair():
    cfg = load_config()
    return cfg, *select_log_and_records(cfg)


def test_the_committed_match_log_passes_masking_verification(real_pair) -> None:
    """The real §9 match must verify clean — masking included, not just positions."""
    cfg, log, records = real_pair
    assert len(replay_match(cfg, log, records)) == int(cfg["game"]["num_games"])


def test_a_leaked_opponent_position_is_caught(real_pair, tmp_path) -> None:
    """Tamper: reveal the opponent on a tick where P5 requires null. Must raise."""
    cfg, log, records = real_pair
    lines, patched = log.read_text(encoding="utf-8").splitlines(), False
    out = []
    for line in lines:
        entry = json.loads(line) if line.strip() else None
        if (
            not patched
            and entry
            and entry.get("direction") == "request"
            and entry.get("url", "").endswith("/request_move")
            and entry["payload"].get("opponent_pos") is None
        ):
            entry["payload"]["opponent_pos"] = [0, 0]  # a referee leaking full visibility
            out.append(json.dumps(entry))
            patched = True
            continue
        out.append(line)
    assert patched, "no masked tick found to tamper with — the fixture assumption broke"

    tampered = tmp_path / "leaked.jsonl"
    tampered.write_text("\n".join(out) + "\n", encoding="utf-8")
    with pytest.raises(ReplayMismatchError, match="P5 masking violated"):
        replay_match(cfg, tampered, records)


def test_a_widened_barrier_report_is_caught(real_pair, tmp_path) -> None:
    """Tamper the OTHER masking field too, so one check cannot cover for the other."""
    cfg, log, records = real_pair
    out, patched = [], False
    for line in log.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line) if line.strip() else None
        if (
            not patched
            and entry
            and entry.get("direction") == "request"
            and entry.get("url", "").endswith("/request_move")
        ):
            entry["payload"]["barriers"] = [[4, 4]]  # a cell no honest mask would report
            out.append(json.dumps(entry))
            patched = True
            continue
        out.append(line)
    assert patched

    tampered = tmp_path / "widened.jsonl"
    tampered.write_text("\n".join(out) + "\n", encoding="utf-8")
    with pytest.raises(ReplayMismatchError, match="P5 masking violated"):
        replay_match(cfg, tampered, records)
