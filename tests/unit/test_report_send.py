"""Report send pipeline tests (T9.3/9.5) — subject/redact/hash/idempotency, no live send."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.api.gatekeeper import DEFERRED
from src.reporting import send as send_mod
from src.reporting.mailer import FakeEmailSender
from src.reporting.schema import validate
from src.sdk.sdk import MarlSDK


def test_validate_enforces_expected_game_count():
    full = _report()  # 6 sub-games
    validate(full, expected_games=6)  # the §3.5 count -> passes
    short = {
        **full,
        "num_games": 5,
        "sub_games": full["sub_games"][:5],
        "totals": {"cop": 100, "thief": 25},  # 5 cop-wins: 20*5 / 5*5
    }
    validate(short)  # structural-only (no count) -> passes
    with pytest.raises(ValueError, match="exactly 6"):
        validate(short, expected_games=6)  # the send-path gate rejects a short report


def _report() -> dict:
    games = [
        {
            "game_id": f"g{i}",
            "grid": [5, 5],
            "winner": "cop",
            "capture": True,
            "steps": 4,
            "scores": {"cop": 20, "thief": 5},
            "seed": i,
        }
        for i in range(6)
    ]
    return {
        "group": "adrl-001",
        "students": [{"full_name": "Pat Doe", "id": "000000000"}],
        "num_games": 6,
        "sub_games": games,
        "totals": {"cop": 120, "thief": 30},
    }


def _cfg_tmp(cfg: dict, tmp_path: Path) -> dict:
    cfg = json.loads(json.dumps(cfg))  # deep copy so we can redirect the sentinel/out dir
    cfg["gmail"]["sentinel"] = str(tmp_path / ".report_sent")
    cfg["gmail"]["output_dir"] = str(tmp_path / "reports")
    return cfg


def test_format_subject_has_no_pii_and_fills_totals(cfg):
    subject = send_mod.format_subject(cfg, _report(), "2026-06-21")
    assert "Cop 120 - Thief 30" in subject
    assert "2026-06-21" in subject
    assert "Pat Doe" not in subject and "000000000" not in subject


def test_redact_report_strips_student_pii():
    redacted = send_mod.redact_report(_report())
    assert redacted["students"] == [{"role": "A"}]
    assert "full_name" not in redacted["students"][0]


def test_report_hash_is_stable():
    assert send_mod.report_hash(_report()) == send_mod.report_hash(_report())


def test_build_date_returns_iso_date(cfg):
    date_str = send_mod.build_date(cfg)
    assert len(date_str) == 10 and date_str[4] == "-" and date_str[7] == "-"


def test_send_report_sends_once_then_is_idempotent(cfg, tmp_path):
    cfg = _cfg_tmp(cfg, tmp_path)
    sender = FakeEmailSender()
    first = send_mod.send_report(cfg, _report(), sender, "2026-06-21")
    second = send_mod.send_report(cfg, _report(), sender, "2026-06-21")
    assert first["sent"] is True and first["to"] == cfg["gmail"]["to"]
    assert second["sent"] is False and second["reason"] == "already_sent"
    assert len(sender.sent) == 1
    redacted = json.loads(Path(first["redacted_path"]).read_text(encoding="utf-8"))
    assert redacted["students"] == [{"role": "A"}]


def test_send_report_redacted_copy_has_no_pii(cfg, tmp_path):
    cfg = _cfg_tmp(cfg, tmp_path)
    send_mod.send_report(cfg, _report(), FakeEmailSender(), "2026-06-21")
    text = (Path(cfg["gmail"]["output_dir"]) / "match_report.redacted.json").read_text(encoding="utf-8")
    assert "Pat Doe" not in text and "000000000" not in text


class _QueueGate:
    """Models the real gatekeeper: execute() queues + defers; drain() runs the queue."""

    def __init__(self):
        self.queue = []

    def execute(self, channel, call):
        self.queue.append(call)
        return DEFERRED

    def drain(self):
        while self.queue:
            self.queue.pop(0)()


def test_send_report_deferral_stays_retryable_then_drain_commits_once(cfg, tmp_path):
    cfg = _cfg_tmp(cfg, tmp_path)
    sender = FakeEmailSender()
    gate = _QueueGate()
    out = send_mod.send_report(cfg, _report(), sender, "2026-06-21", gatekeeper=gate)
    # deferred: nothing sent yet, sentinel NOT written -> the call stays retryable
    assert out["sent"] is False and out["reason"] == "deferred"
    assert sender.sent == [] and not Path(cfg["gmail"]["sentinel"]).exists()
    # when the gatekeeper drains, the queued send fires AND records the sentinel together
    gate.drain()
    assert len(sender.sent) == 1
    assert Path(cfg["gmail"]["sentinel"]).exists()  # a later retry will now see already_sent


def test_inner_recheck_reports_already_sent_not_true(cfg, tmp_path):
    cfg = _cfg_tmp(cfg, tmp_path)
    sender = FakeEmailSender()
    report = _report()
    digest = send_mod.report_hash(report)

    class _PreMarkGate:
        # simulate a prior queued send draining (marking the sentinel) right before this thunk runs
        def execute(self, channel, call):
            send_mod.mark_sent(cfg["gmail"]["sentinel"], digest)
            return call()

    out = send_mod.send_report(cfg, report, sender, "2026-06-21", gatekeeper=_PreMarkGate())
    assert out["sent"] is False and out["reason"] == "already_sent"  # accurate status, not sent=True
    assert len(sender.sent) == 0  # the inner recheck prevented the duplicate send


def test_shared_gate_retry_during_deferral_still_sends_once(cfg, tmp_path):
    cfg = _cfg_tmp(cfg, tmp_path)
    sender = FakeEmailSender()
    gate = _QueueGate()  # a SHARED, exhausted gate: both calls defer (queue)
    report = _report()
    send_mod.send_report(cfg, report, sender, "2026-06-21", gatekeeper=gate)
    send_mod.send_report(cfg, report, sender, "2026-06-21", gatekeeper=gate)  # retry before drain
    assert len(gate.queue) == 2  # both queued (sentinel not yet written)
    gate.drain()  # runs both thunks back-to-back
    assert len(sender.sent) == 1  # the 2nd thunk rechecks the sentinel and skips -> no double-email


def test_sdk_send_final_report_dry_run(cfg, tmp_path):
    cfg = _cfg_tmp(cfg, tmp_path)
    sender = FakeEmailSender()
    out = MarlSDK(cfg).send_final_report(_report(), sender=sender, date_str="2026-06-21")
    assert out["sent"] is True and out["to"] == cfg["gmail"]["to"]
    assert len(sender.sent) == 1
