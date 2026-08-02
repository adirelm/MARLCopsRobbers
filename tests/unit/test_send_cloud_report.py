"""The §5.3-cloud §3.5 report path — assembled from the RECORDED distributed match."""

from __future__ import annotations

import datetime
import inspect
import json
from pathlib import Path

import pytest

import scripts.send_cloud_report as mod
from scripts.send_cloud_report import build_cloud_report, main
from src.sdk.sdk import MarlSDK


def test_cloud_report_is_assembled_and_validates(cfg) -> None:
    """The recorded cloud match must assemble into a §3.5-valid body (no hand-stitching)."""
    report = build_cloud_report(cfg)
    assert len(report["sub_games"]) == int(cfg["game"]["num_games"])
    assert report["timezone"] == cfg["project"]["timezone"]
    assert report["totals"] == {
        role: sum(g["scores"][role] for g in report["sub_games"]) for role in ("cop", "thief")
    }


def test_cloud_report_carries_the_real_distributed_timings(cfg) -> None:
    """The point of this path: the emailed body must show the internet match's wall-clock.

    A fresh in-memory match finishes in well under a second; the recorded cloud run spans
    minutes. If this ever collapses to sub-second, the script is reporting the wrong match.
    """
    games = build_cloud_report(cfg)["sub_games"]
    span = datetime.datetime.fromisoformat(games[-1]["end"]) - datetime.datetime.fromisoformat(
        games[0]["start"]
    )
    assert span.total_seconds() > 60, "cloud report lost its distributed timings"


def test_cloud_report_matches_the_committed_record(cfg) -> None:
    """Every score/winner/move must come from the tracked record — only identities differ."""
    recorded = json.loads(Path("results/subgames/cloud_match_5x5.redacted.json").read_text(encoding="utf-8"))
    built = build_cloud_report(cfg)
    for got, want in zip(built["sub_games"], recorded["sub_games"], strict=True):
        assert (got["moves"], got["winner"], got["scores"]) == (
            want["moves"],
            want["winner"],
            want["scores"],
        )
    assert built["totals"] == recorded["totals"]


def test_main_without_send_performs_no_egress(cfg, capsys) -> None:
    """The default run is report-only — the lecturer send stays behind an explicit --send."""
    report = main(cfg, send=False)
    assert report["totals"]["cop"] + report["totals"]["thief"] > 0
    assert "sub-games" in capsys.readouterr().out


def test_missing_record_fails_loudly(cfg, tmp_path) -> None:
    """A missing cloud record must raise, never silently fall back to an empty report."""
    with pytest.raises(FileNotFoundError):
        build_cloud_report(cfg, tmp_path / "absent.json")


def test_send_binds_date_str_by_keyword_not_position(cfg, monkeypatch) -> None:
    """REGRESSION: ``send_final_report``'s 2nd positional is ``sender``, NOT ``date_str``.

    Passing the date positionally silently injects it as the mailer, so this asserts the
    real signature binding rather than just "a send happened".
    """
    assert list(inspect.signature(MarlSDK.send_final_report).parameters)[2] == "sender"

    seen: dict = {}

    class _SpySDK:
        def __init__(self, _cfg: dict) -> None: ...

        def send_final_report(self, report: dict, sender: object = None, date_str=None) -> dict:
            seen.update(report=report, sender=sender, date_str=date_str)
            return {"sent": True, "to": "spy"}

    monkeypatch.setattr(mod, "MarlSDK", _SpySDK)
    mod.main(cfg, send=True)

    assert seen["sender"] is None, "date_str leaked into the `sender` slot"
    assert seen["date_str"] and len(seen["date_str"]) == len("YYYY-MM-DD")
    assert seen["report"]["totals"]
