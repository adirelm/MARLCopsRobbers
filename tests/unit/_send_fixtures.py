"""Shared fixtures for the §3.5 send-path tests (report body + tmp-redirected config).

Lives in a helper module so ``test_report_send`` and ``test_send_guards`` each stay
well under the 150-LOC gate while sharing one report builder (DRY). The body uses
PLACEHOLDER identity only and Table-1-consistent scores (winner cop -> 20/5).
"""

from __future__ import annotations

import json
from pathlib import Path


def make_report() -> dict:
    """Return a valid 6-sub-game §3.5 body (placeholder PII, Table-1 scores)."""
    games = [
        {
            "id": i + 1,
            "start": f"2026-06-17T18:0{i}:00.000+03:00",
            "end": f"2026-06-17T18:0{i}:30.000+03:00",
            "moves": 4,
            "winner": "cop",
            "scores": {"cop": 20, "thief": 5},
        }
        for i in range(6)
    ]
    return {
        "group_name": "adrl-001",
        "students": [{"role": "A", "full_name": "Pat Doe", "id": "000000000"}],
        "github_repo": "https://github.com/example/marl",
        "timezone": "Asia/Jerusalem",
        "sub_games": games,
        "totals": {"cop": 120, "thief": 30},
    }


def cfg_tmp(cfg: dict, tmp_path: Path) -> dict:
    """Deep-copy ``cfg`` with the sentinel + output dir redirected into ``tmp_path``."""
    cfg = json.loads(json.dumps(cfg))
    cfg["gmail"]["sentinel"] = str(tmp_path / ".report_sent")
    cfg["gmail"]["output_dir"] = str(tmp_path / "reports")
    return cfg
