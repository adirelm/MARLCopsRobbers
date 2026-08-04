"""Committed run-log integrity — the data behind every quoted README/ANALYSIS number.

Guards the finding class "silently duplicated appends narrow the SE bands": each
``results/runs/*.jsonl`` must hold NO byte-identical duplicate lines, and the three
headline final mean±SE values quoted in README §7.3 / ANALYSIS §11 must recompute
from the committed log. Tests SKIP when a log is absent (heavy artifacts may be
git-ignored on a fresh clone — never assert an ignored artifact exists).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.results.aggregate import final_by_algorithm, load_runs
from src.utils.config_loader import load_config

_RUNS_DIR = Path(__file__).resolve().parents[2] / "results" / "runs"
_FOCUS_STAGE = 2  # the 4x4 two-cop comparison focus (README §7.3)
_LAST_K = int(load_config()["results"]["final_window_rounds"])  # the published averaging window
# The doc-quoted headline stats: algorithm -> (mean, seed-level SE), 3 decimals.
_QUOTED = {"iql": (0.816, 0.010), "vdn": (0.845, 0.016), "qmix": (0.628, 0.102)}


@pytest.mark.parametrize("name", ["history.jsonl", "returns_history.jsonl"])
def test_run_log_has_no_duplicate_lines(name):
    path = _RUNS_DIR / name
    if not path.exists():
        pytest.skip(f"{name} not present (git-ignored heavy artifact)")
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == len(set(lines)), f"{name} holds duplicated appended records"


def test_readme_final_stats_recompute_from_the_committed_log():
    path = _RUNS_DIR / "history.jsonl"
    if not path.exists():
        pytest.skip("history.jsonl not present (git-ignored heavy artifact)")
    stats = final_by_algorithm(load_runs(path), "capture_rate", _FOCUS_STAGE, _LAST_K)
    for algorithm, (mean, se) in _QUOTED.items():
        assert stats[algorithm][0] == pytest.approx(mean, abs=5e-4)
        assert stats[algorithm][1] == pytest.approx(se, abs=5e-4)
