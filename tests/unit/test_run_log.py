"""run_log tests (T10.1) — per-round records + append + resumable done_runs (fake sdk)."""

from __future__ import annotations

from src.results.run_log import append_records, done_runs, history_records, run_and_log


def _history():
    return [
        {"round": 0, "role": "cop", "loss": 0.5, "capture_rate": 0.2},
        {"round": 1, "role": "thief", "loss": 0.4, "capture_rate": 0.3},
    ]


def test_history_records_carries_run_keys_and_metrics(cfg):
    records = history_records(cfg, "qmix", 7, 0, _history())
    assert len(records) == 2
    assert records[0] == {
        "algorithm": "qmix",
        "seed": 7,
        "stage": 0,
        "grid": 2,
        "round": 0,
        "role": "cop",
        "loss": 0.5,
        "capture_rate": 0.2,
    }


def test_append_and_done_runs_roundtrip(tmp_path, cfg):
    out = tmp_path / "history.jsonl"
    append_records(out, history_records(cfg, "qmix", 7, 0, _history()))
    append_records(out, history_records(cfg, "iql", 17, 3, _history()))
    out.write_text(out.read_text(encoding="utf-8") + "\n", encoding="utf-8")  # a blank line is skipped
    assert done_runs(out) == {("qmix", 7, 0), ("iql", 17, 3)}
    assert done_runs(tmp_path / "nope.jsonl") == set()


def test_run_and_log_uses_sdk_and_appends(tmp_path, cfg):
    out = tmp_path / "h.jsonl"

    class _SDK:
        def train(self, algorithm, seed, stage_idx):
            return _history()

    records = run_and_log(_SDK(), cfg, "vdn", 37, 1, out)
    assert len(records) == 2 and records[0]["algorithm"] == "vdn"
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2
