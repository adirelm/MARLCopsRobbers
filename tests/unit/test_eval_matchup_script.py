"""eval_matchup script flags — ``--base-seed`` override + ``--json-out`` artifact (ANALYSIS §12).

The committed evidence blocks ``results/matchup/block_1000.json`` / ``block_5000.json``
are produced by these flags; this pins that the override reaches the config-driven block
and that the JSON dump equals the returned arm summaries. The SDK is a test double —
the real arms are covered by ``tests/unit/test_matchup_eval.py``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "eval_matchup.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("eval_matchup_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeSDK:
    """Echoes the block config back so the override is observable per arm."""

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg

    def run_matchup_eval(self, thief_kind: str, thief_eps: float = 0.0) -> dict:
        block = self._cfg["matchup_eval"]
        return {
            "captures": 1,
            "games": block["n_games"],
            "moves": 5,
            "cop_actions": {"UP": 5},
            "thief_eps": thief_eps,
            "base_seed": block["base_seed"],
        }


def _cfg(eps_grid: list[float]) -> dict:
    return {"matchup_eval": {"n_games": 2, "base_seed": 1000, "thief_eps_grid": eps_grid}}


def test_base_seed_override_and_json_out_artifact(tmp_path, monkeypatch):
    module = _load_script()
    monkeypatch.setattr(module, "MarlSDK", _FakeSDK)
    out_path = tmp_path / "matchup" / "block.json"
    results = module.main(cfg=_cfg([0.0]), base_seed=5000, json_out=str(out_path))
    assert [arm["base_seed"] for arm in results] == [5000] * 3  # flee + random + one net arm
    assert json.loads(out_path.read_text(encoding="utf-8")) == results


def test_defaults_unchanged_without_flags(monkeypatch):
    module = _load_script()
    monkeypatch.setattr(module, "MarlSDK", _FakeSDK)
    results = module.main(cfg=_cfg([0.0, 0.1]))
    assert [arm["base_seed"] for arm in results] == [1000] * 4  # config block untouched
    assert [arm["thief_kind"] for arm in results] == ["flee", "random", "net", "net"]
