"""Unit tests for src/marl/env/scorer.py (T1.4) — REPORT-ONLY scoreboard.

Covers the §3.4 Table-1 totals (capture 20/5, timeout 5/10) and the
architecture invariant that the Scorer is a DISTINCT module whose output is
never consumed by the RL reward path (reward.py must not import scorer.py).
"""

from __future__ import annotations

import ast
import inspect

import pytest

from src.marl.env import reward as reward_mod
from src.marl.env.reward import RewardModel
from src.marl.env.scorer import Scorer


def test_capture_scoreboard_totals(cfg):
    # Capture => cop wins (20), thief is captured/loses (5).
    out = Scorer(cfg).score("cop")
    assert out["cop"] == cfg["game"]["scoring"]["cop_win"] == 20
    assert out["thief"] == cfg["game"]["scoring"]["thief_loss"] == 5


def test_timeout_scoreboard_totals(cfg):
    # Timeout => thief evades/wins (10), cop loses (5).
    out = Scorer(cfg).score("thief")
    assert out["cop"] == cfg["game"]["scoring"]["cop_loss"] == 5
    assert out["thief"] == cfg["game"]["scoring"]["thief_win"] == 10


def test_scorer_returns_ints(cfg):
    out = Scorer(cfg).score("cop")
    assert isinstance(out["cop"], int)
    assert isinstance(out["thief"], int)
    # bool is an int subclass, so isinstance(True, int) is True; assert the
    # values are real ints (not bool) so the JSON report serializes 20/5, not
    # true/false.
    assert isinstance(out["cop"], bool) is False
    assert isinstance(out["thief"], bool) is False


@pytest.mark.parametrize("bad_winner", ["draw", ""])
def test_scorer_rejects_unknown_winner(cfg, bad_winner):
    # Only "cop"/"thief" are valid; anything else must fail loudly, never
    # silently produce a (potentially emailed) bogus scoreboard.
    with pytest.raises(ValueError):
        Scorer(cfg).score(bad_winner)


def test_scorer_distinct_from_reward_model():
    assert Scorer is not RewardModel
    assert Scorer.__module__ != RewardModel.__module__


def _imported_modules(module) -> set[str]:
    """Return every dotted module name imported by ``module`` (AST-level)."""
    tree = ast.parse(inspect.getsource(module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_reward_module_does_not_import_scorer():
    # Architecture invariant: the RL training signal never imports the scoreboard.
    imports = _imported_modules(reward_mod)
    assert not any("scorer" in name.lower() for name in imports)
    assert not hasattr(reward_mod, "Scorer")
