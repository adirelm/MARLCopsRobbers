"""Import-boundary exit-gate (K3 / FR-ALG-5) — the runtime/exec path stays clean.

The decentralized EXECUTION path (the env + the agent nets present at P4) must
NEVER import the CTDE TRAIN-ONLY machinery: the learners, the mixers, the
centralized episode buffer, or the cross-cutting ``services`` layer. The global
state ``GlobalState`` is the env's OWN type (defined in ``src.marl.env.types``),
so the env using it is not a leak — but ``src.marl.nets`` (pure execution nets)
must not reach for it at all.

This is a STATIC source scan with TEETH: every module under ``src/marl/env`` and
``src/marl/nets`` is parsed with :mod:`ast`, every ``import`` / ``from`` target
collected, and asserted to miss every forbidden prefix. The MCP/runtime half of
this boundary is EXTENDED in P5 (the deployed actor); this P4b test is scoped to
the modules that exist now and does NOT assert any nonexistent P5 module. The
negative-control tests plant a bad import in a synthetic source string and prove
the scanner catches both an ``import`` and a ``from ... import`` leak.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Train-only module prefixes the execution path must never import.
_FORBIDDEN = (
    "src.marl.learner",
    "src.marl.mixers",
    "src.marl.replay.episode_buffer",
    "src.services",
)
# GlobalState is the env's own type; nets must not reach for it.
_FORBIDDEN_NAMES = {"GlobalState"}
_SRC = Path(__file__).resolve().parents[2] / "src" / "marl"


def _imported_modules(source: str) -> set[str]:
    """Return every module path imported by ``source`` (``import`` + ``from``)."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            found.add(node.module)
    return found


def _imported_names(source: str) -> set[str]:
    """Return every bare name pulled in via ``from ... import name``."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
    return names


def _violations(source: str, *, forbid_names: bool) -> list[str]:
    """Return the forbidden module prefixes (+ names) imported by ``source``."""
    hits = [m for m in _imported_modules(source) for p in _FORBIDDEN if m == p or m.startswith(p + ".")]
    if forbid_names:
        hits += sorted(_imported_names(source) & _FORBIDDEN_NAMES)
    return hits


def _runtime_modules() -> list[tuple[Path, bool]]:
    """Yield each (module, forbid_names) under env/ (no names) + nets/ (names too)."""
    env = [(p, False) for p in (_SRC / "env").rglob("*.py")]
    nets = [(p, True) for p in (_SRC / "nets").rglob("*.py")]
    return env + nets


@pytest.mark.parametrize("path,forbid_names", _runtime_modules())
def test_runtime_module_imports_no_train_only(path: Path, forbid_names: bool) -> None:
    """Every env/nets module imports NONE of the train-only machinery."""
    violations = _violations(path.read_text(encoding="utf-8"), forbid_names=forbid_names)
    assert not violations, f"{path.name} imports train-only: {violations}"


def test_scan_finds_modules() -> None:
    """Sanity: the scan actually covers the real env + nets modules (not empty)."""
    mods = _runtime_modules()
    assert len(mods) >= 2  # at least cops_robbers_env.py + agent_net.py
    assert any(p.name == "cops_robbers_env.py" for p, _ in mods)
    assert any(p.name == "agent_net.py" for p, _ in mods)


def test_scanner_catches_planted_module_import() -> None:
    """Negative control: a planted ``import src.marl.mixers...`` is caught."""
    bad = "import src.marl.mixers.vdn_mixer\nx = 1\n"
    assert _violations(bad, forbid_names=False)


def test_scanner_catches_planted_from_import() -> None:
    """Negative control: a planted ``from src.marl.learner ...`` is caught."""
    bad = "from src.marl.learner.learner_base import QmixLearner\n"
    assert _violations(bad, forbid_names=False)


def test_scanner_catches_planted_globalstate_in_nets() -> None:
    """Negative control: a nets module importing GlobalState is caught."""
    bad = "from src.marl.env.types import GlobalState\n"
    assert _violations(bad, forbid_names=True)


def test_clean_source_has_no_violations() -> None:
    """A clean execution-path import (torch + dims) raises no false positive."""
    clean = "import torch\nfrom src.marl.nets.dims import obs_dim\n"
    assert not _violations(clean, forbid_names=True)
