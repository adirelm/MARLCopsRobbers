"""V3 §8.4 uv-only gate — no bare ``python -m`` / ``pip`` / ``venv`` anywhere tracked.

The guideline is explicit that the ban covers **code, scripts, CI/CD AND documentation**
("אין קריאות ישירות ל-pip או python -m בקוד, סקריפטים, CI/CD או תיעוד"), and §19.1's
quick-reference card marks the package-manager rule as an AUTOMATED check — so it gets one.
The 2026-07-30 audit found 11 real hits (7 README generator cells, a dead docs/TODO command,
and three source docstrings) that six earlier review rounds missed because every gate looked
at ``src/tests/scripts`` code, never at documentation prose.

A line that RESTATES the prohibition (e.g. the guideline crib sheet) is not a violation, so
lines carrying a negation marker next to the term are exempt.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]

# Bare invocations the guideline forbids. `python -m` is legal ONLY as `uv run python -m`.
_FORBIDDEN = {
    "python -m": re.compile(r"(?<!uv run )python -m"),
    "pip install": re.compile(r"\bpip install\b"),
    "python -m venv": re.compile(r"\bpython -m venv\b"),
}
# A line that forbids/negates the term is documentation OF the rule, not a breach of it.
_NEGATION = re.compile(
    r"\b(no|not|never|avoid|forbid|forbidden|instead of|NOT)\b|אין|אסור|ללא", re.IGNORECASE
)
_SKIP_SUFFIX = (".png", ".pt", ".jsonl", ".lock", ".ipynb", ".pdf", ".docx")
# This file DEFINES the forbidden literals, so it must exempt itself: a guard that spells out
# what it forbids fails against itself (the same trap the PII host-path gate hit on 2026-07-22).
_SELF = Path(__file__).relative_to(_ROOT).as_posix()


def _tracked_text_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    return [p for p in out if not p.endswith(_SKIP_SUFFIX) and p != _SELF]


@pytest.mark.parametrize("term", sorted(_FORBIDDEN))
def test_no_bare_uv_bypassing_invocation_in_tracked_content(term: str) -> None:
    """Every tool invocation in code AND docs must go through ``uv run`` (V3 §8.4)."""
    pattern = _FORBIDDEN[term]
    offenders: list[str] = []
    for rel in _tracked_text_files():
        try:
            text = (_ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line) and not _NEGATION.search(line):
                offenders.append(f"{rel}:{number}")
    assert not offenders, (
        f"V3 §8.4 forbids bare `{term}` in code, scripts, CI/CD or documentation — "
        f"prefix with `uv run`: " + "; ".join(offenders[:12])
    )
