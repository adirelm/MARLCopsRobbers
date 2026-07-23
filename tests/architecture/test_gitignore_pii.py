"""Pin the PII .gitignore invariant: ``*.real.json`` is ignored EVERYWHERE.

The un-redacted report copies carry real identities (A1-A5 PII deny-list). Before
this pin only ``results/reports/*.real.json`` was ignored, so a stray un-redacted
copy anywhere else in the tree was committable. ``git check-ignore --no-index``
evaluates the ignore RULES alone (hypothetical paths; nothing must exist), so this
never asserts a git-ignored PII artifact exists — only that the rule would catch it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _ignored(path: str) -> bool:
    """True when the .gitignore rules (index-independent) would ignore ``path``."""
    result = subprocess.run(  # fixed argv, no shell
        ["git", "check-ignore", "--no-index", "-q", path],
        cwd=_ROOT,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


@pytest.mark.parametrize(
    "path",
    [
        "results/reports/bonus_draft.real.json",  # the original (pre-broadening) location
        "docs/notes.real.json",  # anywhere else in the tree
        "stray.real.json",  # repo root
    ],
)
def test_real_json_is_ignored_everywhere(path: str):
    assert _ignored(path), f"{path} must be git-ignored (un-redacted reports are PII-bearing)"


def test_rehearsal_placeholder_is_not_caught_by_the_real_json_rule():
    placeholder = "results/reports/bonus_draft.rehearsal.placeholder.json"
    assert not _ignored(placeholder), f"{placeholder} is the TRACKED redacted placeholder"
