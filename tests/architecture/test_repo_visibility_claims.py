"""No tracked doc may describe this repository's visibility incorrectly.

Visibility has flipped twice and both flips silently invalidated prose. It went PRIVATE for
the §9 window so no opponent could train a best-response against our published weights
(ANALYSIS §13), which falsified three files at once; it went PUBLIC again on 2026-08-04
once the match was played and agreed, which falsified the corrections.

The guard therefore tracks the CURRENT state via `EXPECTED_VISIBILITY` rather than
hard-coding one direction. Note the earlier version forbade the string "repo is public" and
kept passing after the flip only because markdown bold sat between "is" and "public" — a
pattern that matches prose is only as good as the prose it happens to see, so the assertion
now names the state explicitly and the mutation test covers BOTH spellings.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DOCS = ("README.md", "docs/ANALYSIS.md", "docs/TODO.md", "docs/PLAN.md", "docs/PRD.md")

EXPECTED_VISIBILITY = "public"  # flipped back 2026-08-04, after the §9 match was agreed

# Markdown emphasis may sit anywhere inside the claim ("the repo is **PUBLIC**"), so the
# separator is permissive; the earlier version missed exactly that and passed a false doc.
_GAP = r"[\s*_`]+"  # whitespace possibly wrapped in markdown emphasis
_WRONG = "private" if EXPECTED_VISIBILITY == "public" else "public"
_WRONG_CLAIM = re.compile(
    rf"repo(?:sitory)?{_GAP}is{_GAP}{_WRONG}\b|{_WRONG}{_GAP}repo(?:sitory)?\b",
    re.IGNORECASE,
)

# ANALYSIS §13 legitimately reasons about the OPPONENT's repo being public (that is the
# asymmetry that decides the §9 exploiter risk), so those lines are exempt.
_OPPONENT_REPO = re.compile(r"\b(?:their|opponent'?s?|partner'?s?)\s+(?:\w+\s+){0,2}repo", re.I)

# The lecturer's read access is accepted, not pending — verified via the GitHub API.
_PENDING_COLLAB = re.compile(r"rmisegal[^.\n]{0,120}?\b(?:still )?pending", re.IGNORECASE)


def _offending_lines(
    pattern: re.Pattern[str], rel: str, *, exempt: re.Pattern[str] | None = None
) -> list[str]:
    """Return ``rel:line: text`` for each line matching ``pattern`` and not ``exempt``."""
    hits = []
    for number, line in enumerate((_ROOT / rel).read_text(encoding="utf-8").splitlines(), start=1):
        if pattern.search(line) and not (exempt and exempt.search(line)):
            hits.append(f"{rel}:{number}: {line.strip()}")
    return hits


@pytest.mark.parametrize("rel", _DOCS)
def test_no_doc_states_the_wrong_visibility(rel: str) -> None:
    """A doc naming the opposite visibility misleads a grader reading that file alone."""
    hits = _offending_lines(_WRONG_CLAIM, rel, exempt=_OPPONENT_REPO)
    assert not hits, f"doc says {_WRONG!r} but the repo is {EXPECTED_VISIBILITY!r}:\n" + "\n".join(hits)


@pytest.mark.parametrize("rel", _DOCS)
def test_no_doc_leaves_the_lecturer_invite_pending(rel: str) -> None:
    """`rmisegal` accepted read access — a 'still pending' note reads as an unmet action."""
    hits = _offending_lines(_PENDING_COLLAB, rel)
    assert not hits, "lecturer access is ACCEPTED, not pending:\n" + "\n".join(hits)


def test_the_guard_detects_a_reintroduced_claim(tmp_path: Path) -> None:
    """Mutation check — must fire on the wrong-visibility prose, INCLUDING the bold spelling.

    The bold case is the one that matters: the previous guard forbade "repo is public" and
    kept passing on "the repo is **PUBLIC**" because the emphasis broke the word gap. A
    pattern over prose is only as good as the spellings it actually sees.
    """
    assert _WRONG_CLAIM.search("the repo is private")
    assert _WRONG_CLAIM.search("the repo is **PRIVATE**"), "markdown bold must not defeat it"
    assert _WRONG_CLAIM.search("kept in a private repo for the §9 window")
    assert _PENDING_COLLAB.search(
        "`rmisegal`) as read collaborator is a pre-submission user action, still pending"
    )
    # ...and must NOT fire on prose stating the CURRENT (correct) visibility.
    assert not _WRONG_CLAIM.search("the repo is **PUBLIC** again since 2026-08-04")
    assert not _WRONG_CLAIM.search("the endpoints are publicly reachable, auth is in OUR app")
    assert _OPPONENT_REPO.search("symmetrically: if their repo is private we cannot exploit them")
