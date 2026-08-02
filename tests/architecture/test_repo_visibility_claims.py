"""No tracked doc may still describe this repository as public.

The repo was flipped to PRIVATE so a §9 bonus opponent cannot train a best-response
exploiter against our published weights (ANALYSIS §13). That flip invalidated prose in
three separate files at once — ANALYSIS's risk paragraph ("our weights are in a PUBLIC
repo"), TODO's T0.8 status line, and README's self-grade note — none of which any test
covered, because each was a standalone sentence rather than a number.

This pins the DECISION, not the wording: any doc may discuss visibility freely, but none
may assert this repo is public, and none may still call the lecturer's read access
pending (it was accepted before submission). Rewording stays free; contradicting the
decision does not.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DOCS = ("README.md", "docs/ANALYSIS.md", "docs/TODO.md", "docs/PLAN.md", "docs/PRD.md")

# Claims that a repo is public. Deliberately narrow: "the endpoints are publicly reachable"
# (about the Render servers, and true) must stay legal.
_PUBLIC_CLAIM = re.compile(
    r"repo(?:sitory)?\s+is\s+public|public\s+repo(?:sitory)?\b",
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
def test_no_doc_claims_this_repo_is_public(rel: str) -> None:
    """The repo is private; a doc saying otherwise misleads a grader reading it alone."""
    hits = _offending_lines(_PUBLIC_CLAIM, rel, exempt=_OPPONENT_REPO)
    assert not hits, "stale public-repo claim (the repo is PRIVATE):\n" + "\n".join(hits)


@pytest.mark.parametrize("rel", _DOCS)
def test_no_doc_leaves_the_lecturer_invite_pending(rel: str) -> None:
    """`rmisegal` accepted read access — a 'still pending' note reads as an unmet action."""
    hits = _offending_lines(_PENDING_COLLAB, rel)
    assert not hits, "lecturer access is ACCEPTED, not pending:\n" + "\n".join(hits)


def test_the_guard_detects_a_reintroduced_claim(tmp_path: Path) -> None:
    """Mutation check — the patterns must actually fire on the exact prose we removed."""
    assert _PUBLIC_CLAIM.search("our weights are in a PUBLIC repo, so a targeted exploiter")
    assert _PUBLIC_CLAIM.search("No numeric self-grade is claimed in this public repo")
    assert _PUBLIC_CLAIM.search("branch; the repo is PUBLIC; adding")
    assert _PENDING_COLLAB.search(
        "`rmisegal`) as read collaborator is a pre-submission user action, still pending"
    )
    # ...and must NOT fire on the legitimate statement about the Render endpoints,
    # nor on ANALYSIS §13's reasoning about the OPPONENT's repo being public.
    assert not _PUBLIC_CLAIM.search("the endpoints are publicly reachable, auth is in OUR app")
    assert _OPPONENT_REPO.search("symmetrically: if their repo is public we can exploit them")
