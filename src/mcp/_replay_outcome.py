"""Re-derive the §9.4 body's PUBLISHED OUTCOME on the replay path.

``totals_by_group`` and ``bonus_claim`` are the two numbers the whole bonus is about, and
until now nothing on the replay path recomputed them: a body could publish 105-0, or a claim
that contradicted its own totals, and ``replay_wire_match.py`` still reported every sub-game
verified. The SEND path was protected (``bonus_send`` calls ``validate_bonus``), but the
replay is the command the README offers a grader and the opposing group as §9.3 evidence,
and it is the one that runs against the artifact a fresh clone has.

Why not just call ``validate_bonus`` here: it requires the full identity blocks, so it
raises ``missing required key 'full_name'`` on the TRACKED redacted body — the exact
artifact this path exists to check. This module is the redaction-tolerant half: it re-derives
from ``sub_games`` alone and never looks at a student block.
"""

from __future__ import annotations

from src.mcp._replay_log import ReplayMismatchError
from src.reporting.bonus import derive_bonus_claim, derive_totals_by_group


def verify_published_outcome(cfg: dict, published: dict, group_names: tuple[str, str]) -> None:
    """Recompute ``totals_by_group`` and ``bonus_claim`` from the sub-games and compare.

    Both are DERIVED fields, so they are never read as given — the same rule §3.5 already
    applies to the match report. ``group_names`` comes from the frozen agreement (config),
    not from the body, so a body cannot pass by renaming the parties it scores.

    Raises:
        ReplayMismatchError: When either published field disagrees with its derivation.
    """
    group_1, group_2 = group_names
    want_totals = derive_totals_by_group(published["sub_games"], group_1, group_2)
    got_totals = {str(k): int(v) for k, v in (published.get("totals_by_group") or {}).items()}
    if got_totals != want_totals:
        raise ReplayMismatchError(
            f"totals_by_group {got_totals} != the totals derived from the sub-games "
            f"{want_totals} — the published margin is not the one the match produced"
        )
    want_claim = derive_bonus_claim(want_totals, cfg["game"]["bonus_claim"])
    got_claim = {str(k): int(v) for k, v in (published.get("bonus_claim") or {}).items()}
    if got_claim != want_claim:
        raise ReplayMismatchError(
            f"bonus_claim {got_claim} != the §9.2 claim derived from the totals {want_claim}"
        )


def verify_body_metadata(published: dict) -> None:
    """Check the §9.4 envelope fields that carry meaning but no arithmetic.

    ``report_type`` and per-sub-game timestamp ORDER were unchecked anywhere on this path: a
    body could declare itself something other than a bonus game, or claim a sub-game that
    ended before it started. Cheap to check, and their absence from the checks was only ever
    an oversight.

    ``mutual_agreement`` is deliberately NOT asserted True here — a draft legitimately
    carries False until the §9.3 byte-compare completes, and the SEND path is what refuses
    to email a False one.

    Raises:
        ReplayMismatchError: On a wrong report_type or an inverted timestamp pair.
    """
    if (kind := published.get("report_type")) != "bonus_game":
        raise ReplayMismatchError(f"report_type {kind!r} != 'bonus_game' — this is not a §9.4 body")
    for game in published["sub_games"]:
        if str(game["end"]) < str(game["start"]):  # ISO-8601 with offset sorts lexicographically
            raise ReplayMismatchError(
                f"sub-game {game['id']}: end {game['end']!r} precedes start {game['start']!r}"
            )
