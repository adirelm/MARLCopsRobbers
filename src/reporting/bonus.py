"""§9 bonus-game report — assembly + validation (ex06 §9.1-9.4; FR-ANL-9, now BUILT).

The §9.4 body = the §3.5 sub-game fields KEPT (id/start/end/moves/winner/scores) plus
``cop_group``/``thief_group`` per sub-game, doubled identity blocks (both groups' students
+ both repo URLs), derived ``totals_by_group``, the §9.2 ``bonus_claim`` (winner 10 /
loser 7 / tie 5-5), and ``mutual_agreement``. §9.1 fixes the role alternation: sub-games
1-3 group_1 is the cop, 4-6 swapped. Everything derivable is DERIVED, never trusted.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.reporting.schema import check_table1_scores
from src.utils.jsonschema_min import validate as _schema_validate

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "docs" / "schema" / "bonus.schema.json"
_GAMES = 6  # §9.1: the bonus match is exactly 6 sub-games (3 per role assignment)


def _cop_group(game_id: int, group_1: str, group_2: str) -> str:
    """Return the cop-side group for ``game_id`` per the §9.1 alternation (1-3 / 4-6)."""
    return group_1 if game_id <= _GAMES // 2 else group_2


def derive_totals_by_group(sub_games: list[dict], group_1: str, group_2: str) -> dict:
    """Sum each group's per-sub-game score for THE ROLE IT PLAYED in that sub-game."""
    totals = {group_1: 0, group_2: 0}
    for game in sub_games:
        totals[game["cop_group"]] += int(game["scores"]["cop"])
        totals[game["thief_group"]] += int(game["scores"]["thief"])
    return totals


def derive_bonus_claim(totals: dict) -> dict:
    """Return the §9.2 claim: higher total 10 / lower 7; equal totals 5 each."""
    (name_a, score_a), (name_b, score_b) = totals.items()
    if score_a == score_b:
        return {name_a: 5, name_b: 5}
    winner, loser = (name_a, name_b) if score_a > score_b else (name_b, name_a)
    return {winner: 10, loser: 7}


def build_bonus_report(  # noqa: PLR0913 — the §9.4 identity blocks are all distinct inputs
    groups: tuple[str, str],
    repos: tuple[str, str],
    students: tuple[list[dict], list[dict]],
    timezone: str,
    results: list[dict],
    mutual_agreement: bool = False,
) -> dict:
    """Assemble the §9.4 bonus JSON from 6 referee result dicts (ids 1..6).

    Args:
        groups: ``(group_1, group_2)`` names — group_1 is the cop in sub-games 1-3.
        repos: ``(github_repo_group_1, github_repo_group_2)``.
        students: the two student blocks (role/full_name/id dicts each).
        timezone: the §3.5 timezone string (``Asia/Jerusalem``).
        results: exactly 6 referee dicts (start/end/moves/winner/scores).
        mutual_agreement: set True ONLY after both groups byte-compared results (§9.3).

    Returns:
        The complete bonus report dict (schema-valid, all derived fields computed).

    Raises:
        ValueError: If ``results`` is not exactly 6 sub-games.
        TypeError: If ``mutual_agreement`` is not a real bool — truthy coercion is
            forbidden (``bool("false")`` is True, which would fabricate a §9.3 agreement).
    """
    if not isinstance(mutual_agreement, bool):
        raise TypeError(
            f"mutual_agreement must be a real bool, got {type(mutual_agreement).__name__} "
            f"({mutual_agreement!r}) — never coerce (§9.3 agreement cannot be implied)"
        )
    if len(results) != _GAMES:
        raise ValueError(f"§9.1 requires exactly {_GAMES} sub-games, got {len(results)}")
    group_1, group_2 = groups
    sub_games = []
    for index, result in enumerate(results):
        game_id = index + 1
        cop_group = _cop_group(game_id, group_1, group_2)
        sub_games.append(
            {
                "id": game_id,
                "start": result["start"],
                "end": result["end"],
                "moves": int(result["moves"]),
                "winner": result["winner"],
                "scores": {"cop": int(result["scores"]["cop"]), "thief": int(result["scores"]["thief"])},
                "cop_group": cop_group,
                "thief_group": group_2 if cop_group == group_1 else group_1,
            }
        )
    totals = derive_totals_by_group(sub_games, group_1, group_2)
    return {
        "report_type": "bonus_game",
        "groups": {"group_1": group_1, "group_2": group_2},
        "github_repo_group_1": repos[0],
        "github_repo_group_2": repos[1],
        "timezone": timezone,
        "students_group_1": list(students[0]),
        "students_group_2": list(students[1]),
        "sub_games": sub_games,
        "totals_by_group": totals,
        "bonus_claim": derive_bonus_claim(totals),
        "mutual_agreement": mutual_agreement,
    }


def validate_bonus(report: dict, scoring: dict | None = None) -> None:
    """Validate a bonus body: schema shape + every §9 semantic invariant.

    Checks, beyond the JSON schema: the §9.1 role alternation (group_1 cop in 1-3,
    group_2 cop in 4-6, thief always the other group), that ``totals_by_group`` equals
    the re-derived per-role sums, and that ``bonus_claim`` equals the re-derived
    §9.2 claim (10/7 or 5/5) — derived fields are never trusted (mirrors §3.5). When
    ``scoring`` (``game.scoring``, Table 1) is given, additionally require every
    sub-game's scores to equal the winner's Table-1 row (the SEND path passes it).

    Raises:
        ValueError: On any schema violation or semantic inconsistency.
    """
    _schema_validate(report, json.loads(_SCHEMA_PATH.read_text(encoding="utf-8")))
    if scoring is not None:
        check_table1_scores(report["sub_games"], scoring)
    group_1 = report["groups"]["group_1"]
    group_2 = report["groups"]["group_2"]
    for game in report["sub_games"]:
        expected_cop = _cop_group(int(game["id"]), group_1, group_2)
        expected_thief = group_2 if expected_cop == group_1 else group_1
        if game["cop_group"] != expected_cop or game["thief_group"] != expected_thief:
            raise ValueError(f"§9.1 role alternation violated in sub-game {game['id']}")
    expected_totals = derive_totals_by_group(report["sub_games"], group_1, group_2)
    if report["totals_by_group"] != expected_totals:
        raise ValueError(f"totals_by_group {report['totals_by_group']} != derived {expected_totals}")
    expected_claim = derive_bonus_claim(expected_totals)
    if report["bonus_claim"] != expected_claim:
        raise ValueError(f"bonus_claim {report['bonus_claim']} != derived §9.2 claim {expected_claim}")
