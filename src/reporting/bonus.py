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


def _cop_group(game_id: int, group_1: str, group_2: str, games: int) -> str:
    """Return the cop-side group for ``game_id`` per the §9.1 alternation (1-3 / 4-6)."""
    return group_1 if game_id <= games // 2 else group_2


def derive_totals_by_group(sub_games: list[dict], group_1: str, group_2: str) -> dict:
    """Sum each group's per-sub-game score for THE ROLE IT PLAYED in that sub-game."""
    totals = {group_1: 0, group_2: 0}
    for game in sub_games:
        totals[game["cop_group"]] += int(game["scores"]["cop"])
        totals[game["thief_group"]] += int(game["scores"]["thief"])
    return totals


def derive_bonus_claim(totals: dict, claim: dict) -> dict:
    """Return the §9.2 claim: higher total gets ``winner``, lower ``loser``, equal ``tie``.

    ``claim`` is ``game.bonus_claim`` and is REQUIRED — no literal fallback. A default here
    would be the hardcode this parameter exists to remove, and it is the one scoreboard the
    OPPOSING GROUP can dispute, so it must be readable without opening source (its sibling
    ``game.scoring`` already was).
    """
    (name_a, score_a), (name_b, score_b) = totals.items()
    if score_a == score_b:
        return dict.fromkeys((name_a, name_b), int(claim["tie"]))
    winner, loser = (name_a, name_b) if score_a > score_b else (name_b, name_a)
    return {winner: int(claim["winner"]), loser: int(claim["loser"])}


def build_bonus_report(  # noqa: PLR0913 — the §9.4 identity blocks are all distinct inputs
    groups: tuple[str, str],
    repos: tuple[str, str],
    students: tuple[list[dict], list[dict]],
    timezone: str,
    results: list[dict],
    game: dict,
    mutual_agreement: bool = False,
) -> dict:
    """Assemble the §9.4 bonus JSON from 6 referee result dicts (ids 1..6).

    Args:
        groups: ``(group_1, group_2)`` names — group_1 is the cop in sub-games 1-3.
        repos: ``(github_repo_group_1, github_repo_group_2)``.
        students: the two student blocks (role/full_name/id dicts each).
        timezone: the §3.5 timezone string (``Asia/Jerusalem``).
        results: exactly 6 referee dicts (start/end/moves/winner/scores).
        game: the ``game`` config section — supplies ``num_games`` (the §9.1 match size
            and therefore the role-alternation midpoint) and ``bonus_claim`` (§9.2).
            Passed whole rather than as two scalars so the match size and the scoreboard
            can never be read from different sources.
        mutual_agreement: set True ONLY after both groups byte-compared results (§9.3).

    Returns:
        The complete bonus report dict (schema-valid, all derived fields computed).

    Raises:
        ValueError: If ``results`` is not exactly ``game['num_games']`` sub-games.
        TypeError: If ``mutual_agreement`` is not a real bool — truthy coercion is
            forbidden (``bool("false")`` is True, which would fabricate a §9.3 agreement).
    """
    if not isinstance(mutual_agreement, bool):
        raise TypeError(
            f"mutual_agreement must be a real bool, got {type(mutual_agreement).__name__} "
            f"({mutual_agreement!r}) — never coerce (§9.3 agreement cannot be implied)"
        )
    games = int(game["num_games"])
    if len(results) != games:
        raise ValueError(f"§9.1 requires exactly {games} sub-games, got {len(results)}")
    group_1, group_2 = groups
    sub_games = []
    for index, result in enumerate(results):
        game_id = index + 1
        cop_group = _cop_group(game_id, group_1, group_2, games)
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
    claim = derive_bonus_claim(totals, game["bonus_claim"])
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
        "bonus_claim": claim,
        "mutual_agreement": mutual_agreement,
    }


def validate_bonus(report: dict, game: dict | None = None) -> None:
    """Validate a bonus body: schema shape + every §9 semantic invariant.

    Always checks, beyond the JSON schema: the §9.1 role alternation (group_1 cop in the
    first half, group_2 cop in the second, thief always the other group) and that
    ``totals_by_group`` equals the re-derived per-role sums — derived fields are never
    trusted (mirrors §3.5). The alternation midpoint comes from the body's OWN sub-game
    count, so the structural pass needs no config at all.

    ``game`` (the ``game`` config section) additionally checks the two SCOREBOARDS, which
    are the parts a value can be wrong about rather than merely mis-shaped: every
    sub-game's scores must equal the winner's Table-1 row (``game.scoring``), and
    ``bonus_claim`` must equal the re-derived §9.2 claim (``game.bonus_claim``). Optional
    for the same reason ``scoring`` was: draft-stage callers validate structure before a
    config is in hand. The SEND path passes it, so nothing leaves the machine unchecked.

    Raises:
        ValueError: On any schema violation or semantic inconsistency.
    """
    _schema_validate(report, json.loads(_SCHEMA_PATH.read_text(encoding="utf-8")))
    if game is not None:
        check_table1_scores(report["sub_games"], game["scoring"])
    group_1 = report["groups"]["group_1"]
    group_2 = report["groups"]["group_2"]
    games = len(report["sub_games"])
    for sub_game in report["sub_games"]:
        expected_cop = _cop_group(int(sub_game["id"]), group_1, group_2, games)
        expected_thief = group_2 if expected_cop == group_1 else group_1
        if sub_game["cop_group"] != expected_cop or sub_game["thief_group"] != expected_thief:
            raise ValueError(f"§9.1 role alternation violated in sub-game {sub_game['id']}")
    expected_totals = derive_totals_by_group(report["sub_games"], group_1, group_2)
    if report["totals_by_group"] != expected_totals:
        raise ValueError(f"totals_by_group {report['totals_by_group']} != derived {expected_totals}")
    if game is None:
        return
    expected_claim = derive_bonus_claim(expected_totals, game["bonus_claim"])
    if report["bonus_claim"] != expected_claim:
        raise ValueError(f"bonus_claim {report['bonus_claim']} != derived §9.2 claim {expected_claim}")
