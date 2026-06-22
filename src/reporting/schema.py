"""§3.5 MatchReport — stdlib dataclasses + JSON-schema validation (T6.3 / T9).

The report body MATCHES the BRIEF §3.5 structure exactly: ``group_name``, ``students``
(role + full_name + id), ``github_repo``, ``timezone``, ``sub_games`` (each ``id`` / ``start``
/ ``end`` Jerusalem ISO-8601 / ``moves`` / ``winner`` / ``scores``), and DERIVED ``totals``.
NO pydantic at runtime. Student PII (``full_name`` / ``id``) is supplied from the git-ignored
``players.local.yaml``; TRACKED content + tests use PLACEHOLDERS only. ``validate`` checks an
assembled body against ``docs/schema/report.schema.json`` + the §3.5 semantic invariants.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.utils.jsonschema_min import validate as _schema_validate

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "docs" / "schema" / "report.schema.json"


@dataclass(frozen=True)
class Student:
    """A submitting student (PII — PLACEHOLDERS only in tracked content)."""

    role: str
    full_name: str
    id: str


@dataclass(frozen=True)
class SubGame:
    """One §3.5 sub-game record (id 1..6, Jerusalem ISO timestamps, moves, winner, scores)."""

    id: int
    start: str
    end: str
    moves: int
    winner: str
    scores: dict


@dataclass(frozen=True)
class MatchReport:
    """The assembled §3.5 report; ``totals`` are DERIVED from the sub-game scores."""

    group_name: str
    students: list[Student]
    github_repo: str
    timezone: str
    sub_games: list[SubGame]

    def totals(self) -> dict[str, int]:
        """Return the per-role score totals summed over the sub-games."""
        return {role: sum(int(g.scores[role]) for g in self.sub_games) for role in ("cop", "thief")}

    def to_dict(self) -> dict:
        """Serialize to the §3.5 JSON body (brief structure + derived totals)."""
        return {
            "group_name": self.group_name,
            "students": [{"role": s.role, "full_name": s.full_name, "id": s.id} for s in self.students],
            "github_repo": self.github_repo,
            "timezone": self.timezone,
            "sub_games": [_game_dict(g) for g in self.sub_games],
            "totals": self.totals(),
        }


def _game_dict(game: SubGame) -> dict:
    """Serialize one SubGame to its §3.5 record (brief field names + int scores)."""
    return {
        "id": int(game.id),
        "start": game.start,
        "end": game.end,
        "moves": int(game.moves),
        "winner": game.winner,
        "scores": {"cop": int(game.scores["cop"]), "thief": int(game.scores["thief"])},
    }


def sub_game_from_result(result: dict, game_id: int) -> SubGame:
    """Build a :class:`SubGame` from a referee result dict (``moves``/``start``/``end``)."""
    return SubGame(
        id=int(game_id),
        start=result["start"],
        end=result["end"],
        moves=int(result["moves"]),
        winner=result["winner"],
        scores=result["scores"],
    )


def build_report(
    group_name: str, students: list[Student], github_repo: str, timezone: str, results: list[dict]
) -> MatchReport:
    """Assemble a :class:`MatchReport` from referee sub-game result dicts (ids 1..N)."""
    return MatchReport(
        group_name=group_name,
        students=list(students),
        github_repo=github_repo,
        timezone=timezone,
        sub_games=[sub_game_from_result(r, i + 1) for i, r in enumerate(results)],
    )


def validate(report: dict, expected_games: int | None = None) -> None:
    """Validate a report dict against the schema AND the §3.5 semantic invariants.

    After the JSON-schema pass, assert each ``totals[role]`` equals the sum of the
    per-sub-game scores (the §3.5 totals are DERIVED, never trusted). When
    ``expected_games`` is given, require EXACTLY that many sub-games — the §3.5 "a full
    game = N sub-games" contract (FR-RPT-2). The SEND path passes ``game.num_games`` (6);
    mid-pipeline self-checks omit it so a parameterized partial match still validates.

    Raises:
        ValueError: On a schema violation, a totals inconsistency, OR a sub-game count
            that differs from ``expected_games`` when it is supplied.
    """
    _schema_validate(report, json.loads(_SCHEMA_PATH.read_text(encoding="utf-8")))
    sub_games = report["sub_games"]
    if expected_games is not None and len(sub_games) != expected_games:
        raise ValueError(f"§3.5 requires exactly {expected_games} sub-games, got {len(sub_games)}")
    for role in ("cop", "thief"):
        expected = sum(int(g["scores"][role]) for g in sub_games)
        if report["totals"][role] != expected:
            raise ValueError(f"totals[{role!r}] {report['totals'][role]} != Σ sub-game scores {expected}")
