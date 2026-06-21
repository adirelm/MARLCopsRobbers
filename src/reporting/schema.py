"""§3.5 MatchReport — stdlib dataclasses + JSON-schema validation (T6.3 / T9).

NO pydantic at runtime: the report is built + validated with stdlib only. Student
PII (``full_name`` / ``id``) is supplied at runtime from the git-ignored
``players.local.yaml``; TRACKED content + tests use PLACEHOLDERS only. ``totals``
are DERIVED from the per-sub-game scores (never trusted from input). ``validate``
checks an assembled body against ``docs/schema/report.schema.json`` (§3.5 contract).
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

    full_name: str
    id: str


@dataclass(frozen=True)
class SubGame:
    """One adjudicated sub-game result (a §3.5 ``sub_games`` item)."""

    game_id: str
    grid: tuple[int, int]
    winner: str
    capture: bool
    steps: int
    scores: dict
    seed: int


@dataclass(frozen=True)
class MatchReport:
    """The assembled §3.5 report; ``totals`` are DERIVED from the sub-game scores."""

    group: str
    students: list[Student]
    sub_games: list[SubGame]

    def totals(self) -> dict[str, int]:
        """Return the per-role score totals summed over the sub-games."""
        return {role: sum(int(g.scores[role]) for g in self.sub_games) for role in ("cop", "thief")}

    def to_dict(self) -> dict:
        """Serialize to the §3.5 JSON body (num_games + derived totals)."""
        return {
            "group": self.group,
            "students": [{"full_name": s.full_name, "id": s.id} for s in self.students],
            "num_games": len(self.sub_games),
            "sub_games": [_game_dict(g) for g in self.sub_games],
            "totals": self.totals(),
        }


def _game_dict(game: SubGame) -> dict:
    """Serialize one SubGame to its §3.5 record (ints + 2-int grid)."""
    return {
        "game_id": game.game_id,
        "grid": [int(game.grid[0]), int(game.grid[1])],
        "winner": game.winner,
        "capture": bool(game.capture),
        "steps": int(game.steps),
        "scores": {"cop": int(game.scores["cop"]), "thief": int(game.scores["thief"])},
        "seed": int(game.seed),
    }


def sub_game_from_result(result: dict) -> SubGame:
    """Build a :class:`SubGame` from a referee ``play_sub_game`` result dict."""
    return SubGame(
        game_id=result["game_id"],
        grid=tuple(result["grid"]),
        winner=result["winner"],
        capture=bool(result["capture"]),
        steps=int(result["steps"]),
        scores=result["scores"],
        seed=int(result["seed"]),
    )


def build_report(group: str, students: list[Student], results: list[dict]) -> MatchReport:
    """Assemble a :class:`MatchReport` from referee sub-game result dicts."""
    return MatchReport(
        group=group, students=list(students), sub_games=[sub_game_from_result(r) for r in results]
    )


def validate(report: dict) -> None:
    """Validate an assembled report dict against ``docs/schema/report.schema.json``."""
    _schema_validate(report, json.loads(_SCHEMA_PATH.read_text(encoding="utf-8")))
