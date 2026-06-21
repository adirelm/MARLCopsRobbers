"""Load the submitting players (group + students) for the §3.5 report (T6.1/T9).

``players.local.yaml`` (GIT-IGNORED) holds the REAL names/IDs; ``players.example.yaml``
is the TRACKED placeholder. The PII (``full_name`` / ``id``) is injected into the
§3.5 body ONLY at send time and is NEVER committed (the committed report copy is
redacted to roles). Prefers the local file when present, else the placeholder.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]


def _default_path() -> Path:
    """Return players.local.yaml if it exists, else the tracked placeholder."""
    local = _ROOT / "players.local.yaml"
    return local if local.exists() else _ROOT / "players.example.yaml"


def load_players(path: str | Path | None = None) -> dict:
    """Return ``{"group", "students": [{"full_name", "id"}, ...]}`` for the report.

    Args:
        path: Optional explicit players file; defaults to the local file when
            present, otherwise the tracked placeholder.

    Returns:
        The group code + the per-student name/id list (placeholders when no local
        file exists — never real PII in tracked content).
    """
    raw = yaml.safe_load(Path(path or _default_path()).read_text(encoding="utf-8"))
    students = [{"full_name": s["full_name"], "id": s["id"]} for s in raw["students"]]
    return {"group": raw["group_name"], "students": students}
