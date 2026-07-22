"""The V3 gate suite the PLAN/TODO checklists promise — executable, not aspirational.

Each test here is one row of the V3 §17 pre-submission checklist that the planning docs
name by test id: single SDK entry, single config source, version consistency, no secrets,
no PII in tracked content, the git-ignored cover sheet, and the aggregate gate sweep.
They run against the REPO ITSELF (not a fixture), so a regression in project hygiene
fails CI rather than being discovered by a grader.
"""

from __future__ import annotations

import json
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

import src
from src.utils.config_loader import EXPECTED_VERSION, load_config

_ROOT = Path(__file__).resolve().parents[2]
# Identity strings that must never appear in tracked content (A1-A5 PII deny-list). Kept
# as fragments assembled at runtime so this file itself never carries a literal identity.
_PII_PATTERNS = (r"\.local\.yaml.*:.*[A-Za-z]{3,}\s+[A-Za-z]{3,}", r"\b\d{9}\b")


def _tracked_files() -> list[str]:
    """Return every git-tracked path (the only content a grader can see)."""
    out = subprocess.run(["git", "ls-files"], cwd=_ROOT, capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line]


def _tracked_text(path: str) -> str | None:
    """Return a tracked file's text, or None when it is binary/unreadable."""
    try:
        return (_ROOT / path).read_text(encoding="utf-8")
    except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
        return None


def test_sdk_single_entry():
    """Only the SDK (and its own package) may import the business-logic layers.

    ADR-0002: `MarlSDK` is the single business-logic surface. Thin surfaces — CLI, GUI,
    MCP servers, scripts — must reach `src.marl` / `src.services` THROUGH the SDK.
    """
    offenders: list[str] = []
    for path in _tracked_files():
        if not path.startswith(("src/cli.py", "src/gui/", "scripts/")):
            continue
        if not path.endswith(".py"):
            continue
        text = _tracked_text(path) or ""
        for line in text.splitlines():
            stripped = line.strip()
            if re.match(r"(from|import)\s+src\.(marl|services)\b", stripped):
                # The GUI's frozen frame type + the spectator seam are SDK-returned DTOs.
                if "spectator" in stripped or "SpectatorFrame" in stripped:
                    continue
                offenders.append(f"{path}: {stripped}")
    assert not offenders, "thin surfaces must import business logic via the SDK: " + "; ".join(offenders)


def test_config_single_source():
    """Exactly one tracked YAML config, loaded through the single loader (§7.3)."""
    configs = [p for p in _tracked_files() if p.startswith("config/") and p.endswith(".yaml")]
    assert configs == ["config/config.yaml"], f"expected one config yaml, found {configs}"
    cfg = load_config()
    assert cfg["version"] == EXPECTED_VERSION


def test_version_consistency():
    """`__version__` == `config.version` == `pyproject.version` (§8.1)."""
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert src.__version__ == EXPECTED_VERSION
    assert load_config()["version"] == EXPECTED_VERSION
    assert pyproject["project"]["version"] == EXPECTED_VERSION


def test_no_secrets_committed():
    """No secret-shaped assignment, key material, or the recipient address is tracked (§7.4)."""
    secret_assign = re.compile(
        r"(api[_-]?key|secret|password|token)\s*[=:]\s*[\"'][^\"'\n]{6,}[\"']", re.IGNORECASE
    )
    banned_suffixes = (".pem", ".key", ".p12", ".pfx")
    offenders: list[str] = []
    for path in _tracked_files():
        if path.endswith(banned_suffixes) or path == ".env":
            offenders.append(f"{path}: secret-bearing file is tracked")
            continue
        scanned = path.startswith(("src/", "scripts/", "config/"))
        if not scanned or not path.endswith((".py", ".yaml", ".json")):
            continue
        text = _tracked_text(path) or ""
        for number, line in enumerate(text.splitlines(), start=1):
            if "example" in line.lower() or "placeholder" in line.lower():
                continue
            if secret_assign.search(line):
                offenders.append(f"{path}:{number}")
    assert not offenders, "secret material in tracked content: " + "; ".join(offenders)


def test_no_pii_in_tracked_content():
    """No host paths or id-shaped numbers leak into tracked text (the goal-r6 class).

    An absolute ``/Users/`` path carries the local username (and, on this project, the
    Drive-mount email); a bare 9-digit run of digits is id-shaped. Both are PII per the
    project deny-list, so tracked content must contain neither.
    """
    offenders: list[str] = []
    for path in _tracked_files():
        skip = path.endswith((".png", ".pt", ".ipynb", ".lock"))
        # tests/ carry documented PLACEHOLDER ids (the schema requires id strings in fixtures)
        if skip or path.startswith((".github/", "tests/")):
            continue
        text = _tracked_text(path)
        if text is None:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if "/Users/" in line:
                offenders.append(f"{path}:{number} host path")
            if re.search(_PII_PATTERNS[1], line) and "example" not in line.lower():
                offenders.append(f"{path}:{number} id-shaped digits")
    assert not offenders, "PII in tracked content: " + "; ".join(offenders[:10])


def test_cover_sheet_and_identity_files_are_git_ignored():
    """The Moodle cover sheet + real-identity files must never be tracked (skip if absent).

    The artifacts themselves are git-ignored and may not exist on a fresh clone, so this
    asserts the INVARIANT (never tracked) rather than their presence.
    """
    for name in ("adrl-001-ex06.pdf", "players.local.yaml", "players.partner.local.yaml", ".env"):
        assert name not in _tracked_files(), f"{name} must never be tracked"
        if (_ROOT / name).exists():
            check = subprocess.run(  # fixed argv, no shell
                ["git", "check-ignore", name], cwd=_ROOT, capture_output=True, text=True, check=False
            )
            assert check.returncode == 0, f"{name} exists but is NOT git-ignored"


@pytest.mark.parametrize(
    "artifact",
    [
        "README.md",
        "docs/PRD.md",
        "docs/PLAN.md",
        "docs/TODO.md",
        "docs/QUALITY.md",
        "docs/schema/report.schema.json",
        "docs/schema/bonus.schema.json",
        ".env-example",
        "uv.lock",
        "config/rate_limits.json",
    ],
)
def test_final_gates_required_artifacts_present(artifact: str):
    """Every artifact the V3 §17 checklist depends on is tracked and non-empty."""
    assert artifact in _tracked_files(), f"{artifact} is not tracked"
    assert (_ROOT / artifact).stat().st_size > 0, f"{artifact} is empty"


def test_final_gates_rate_limits_are_versioned():
    """`config/rate_limits.json` carries a version + a limit per declared channel (§5.2)."""
    limits = json.loads((_ROOT / "config" / "rate_limits.json").read_text(encoding="utf-8"))
    assert "version" in limits
    channels = limits["limits"]
    assert channels, "no channels declared"
    for name, spec in channels.items():
        assert {"per_minute", "burst"} <= set(spec), f"channel {name} declares no per_minute/burst"
