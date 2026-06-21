"""Required-docs gate (T10.8/10.9) — QUALITY.md headings, COST_ANALYSIS.md, SDK-only notebook."""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

_ISO_25010 = [
    "Functional Suitability",
    "Performance Efficiency",
    "Compatibility",
    "Usability",
    "Reliability",
    "Security",
    "Maintainability",
    "Portability",
]


def test_required_docs_exist():
    required = [
        "docs/QUALITY.md",
        "docs/COST_ANALYSIS.md",
        "docs/PRD.md",
        "docs/PLAN.md",
        "docs/ANALYSIS.md",
        "docs/UX.md",
        "notebooks/analysis.ipynb",
    ]
    for rel in required:
        assert (_ROOT / rel).exists(), f"missing required doc: {rel}"


def test_quality_doc_covers_all_iso_25010_characteristics():
    text = (_ROOT / "docs" / "QUALITY.md").read_text(encoding="utf-8")
    for characteristic in _ISO_25010:
        assert characteristic in text, f"QUALITY.md missing ISO 25010 characteristic: {characteristic}"


def test_cost_analysis_has_token_table_and_training_envelope():
    text = (_ROOT / "docs" / "COST_ANALYSIS.md").read_text(encoding="utf-8")
    assert "Claude Opus" in text and "$/1M" in text  # AI token-cost table
    assert "episodes" in text.lower() and "5,000" in text  # local training envelope


def test_analysis_notebook_imports_sdk_only():
    notebook = json.loads((_ROOT / "notebooks" / "analysis.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "src.sdk" in code  # the notebook consumes the SDK
    for forbidden in ["src.marl", "src.mcp", "src.services", "src.learn"]:
        assert forbidden not in code, f"notebook must not import {forbidden} (SDK-only, T10.7)"
