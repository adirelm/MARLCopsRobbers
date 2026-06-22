"""Required-docs gate (T10.8/10.9) — QUALITY.md headings, COST_ANALYSIS.md, SDK-only notebook."""

from __future__ import annotations

import json
import re
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
        "docs/shared/PROMPTS.md",
        "docs/prd/PRD-ENV.md",  # NFR-11 per-area PRDs (V3 §2 dedicated-PRD-per-mechanism)
        "docs/prd/PRD-CTDE.md",
        "docs/prd/PRD-MCP.md",
        "docs/prd/PRD-REPORT.md",
        "notebooks/analysis.ipynb",
    ]
    for rel in required:
        assert (_ROOT / rel).exists(), f"missing required doc: {rel}"


def test_plan_indexes_all_adrs_0001_through_0014():
    """NFR-11 ADR-count gate: PLAN §6 is the canonical ADR record indexing ADR-0001..0014."""
    text = (_ROOT / "docs" / "PLAN.md").read_text(encoding="utf-8")
    for i in range(1, 15):
        assert f"**{i:04d}**" in text, f"PLAN.md ADR index missing ADR {i:04d}"


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


def test_prd_cited_test_files_resolve():
    """Every ``test_*.py`` cited in PRD.md (AC + Evidence) resolves to a real test (by basename).

    Permanently closes the doc->test-pointer drift class (goal-loop r5/r7/r10): the pre-code PRD
    cited test names that were later renamed/consolidated; this gate fails CI if any PRD-cited test
    file no longer exists, so the requirement->test evidence trail a grader follows stays accurate.
    """
    prd = (_ROOT / "docs" / "PRD.md").read_text(encoding="utf-8")
    cited = set(re.findall(r"`([\w./-]*test_\w+\.py)`", prd))
    actual = {p.name for p in (_ROOT / "tests").rglob("test_*.py")}
    missing = sorted(c for c in cited if c.split("/")[-1] not in actual)
    assert not missing, f"PRD.md cites nonexistent test file(s): {missing}"
