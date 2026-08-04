"""Shared unit-test fixtures.

``real_match`` lives here rather than in one test module because three modules assert
against the SAME committed §9 artifacts. Importing a fixture by name into another module
shadows the test's own parameter (ruff F811); conftest is how pytest is meant to share one.
"""

from __future__ import annotations

import json

import pytest

from src.mcp._replay_log import select_log_and_records
from src.utils.config_loader import load_config


@pytest.fixture
def real_match():
    """The committed §9 match (cfg, log path, parsed §9.4 body) — skipped if absent."""
    cfg = load_config()
    try:
        log, records = select_log_and_records(cfg)
    except SystemExit:  # pragma: no cover - only on a checkout without the match artifacts
        pytest.skip("no committed wire match in this checkout")
    return cfg, log, json.loads(records.read_text(encoding="utf-8"))


@pytest.fixture
def write_body(tmp_path):
    """Write a tampered §9.4 body to a temp path and return it."""

    def _write(body: dict):
        path = tmp_path / "tampered_records.json"
        path.write_text(json.dumps(body), encoding="utf-8")
        return path

    return _write
