"""Unit tests for src.utils.config_loader (T0.3)."""

from __future__ import annotations

import copy
import textwrap

import pytest

from src.utils.config_loader import (
    REQUIRED_SECTIONS,
    _validate_config,
    load_config,
)

EXPECTED_SECTIONS = {
    "version",
    "project",
    "game",
    "env",
    "algo",
    "nets",
    "compute",
    "olora",
    "bc",
    "replay",
    "selfplay",
    "training",
    "reward",
    "mcp",
    "cloud",
    "gmail",
    "gui",
    "paths",
    "logging",
}


def test_required_sections_pinned():
    """The loader's REQUIRED_SECTIONS matches the frozen spec list."""
    assert set(REQUIRED_SECTIONS) == EXPECTED_SECTIONS


def test_round_trips_real_config():
    """load_config() reads the real config and returns a validated dict."""
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert cfg["version"] == "1.0.0"
    for section in EXPECTED_SECTIONS:
        assert section in cfg
    assert cfg["game"]["grid_size"] == 5
    assert cfg["env"]["actions"]["a_cop"] == 5
    assert cfg["paths"]["runs_dir"] == "results/runs"


def _write(tmp_path, body: str):
    path = tmp_path / "cfg.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_bad_version_raises(tmp_path):
    """A version other than 1.0.0 raises ValueError."""
    body = 'version: "9.9.9"\n'
    for section in EXPECTED_SECTIONS - {"version"}:
        body += f"{section}: {{}}\n"
    path = _write(tmp_path, body)
    with pytest.raises(ValueError, match="version"):
        load_config(path)


def test_missing_section_raises(tmp_path):
    """A missing required top-level section raises ValueError."""
    body = 'version: "1.0.0"\n'
    for section in EXPECTED_SECTIONS - {"version", "gui"}:
        body += f"{section}: {{}}\n"
    path = _write(tmp_path, body)
    with pytest.raises(ValueError, match="gui"):
        load_config(path)


def test_env_var_interpolation(tmp_path, monkeypatch):
    """${VAR} tokens expand from os.environ during load."""
    monkeypatch.setenv("TEST_VAR", "interpolated-value")
    body = 'version: "1.0.0"\n'
    for section in EXPECTED_SECTIONS - {"version", "project"}:
        body += f"{section}: {{}}\n"
    body += "project:\n  token: ${TEST_VAR}\n"
    path = _write(tmp_path, body)
    cfg = load_config(path)
    assert cfg["project"]["token"] == "interpolated-value"


def test_env_var_in_list_and_scalars_passthrough(tmp_path, monkeypatch):
    """${VAR} inside a list element expands; non-str scalars pass through."""
    monkeypatch.setenv("TEST_VAR", "expanded")
    body = 'version: "1.0.0"\n'
    for section in EXPECTED_SECTIONS - {"version", "project"}:
        body += f"{section}: {{}}\n"
    body += (
        "project:\n"
        "  urls:\n"
        "    - https://${TEST_VAR}.example\n"
        "    - plain\n"
        "  count: 7\n"
        "  enabled: true\n"
        "  ratio: 0.5\n"
    )
    path = _write(tmp_path, body)
    cfg = load_config(path)
    assert cfg["project"]["urls"] == ["https://expanded.example", "plain"]
    assert cfg["project"]["count"] == 7
    assert cfg["project"]["count"] is not True  # int 7 stays int, not coerced
    assert cfg["project"]["enabled"] is True
    assert cfg["project"]["ratio"] == 0.5


def test_returns_independent_deep_copies():
    """Two load_config() calls return distinct objects; mutation is isolated."""
    first = load_config()
    second = load_config()
    assert first is not second
    assert first["game"] is not second["game"]
    first["game"]["grid_size"] = 999
    first["game"]["__injected__"] = "leak"
    assert second["game"]["grid_size"] == 5
    assert "__injected__" not in second["game"]
    # A subsequent call also sees the pristine cached value, not the mutation.
    assert load_config()["game"]["grid_size"] == 5


def test_none_section_raises(tmp_path):
    """A present-but-empty (None) required section raises ValueError."""
    body = 'version: "1.0.0"\n'
    for section in EXPECTED_SECTIONS - {"version", "gui"}:
        body += f"{section}: {{}}\n"
    body += "gui:\n"  # empty YAML block -> parses to None
    path = _write(tmp_path, body)
    with pytest.raises(ValueError, match="gui"):
        load_config(path)


def test_none_section_from_deep_copied_dict_raises():
    """A required section set to None on a parsed dict is rejected on revalidate."""
    cfg = copy.deepcopy(load_config())
    cfg["gui"] = None
    with pytest.raises(ValueError, match="gui"):
        _validate_config(cfg)


def test_non_mapping_section_raises():
    """A required mapping section that is present but not a dict is rejected."""
    cfg = copy.deepcopy(load_config())
    cfg["game"] = 5
    with pytest.raises(ValueError, match="mapping"):
        _validate_config(cfg)


def test_non_mapping_root_raises(tmp_path):
    """A top-level YAML list/scalar raises ValueError, not AttributeError."""
    path = _write(tmp_path, "- not\n- a\n- mapping\n")
    with pytest.raises(ValueError, match="root must be a mapping"):
        load_config(path)
