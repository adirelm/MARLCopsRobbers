"""V3 §16 building-block contract — the MCP server factory and the report send path.

The two externally-facing components: ``build_server`` (config + role input, injected
controller/verifier) and ``send_report`` (validates an outbound payload). Both must
reject a bad config with ValueError and a wrong-typed input with TypeError, while still
accepting the real config. Companion to ``test_building_blocks.py``.
"""

from __future__ import annotations

import copy

import pytest
from fastmcp import FastMCP

from src.mcp.server_builder import build_server
from src.reporting import send as send_mod
from src.reporting.mailer import FakeEmailSender
from tests.unit.test_report_send import _cfg_tmp, _report


class _StubController:
    """Injected AgentController stand-in — §16 testability via dependency injection."""

    def new_session(self, session_id: str) -> None:
        """Record a new session (no state needed for the factory contract)."""

    def act(self, session_id, tick, image, scalars, legal_mask) -> int:
        """Return a fixed action."""
        return 0


def test_server_builder_rejects_a_config_without_the_observation_block(cfg):
    """mcp.observation.view_radius gates reveal_location — its absence is a config error."""
    bad = copy.deepcopy(cfg)
    del bad["mcp"]["observation"]
    with pytest.raises(ValueError, match="observation"):
        build_server(bad, "cop", _StubController(), None)


def test_server_builder_rejects_a_config_without_a_protocol_version(cfg):
    """The cross-server contract version is mandatory (health() handshake)."""
    bad = copy.deepcopy(cfg)
    bad["mcp"]["protocol_version"] = ""
    with pytest.raises(ValueError, match="protocol_version"):
        build_server(bad, "cop", _StubController(), None)


def test_server_builder_rejects_a_non_string_role(cfg):
    """The role names the server and selects the cop-only report tool."""
    with pytest.raises(TypeError, match="role"):
        build_server(cfg, 7, _StubController(), None)


def test_server_builder_rejects_a_controller_without_act(cfg):
    """The injected controller must expose the AgentController seam."""
    with pytest.raises(TypeError, match="controller"):
        build_server(cfg, "cop", object(), None)


def test_server_builder_accepts_the_real_config_and_an_injected_controller(cfg):
    """Happy path: a valid config + stub controller still yields a FastMCP server."""
    assert isinstance(build_server(cfg, "cop", _StubController(), None), FastMCP)


def test_send_report_rejects_a_config_missing_a_gmail_key(cfg, tmp_path):
    """gmail.subject_template is formatted on every send — its absence is a config error."""
    bad = _cfg_tmp(cfg, tmp_path)
    del bad["gmail"]["subject_template"]
    with pytest.raises(ValueError, match="subject_template"):
        send_mod.send_report(bad, _report(), FakeEmailSender(), "2026-06-21")


def test_send_report_rejects_a_non_positive_num_games(cfg, tmp_path):
    """game.num_games is the §3.5 expected sub-game count — it must be >= 1."""
    bad = _cfg_tmp(cfg, tmp_path)
    bad["game"]["num_games"] = 0
    with pytest.raises(ValueError, match="num_games"):
        send_mod.send_report(bad, _report(), FakeEmailSender(), "2026-06-21")


def test_send_report_rejects_a_non_dict_report(cfg, tmp_path):
    """The outbound payload must be a mapping before any schema validation runs."""
    with pytest.raises(TypeError, match="report"):
        send_mod.send_report(_cfg_tmp(cfg, tmp_path), "not-a-report", FakeEmailSender(), "2026-06-21")


def test_send_report_rejects_a_sender_without_send(cfg, tmp_path):
    """The injected sender is the egress seam — it must expose send()."""
    with pytest.raises(TypeError, match="sender"):
        send_mod.send_report(_cfg_tmp(cfg, tmp_path), _report(), object(), "2026-06-21")


def test_send_report_rejects_a_non_string_date(cfg, tmp_path):
    """date_str is formatted into the subject line."""
    with pytest.raises(TypeError, match="date_str"):
        send_mod.send_report(_cfg_tmp(cfg, tmp_path), _report(), FakeEmailSender(), 20260621)


def test_send_report_accepts_the_real_config_and_a_valid_payload(cfg, tmp_path):
    """Happy path: a valid config + report + injected fake sender still sends exactly once."""
    sender = FakeEmailSender()
    out = send_mod.send_report(_cfg_tmp(cfg, tmp_path), _report(), sender, "2026-06-21")
    assert out["sent"] is True and len(sender.sent) == 1
