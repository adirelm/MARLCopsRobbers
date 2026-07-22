"""Serving factories for the §9 wire agents (src.mcp.wire_serve).

The dress-rehearsal / match-day serving seam: the SHIPPED policy lineup per role
(trained serving cop net; AdaptiveThiefPolicy thief — the README §9 lineup), ports
parsed from the SAME wire_match.groups URLs the referee dials (single source), the
local-vs-remote group filter, and start_group_agents wiring one single-role wire
agent per (group, role) with per-group bearer tokens from the config-NAMED env vars.
"""

from __future__ import annotations

import socket

import httpx
import pytest

from src.marl.env.actions import Action
from src.mcp import wire_serve
from src.services.bonus_policies import AdaptiveThiefPolicy
from src.services.policy import RecurrentPolicy

_NEW = {"session_id": "sg-0", "grid": [5, 5], "your_role": "cop", "your_pos": [0, 0], "max_moves": 25}


class _StubPolicy:
    """Minimal act()+reset() double (no torch) for socket-level wiring tests."""

    def reset(self):
        pass

    def act(self, obs_list, legal_masks, epsilon, rng, state=None):
        return [Action.UP]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _rewire(cfg, key, cop_port, thief_port):
    spec = cfg["wire_match"]["groups"][key]
    spec["cop_url"] = f"http://127.0.0.1:{cop_port}"
    spec["thief_url"] = f"http://127.0.0.1:{thief_port}"


def test_shipped_policy_is_the_advertised_lineup(cfg):
    assert isinstance(wire_serve.shipped_policy(cfg, "cop"), RecurrentPolicy)
    thief = wire_serve.shipped_policy(cfg, "thief")
    assert isinstance(thief, AdaptiveThiefPolicy) and thief.switched is False


def test_group_ports_come_from_the_configured_urls(cfg):
    _rewire(cfg, "group_1", 8201, 8203)
    assert wire_serve.group_ports(cfg, "group_1") == {"cop": 8201, "thief": 8203}


def test_group_ports_require_an_explicit_port(cfg):
    cfg["wire_match"]["groups"]["group_2"]["thief_url"] = "https://partner.example.com"
    with pytest.raises(ValueError):
        wire_serve.group_ports(cfg, "group_2")


def test_local_group_keys_skip_remote_partner_urls(cfg):
    cfg["wire_match"]["groups"]["group_2"]["cop_url"] = "https://partner.example.com:8443"
    assert wire_serve.local_group_keys(cfg) == ["group_1"]


def test_start_group_agents_serve_split_roles_with_env_tokens(cfg, monkeypatch):
    ports = {key: (_free_port(), _free_port()) for key in ("group_1", "group_2")}
    for key, (cop_port, thief_port) in ports.items():
        _rewire(cfg, key, cop_port, thief_port)
    monkeypatch.setenv("WIRE_GROUP_1_TOKEN", "tok-one")
    monkeypatch.setenv("WIRE_GROUP_2_TOKEN", "tok-two")
    agents = wire_serve.start_group_agents(
        cfg, ["group_1", "group_2"], policy_factory=lambda k, r: _StubPolicy()
    )
    try:
        for cop_port, thief_port in ports.values():
            for port in (cop_port, thief_port):
                assert httpx.get(f"http://127.0.0.1:{port}/health").json() == {"status": "ok"}
        auth = {"Authorization": "Bearer tok-one"}
        g1_cop = f"http://127.0.0.1:{ports['group_1'][0]}/new_sub_game"
        assert httpx.post(g1_cop, json={**_NEW, "your_role": "thief"}, headers=auth).status_code == 400
        assert httpx.post(g1_cop, json=_NEW, headers=auth).json() == {"ok": True}  # role split holds
        g2_cop = f"http://127.0.0.1:{ports['group_2'][0]}/new_sub_game"
        assert httpx.post(g2_cop, json=_NEW, headers=auth).status_code == 401  # group_2 wants tok-two
        assert httpx.post(g2_cop, json=_NEW, headers={"Authorization": "Bearer tok-two"}).json() == {
            "ok": True
        }
    finally:
        for agent in agents:
            agent.close()


def test_start_group_agents_default_factory_serves_the_shipped_lineup(cfg, monkeypatch):
    ports = (_free_port(), _free_port())
    _rewire(cfg, "group_1", *ports)
    monkeypatch.setenv("WIRE_GROUP_1_TOKEN", "tok-one")
    agents = wire_serve.start_group_agents(cfg, ["group_1"])  # no factory -> shipped serving nets
    try:
        for port in ports:
            assert httpx.get(f"http://127.0.0.1:{port}/health").json() == {"status": "ok"}
    finally:
        for agent in agents:
            agent.close()


def test_start_group_agents_refuse_a_missing_token_and_clean_up(cfg, monkeypatch):
    ports = {key: (_free_port(), _free_port()) for key in ("group_1", "group_2")}
    for key, (cop_port, thief_port) in ports.items():
        _rewire(cfg, key, cop_port, thief_port)
    monkeypatch.setenv("WIRE_GROUP_1_TOKEN", "tok-one")
    monkeypatch.delenv("WIRE_GROUP_2_TOKEN", raising=False)
    with pytest.raises(ValueError, match="WIRE_GROUP_2_TOKEN"):
        wire_serve.start_group_agents(cfg, ["group_1", "group_2"], policy_factory=lambda k, r: _StubPolicy())
    with pytest.raises(httpx.TransportError):  # group_1's already-started agents were closed again
        httpx.get(f"http://127.0.0.1:{ports['group_1'][0]}/health", timeout=0.5)
