"""Match-day tripwires + serving flags — the paths that guard a REAL §9 match."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.run_bonus_match import _REHEARSAL_SEEDS, _real_match_guards, build_clients
from scripts.serve_wire_agents import main as serve_main
from src.mcp.wire_serve import conformance_policy, resolve_token
from src.utils.config_loader import load_config


@pytest.fixture(scope="module")
def cfg():
    """One loaded config for the module (copied before mutation in each test)."""
    return load_config()


def test_rehearsal_mode_passes_the_guards(cfg):
    """Placeholder group_2 == rehearsal: rehearsal seeds + no partner file are fine.

    Builds the rehearsal shape EXPLICITLY instead of assuming the live config still has
    it. Once the real §9 match was played, config legitimately names a real partner, and
    the guard then demands the git-ignored players.partner.local.yaml — which exists on a
    developer machine and never in CI. The test passed locally and failed in CI for a
    reason that had nothing to do with the guard it covers.
    """
    rehearsal = copy.deepcopy(cfg)
    rehearsal["wire_match"]["groups"]["group_2"]["name"] = "partner-group"
    rehearsal["wire_match"]["seeds"] = list(_REHEARSAL_SEEDS)
    _real_match_guards(rehearsal)  # must not raise


def test_real_match_refuses_rehearsal_seeds(cfg):
    """A real partner configured + the committed rehearsal seed list -> refuse to play."""
    real = copy.deepcopy(cfg)
    real["wire_match"]["groups"]["group_2"]["name"] = "biu-rl99"
    real["wire_match"]["seeds"] = list(_REHEARSAL_SEEDS)
    with pytest.raises(SystemExit, match="rehearsal list"):
        _real_match_guards(real)


def test_real_match_refuses_missing_partner_intake(cfg, tmp_path, monkeypatch):
    """Real partner + frozen seeds but no players.partner.local.yaml -> refuse."""
    monkeypatch.chdir(tmp_path)  # guaranteed-absent partner file
    real = copy.deepcopy(cfg)
    real["wire_match"]["groups"]["group_2"]["name"] = "biu-rl99"
    real["wire_match"]["seeds"] = [7, 8, 9, 10, 11, 12]
    with pytest.raises(SystemExit, match=r"players\.partner\.local\.yaml"):
        _real_match_guards(real)


def test_build_clients_refuses_an_unset_token_env(cfg, monkeypatch):
    """The client side must fail FAST on a missing bearer, never dial fail-open."""
    for key in ("group_1", "group_2"):
        monkeypatch.delenv(cfg["wire_match"]["groups"][key]["token_env"], raising=False)
    with pytest.raises(ValueError, match="refusing"):
        build_clients(cfg, on_event=None)


def test_resolve_token_returns_the_env_value(cfg, monkeypatch):
    """Happy path: the env var the config NAMES supplies the bearer value."""
    spec = cfg["wire_match"]["groups"]["group_1"]
    monkeypatch.setenv(spec["token_env"], "match-scoped-token")
    assert resolve_token(spec) == "match-scoped-token"


def test_conformance_policy_is_scripted_and_legal_only(cfg):
    """The brief-§2 conformance agent: act()+reset(), legal actions only, no lineup."""
    policy = conformance_policy(cfg, "cop")
    policy.reset()
    rng = np.random.default_rng(0)
    mask = [False, True, False, True, False, False]
    picks = {policy.act([None], [mask], 0.0, rng)[0] for _ in range(40)}
    assert picks <= {1, 3}


def test_serve_main_wires_the_conformance_factory(cfg, monkeypatch):
    """--conformance serves scripted policies; without it the lineup factory (None)."""
    captured: dict = {}

    import scripts.serve_wire_agents as mod  # noqa: PLC0415 — patch target must be the module object

    class _Handle:
        def close(self):
            captured["closed"] = True

    def fake_start(cfg_, keys, policy_factory=None, sdk=None):
        captured["keys"], captured["factory"] = keys, policy_factory
        return [_Handle()]

    monkeypatch.setattr(mod, "load_config", lambda: cfg)
    monkeypatch.setattr(mod, "start_group_agents", fake_start)
    monkeypatch.setattr(mod.time, "sleep", lambda _s: (_ for _ in ()).throw(KeyboardInterrupt))
    serve_main(["group_1", "--conformance"])
    assert captured["keys"] == ["group_1"]
    assert captured["factory"] is not None  # scripted, not the lineup
    scripted = captured["factory"]("group_1", "thief")
    assert callable(scripted.act) and callable(scripted.reset)
    assert captured["closed"] is True  # agents closed on the way out
