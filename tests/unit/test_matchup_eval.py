"""matchup_eval service — DI-tested with scripted doubles on a tiny board (fast)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from src.services.matchup_eval import UniformRandomPolicy, evaluate_matchup, play_matchup, run_arm
from src.utils.config_loader import load_config

_TINY = (2, 2, 1)


@pytest.fixture(scope="module")
def cfg():
    """One loaded config for the module (read-only)."""
    return load_config()


class _FirstLegal:
    """Deterministic scripted policy: always the first legal action."""

    def reset(self) -> None:
        """No state."""

    def act(self, _obs, masks, _eps, _rng, state=None):
        return [next(i for i, ok in enumerate(masks[0]) if ok)]


class _EpsRecorder(_FirstLegal):
    """Records every epsilon it is offered — proves the eps plumbing."""

    def __init__(self) -> None:
        self.seen: list[float] = []

    def act(self, obs, masks, eps, rng, state=None):
        self.seen.append(float(eps))
        return super().act(obs, masks, eps, rng, state)


def test_play_matchup_returns_outcome_actions_and_moves(cfg):
    """One tiny sub-game yields a bool, a non-empty action tally, and >=1 move."""
    captured, tally, moves = play_matchup(cfg, _FirstLegal(), _FirstLegal(), 3, _TINY)
    assert isinstance(captured, bool)
    assert moves >= 1
    assert sum(tally.values()) == moves  # one cop action recorded per move


def test_evaluate_matchup_is_deterministic(cfg):
    """Two identical calls give identical summaries (the §12 reproducibility property)."""
    args = (cfg, _FirstLegal, _FirstLegal, 3, 11)
    assert evaluate_matchup(*args, stage=_TINY) == evaluate_matchup(*args, stage=_TINY)


def test_evaluate_matchup_forwards_thief_eps_to_the_thief_only(cfg):
    """The thief sees the injected eps every tick; the cop always stays greedy (0.0)."""
    thief = _EpsRecorder()
    cop = _EpsRecorder()
    play_matchup(cfg, cop, thief, 5, _TINY, thief_eps=0.25)
    assert set(thief.seen) == {0.25}
    assert set(cop.seen) == {0.0}


def test_uniform_random_policy_only_picks_legal_actions():
    """The control floor never emits a masked-out action."""
    rng = np.random.default_rng(0)
    mask = [False, True, False, True, False, False]
    picks = {UniformRandomPolicy().act([None], [mask], 0.0, rng)[0] for _ in range(50)}
    assert picks <= {1, 3}


def test_run_arm_rejects_an_unknown_thief_kind(cfg):
    """A typo'd arm name fails loudly, listing the valid kinds."""

    class _SdkStub:
        def serving_net(self, role, n_agents=None):
            return object()

    with pytest.raises(ValueError, match=r"flee.*net.*random"):
        run_arm(_SdkStub(), cfg, "adaptive")


def test_run_arm_wires_the_serving_nets_through_the_sdk(cfg):
    """`run_arm` builds its policies via the injected SDK seam (DI, V3 §16)."""
    calls: list[tuple] = []

    class _SdkStub:
        def serving_net(self, role, n_agents=None):
            calls.append(("serving_net", role))
            return f"net:{role}"

        def build_policy(self, role, net, n_agents=1):
            calls.append(("build_policy", role))
            return _FirstLegal()

    small = copy.deepcopy(cfg)
    small["matchup_eval"] = {"n_games": 2, "base_seed": 0}
    out = run_arm(_SdkStub(), small, "net")
    assert out["games"] == 2
    assert ("serving_net", "cop") in calls
    assert ("serving_net", "thief") in calls
    assert ("build_policy", "cop") in calls
