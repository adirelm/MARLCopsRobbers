"""RED->GREEN tests for the §9 match policies — obs-driven flee + auto-adaptive thief.

The match runtime serves ONLY local observations over MCP, so the primary flee thief
must act from the egocentric obs (no god-view). The adaptive wrapper switches to the
trained net PERMANENTLY once the opponent cop is seen placing barriers — §3 forbids
mid-match operator intervention, so the switch must be autonomous.
"""

from __future__ import annotations

from random import Random

import numpy as np
import pytest

from src.marl.env.actions import Action
from src.sdk.sdk import MarlSDK
from src.services.bonus_policies import AdaptiveThiefPolicy, ObsFleePolicy
from src.utils.config_loader import load_config

_CFG = load_config()
_R = int(_CFG["env"]["view_radius_max"])  # egocentric window center offset
_W = 2 * _R + 1
_ALL = [True] * 5


def _obs(other_rel: tuple | None = None, barrier_rel: tuple | None = None) -> dict:
    """Craft a local Observation: optional opponent / barrier at a RELATIVE offset."""
    image = np.zeros((int(_CFG["env"]["obs_channels"]), _W, _W), dtype=np.float32)
    image[0, _R, _R] = 1.0  # self
    if other_rel is not None:
        image[1, _R + other_rel[0], _R + other_rel[1]] = 1.0
    if barrier_rel is not None:
        image[2, _R + barrier_rel[0], _R + barrier_rel[1]] = 1.0
    scalars = np.zeros(int(_CFG["env"]["obs_scalars"]), dtype=np.float32)
    return {"image": image, "scalars": scalars}


def test_flee_maximizes_distance_from_the_visible_cop():
    """Cop below (rel +2,0) -> UP is the unique distance-maximizing legal move."""
    policy = ObsFleePolicy(_CFG)
    policy.reset()
    action = policy.act([_obs(other_rel=(2, 0))], [_ALL], 0.0, Random(0))[0]
    assert action == Action.UP


def test_flee_remembers_last_seen_and_updates_for_own_motion():
    """A cop seen to the right keeps steering the thief AWAY even after visual contact is lost.

    Manhattan flight ties are broken first-in-order (UP first — the same semantics as the
    evaluated 0/180 heuristic); the memory test constrains the mask to LEFT/RIGHT so the
    remembered rightward cop deterministically forces LEFT (no-memory would be a coin flip).
    """
    policy = ObsFleePolicy(_CFG)
    policy.reset()
    first = policy.act([_obs(other_rel=(0, 2))], [_ALL], 0.0, Random(0))[0]
    assert first == Action.UP  # d=3 tie broken first-in-order
    lr_only = [False, False, True, True, False]
    second = policy.act([_obs()], [lr_only], 0.0, Random(0))[0]
    assert second == Action.LEFT  # memory says the cop is to the RIGHT -> flee LEFT


def test_flee_unseen_picks_a_legal_directional_move():
    """Never-seen cop -> a random LEGAL directional move (never PLACE_BARRIER)."""
    policy = ObsFleePolicy(_CFG)
    policy.reset()
    mask = [False, False, False, True, True]  # only RIGHT legal among moves
    action = policy.act([_obs()], [mask], 0.0, Random(0))[0]
    assert action == Action.RIGHT


def test_adaptive_switches_permanently_on_a_seen_barrier():
    """Flee until a barrier appears in view -> then the trained net acts, stickily."""
    net = MarlSDK(_CFG).fresh_net("thief")
    policy = AdaptiveThiefPolicy(_CFG, net)
    policy.reset()
    pre = policy.act([_obs(other_rel=(2, 0))], [_ALL], 0.0, Random(0))[0]
    assert pre == Action.UP and policy.switched is False  # flee semantics pre-switch
    policy.act([_obs(barrier_rel=(1, 1))], [_ALL], 0.0, Random(0))
    assert policy.switched is True
    policy.reset()  # a new sub-game resets hidden/memory but NOT the switch
    assert policy.switched is True
    action = policy.act([_obs()], [_ALL], 0.0, Random(0))[0]
    assert action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)


def test_build_policy_passes_a_ready_policy_through_and_resets_it():
    """The SDK acting seam accepts a prebuilt policy object (the §9 injection path)."""

    class _Stub:
        def __init__(self) -> None:
            self.resets = 0

        def reset(self) -> None:
            self.resets += 1

        def act(self, obs_list, masks, eps, rng, state=None):
            return [Action.UP]

    stub = _Stub()
    out = MarlSDK(_CFG).build_policy("thief", stub, n_agents=1)
    assert out is stub and stub.resets == 1


def test_flee_rejects_multi_agent_use():
    """The flee policy drives exactly one thief; n>1 obs lists are a usage error."""
    policy = ObsFleePolicy(_CFG)
    policy.reset()
    with pytest.raises(ValueError):
        policy.act([_obs(), _obs()], [_ALL, _ALL], 0.0, Random(0))
