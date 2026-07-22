"""Foreign-cop battery UNIT tests — interface, pursuit behavior, oracle privilege."""

from __future__ import annotations

from random import Random

import pytest

from src.marl.env.actions import Action
from src.services.foreign_cops import OracleBfsCop, PursuitCop, foreign_cop_factories
from tests.unit._foreign_cop_fixtures import MOVES_ONLY, make_obs


def test_factories_build_fresh_instances_of_all_three_cops(cfg):
    """The convenience dict exposes all three cops as fresh-instance factories."""
    factories = foreign_cop_factories(cfg)
    assert set(factories) == {"oracle_bfs", "pursuit", "barrier_pursuit"}
    for factory in factories.values():
        one, two = factory(), factory()
        assert one is not two
        one.reset()
        assert callable(one.act)


def test_pursuit_cop_moves_toward_a_visible_thief(cfg):
    """With the thief visible two cells right, the distance-minimising move is RIGHT."""
    action = PursuitCop(cfg).act([make_obs(cfg, (0, 2))], [MOVES_ONLY], 0.0, Random(0))[0]
    assert action == Action.RIGHT


def test_pursuit_cop_returns_a_legal_action_under_a_restrictive_mask(cfg):
    """When only DOWN is legal, the cop picks DOWN even though RIGHT closes faster."""
    only_down = [False, True, False, False, False]
    action = PursuitCop(cfg).act([make_obs(cfg, (0, 2))], [only_down], 0.0, Random(0))[0]
    assert action == Action.DOWN


def test_pursuit_cop_pre_contact_searches_uniformly_over_legal_moves(cfg):
    """Before any sighting the cop samples legal directional moves (not a fixed one)."""
    cop = PursuitCop(cfg)
    rng = Random(1)
    picks = {cop.act([make_obs(cfg)], [MOVES_ONLY], 0.0, rng)[0] for _ in range(30)}
    assert picks <= {Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT}
    assert len(picks) > 1


def test_pursuit_cop_heads_to_the_last_seen_cell_after_the_thief_leaves_view(cfg):
    """The remembered offset is dead-reckoned across own moves and still pulls RIGHT."""
    cop = PursuitCop(cfg)
    assert cop.act([make_obs(cfg, (0, 2))], [MOVES_ONLY], 0.0, Random(0))[0] == Action.RIGHT
    assert cop.act([make_obs(cfg)], [MOVES_ONLY], 0.0, Random(0))[0] == Action.RIGHT


def test_pursuit_cop_rejects_a_multi_agent_call(cfg):
    """The battery drives exactly one cop; two obs is a caller bug."""
    with pytest.raises(ValueError, match="one"):
        PursuitCop(cfg).act([make_obs(cfg), make_obs(cfg)], [MOVES_ONLY] * 2, 0.0, Random(0))


def test_oracle_bfs_cop_requires_the_true_state(cfg):
    """The oracle is privileged: calling it without a GlobalState fails loudly."""
    with pytest.raises(ValueError, match="GlobalState"):
        OracleBfsCop(cfg).act([make_obs(cfg)], [MOVES_ONLY], 0.0, Random(0))


def test_oracle_bfs_cop_plays_the_bfs_step_from_the_true_state(cfg, make_state):
    """Cop at (0,0), thief at (0,4): the BFS shortest-path first step is RIGHT."""
    state = make_state(cop_pos=(0, 0), thief_pos=(0, 4))
    action = OracleBfsCop(cfg).act([make_obs(cfg)], [MOVES_ONLY], 0.0, Random(0), state=state)[0]
    assert action == Action.RIGHT
