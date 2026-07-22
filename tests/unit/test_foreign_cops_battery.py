"""Foreign-cop battery INTEGRATION tests — barrier budget, capture, state passthrough."""

from __future__ import annotations

from itertools import pairwise
from random import Random

from src.marl.env.actions import Action
from src.marl.env.types import GlobalState
from src.services.bonus_policies import ObsFleePolicy
from src.services.foreign_cops import BarrierPursuitCop, foreign_cop_factories
from src.services.matchup_eval import play_matchup
from tests.unit._foreign_cop_fixtures import MOVES_ONLY, STAGE, WITH_PLACE, UpThief, make_obs


def test_barrier_pursuit_cop_places_within_cap_and_never_twice_in_a_row(cfg):
    """Placements happen, stay <= game.max_barriers, and alternate with pursuit moves."""
    cop = BarrierPursuitCop(cfg)
    actions = [cop.act([make_obs(cfg, (0, 2))], [WITH_PLACE], 0.0, Random(0))[0] for _ in range(20)]
    places = [a == Action.PLACE_BARRIER for a in actions]
    assert 1 <= sum(places) <= cfg["game"]["max_barriers"]
    assert not any(a and b for a, b in pairwise(places))


def test_barrier_pursuit_cop_never_places_when_the_mask_forbids_it(cfg):
    """With PLACE_BARRIER masked out (budget spent), the cop only pursues."""
    cop = BarrierPursuitCop(cfg)
    actions = [cop.act([make_obs(cfg, (0, 2))], [MOVES_ONLY], 0.0, Random(0))[0] for _ in range(20)]
    assert Action.PLACE_BARRIER not in actions


def test_barrier_pursuit_cop_places_even_when_adjacent(cfg):
    """Adjacent, a move-in never converts vs a competent thief (0/180): drop the barrier.

    The adjacent drop lands inside the thief's view, so it is also the reliable
    trigger for AdaptiveThiefPolicy's barrier switch (7/10 vs 1/10 in the probe).
    """
    action = BarrierPursuitCop(cfg).act([make_obs(cfg, (0, 1))], [WITH_PLACE], 0.0, Random(0))[0]
    assert action == Action.PLACE_BARRIER


def test_barrier_pursuit_cop_stays_within_budget_over_a_full_game(cfg):
    """A full 5x5 sub-game never records more PLACE_BARRIER actions than the cap."""
    _captured, tally, _moves = play_matchup(cfg, BarrierPursuitCop(cfg), ObsFleePolicy(cfg), 7, STAGE)
    assert tally.get("PLACE_BARRIER", 0) <= cfg["game"]["max_barriers"]


def test_oracle_bfs_cop_captures_a_stationary_thief_within_the_move_cap(cfg):
    """On a tiny board the oracle runs down an (eventually) stationary thief."""
    oracle = foreign_cop_factories(cfg)["oracle_bfs"]()
    captured, _tally, moves = play_matchup(cfg, oracle, UpThief(), 11, (3, 3, 1))
    assert captured
    assert moves <= cfg["game"]["max_moves"]


def test_play_matchup_passes_the_global_state_to_the_cop(cfg):
    """The eval loop hands the cop the sanctioned train-only state every tick."""
    seen: list[object] = []

    class _Recorder:
        def reset(self) -> None:
            """No state."""

        def act(self, _obs, masks, _eps, _rng, state=None):
            seen.append(state)
            return [next(i for i, ok in enumerate(masks[0]) if ok)]

    play_matchup(cfg, _Recorder(), UpThief(), 3, (2, 2, 1))
    assert seen
    assert all(isinstance(s, GlobalState) for s in seen)


def test_foreign_cop_matchups_are_deterministic(cfg):
    """Same seed + same cop arm => bit-identical (captured, tally, moves) twice."""
    for name, factory in foreign_cop_factories(cfg).items():
        runs = [play_matchup(cfg, factory(), ObsFleePolicy(cfg), 5, STAGE) for _ in range(2)]
        assert runs[0] == runs[1], name
