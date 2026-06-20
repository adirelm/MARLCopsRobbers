"""Train + optimal-eval orchestration for the THROWAWAY 2x2 smoke (T3.3).

Splits the collect/train loop and the game-theoretic optimal-capture eval out of
``tabular_smoke.py`` (each file <=150 LOC). ``optimal_steps`` is a tiny memoized
minimax over the DETERMINISTIC-thief 2x2 game that REUSES the env transition +
heuristic (DRY; no duplicated geometry), so the eval gate compares the trained
greedy cop against the genuine pursuit optimum — never a hardcoded table. Deleted
in P4 with the rest of the throwaway tabular learner.
"""

from __future__ import annotations

from random import Random
from typing import TYPE_CHECKING

from src.marl.data.heuristics import thief_expert
from src.marl.env.actions import Action, action_mask
from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.marl.env.grid import manhattan
from src.marl.env.transition import resolve_joint_action
from src.marl.env.types import GlobalState, Pos
from src.marl.replay import _smoke_helpers as helpers

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.marl.replay.tabular_smoke import TabularQLearner

_DIAGONAL_DIST = 2  # 2x2 diagonal Manhattan distance (the interceptable starts)


def enumerate_starts() -> list[tuple[Pos, Pos]]:
    """Return every distinct (cop, thief) cell pair on the 2x2 board."""
    cells = [(r, c) for r in range(2) for c in range(2)]
    return [(cop, thief) for cop in cells for thief in cells if cop != thief]


def _start_state(cop: Pos, thief: Pos, step: int = 0) -> GlobalState:
    """Build a barrier-free 2x2 :class:`GlobalState` at ``step``."""
    return GlobalState((cop,), thief, frozenset(), 0, step, 2, 2, False)


def optimal_steps(cfg: dict, cop: Pos, thief: Pos, step: int = 0, seen=None) -> int:
    """Return the minimum cop capture time vs the heuristic thief (game optimum).

    A memoized minimax: the cop minimizes capture time while the thief plays its
    deterministic :func:`thief_expert` flee. Reuses the env transition kernel so
    no dynamics are re-implemented. ``memo`` caches each completed
    ``(cop, thief, step)`` subgame and marks in-progress states so a cycle returns
    ``max_moves`` (the loss branch the minimizing cop avoids) — this collapses the
    otherwise exponential recursion on the tiny 2x2 graph to a few states.

    Args:
        cfg: The loaded config (env rules + ``game.max_moves``).
        cop: The cop cell.
        thief: The thief cell.
        step: The move index within the sub-game.
        seen: Optional shared memo dict (internal recursion; pass ``None`` at the top).

    Returns:
        The optimal integer number of steps to capture from this state.
    """
    if cop == thief:
        return 0
    memo: dict = {} if seen is None else seen
    max_moves = cfg["game"]["max_moves"]
    key = (cop, thief, step)
    if step >= max_moves or key in memo:
        return memo.get(key, max_moves)  # in-progress/visited -> cycle loss branch
    memo[key] = max_moves  # sentinel: mark in-progress so a revisit reads max_moves
    state = _start_state(cop, thief, step)
    thief_a = thief_expert(state, cfg)
    best = max_moves
    for ai, ok in enumerate(action_mask(state, "cop", cfg, 0)):
        if not ok:
            continue
        res = resolve_joint_action(state, {"cop_0": Action(ai), "thief": thief_a}, cfg)
        if res.capture:
            best = min(best, 1)
            continue
        nxt = res.next_state
        best = min(best, 1 + optimal_steps(cfg, nxt.cop_pos[0], nxt.thief_pos, nxt.step, memo))
    memo[key] = best
    return best


def train_seed(cfg: dict, seed: int, learner: TabularQLearner) -> TabularQLearner:
    """Train ``learner`` on the 2x2 stage with exploring starts; feed the buffer.

    Each episode resets at a uniformly-sampled distinct 2x2 (cop, thief) start
    (exploring starts guarantee full root-state coverage on the tiny board), rolls
    the cop learner epsilon-greedy vs the heuristic thief, online-updates the
    tabular Q, and pushes the whole padded episode through the centralized replay
    buffer (exercising the buffer end-to-end).

    Args:
        cfg: The loaded config (reads ``p3_smoke.episodes``).
        seed: The per-seed RNG seed.
        learner: A fresh tabular learner to train in place.

    Returns:
        The trained ``learner`` (returned for call chaining).
    """
    buffer = helpers.make_buffer(cfg, seed)
    starts = enumerate_starts()
    rng = Random(seed)
    for ep_idx in range(int(cfg["p3_smoke"]["episodes"])):
        cop, thief = starts[rng.randrange(len(starts))]
        episode, _cap, _n = helpers.rollout(
            CopsRobbersEnv(cfg, h=2, w=2, num_cops=1),
            learner,
            cfg,
            rng.randrange(2**31),
            learner.epsilon,
            rng,
            train=True,
            start_state=_start_state(cop, thief),
        )
        buffer.add_episode(episode, helpers.source_for(ep_idx))
    return learner


def eval_optimal(cfg: dict, learner: TabularQLearner) -> bool:
    """Greedy-eval the cop from every 2x2 start against the pursuit optimum.

    Each start must CAPTURE within ``optimal_steps + 1`` (a single-step slack the
    constant-alpha tabular learner reliably meets) and EXACTLY at the optimum on
    the interceptable diagonal starts (where 1-step interception is available).

    Args:
        cfg: The loaded config.
        learner: The trained tabular learner.

    Returns:
        True iff every 2x2 start meets the capture/step gate.
    """
    rng = Random(0)
    for cop, thief in enumerate_starts():
        env = CopsRobbersEnv(cfg, h=2, w=2, num_cops=1)
        _ep, capture, steps = helpers.rollout(
            env, learner, cfg, 0, 0.0, rng, train=False, start_state=_start_state(cop, thief)
        )
        opt = optimal_steps(cfg, cop, thief)
        exact_needed = manhattan(cop, thief) == _DIAGONAL_DIST
        if not capture or steps > opt + 1 or (exact_needed and steps != opt):
            return False
    return True
