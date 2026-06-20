"""Private wiring for the MarlSDK facade (keeps sdk.py <=150 LOC, T0.7).

Holds the generic episode-collection loop (arbitrary cop/thief policies) and the
P3 artifact builders (a minimal sub-game JSON record + a headless god-view
``render_state`` dict). All business logic is DELEGATED to ``src.marl.*`` here:
episode padding reuses :func:`src.marl.replay._smoke_helpers._pad_episode`, the
greedy eval reuses the throwaway tabular learner + the env transition kernel, and
the spectator dict reuses :func:`src.marl.env.render_state.render_state` (the
sanctioned no-GlobalState seam). The SDK is the SINGLE business entry (§4).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.marl.env.render_state import render_state
from src.marl.env.scorer import Scorer
from src.marl.replay import _smoke_helpers as smoke
from src.marl.replay import _smoke_solve as solve
from src.marl.replay.tabular_smoke import TabularQLearner

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.marl.env.actions import Action
    from src.marl.env.types import Pos

Policy = Callable[..., "Action"]


def make_env(cfg: dict, h: int | None, w: int | None, num_cops: int | None) -> CopsRobbersEnv:
    """Construct a P2 :class:`CopsRobbersEnv` (config defaults when args None)."""
    return CopsRobbersEnv(cfg, h=h, w=w, num_cops=num_cops)


def run_episode(env: CopsRobbersEnv, cop: Policy, thief: Policy, seed: int, cfg: dict) -> dict:
    """Roll one episode with arbitrary policies into a padded buffer-schema dict.

    Each policy is called ``policy(state, mask, cfg) -> Action`` with the train-only
    global state (collection is a TRAIN-time activity) and the agent's legal mask.
    The per-step records are padded to ``game.max_moves`` via the shared smoke
    padder (DRY; identical N=1 cop schema the replay buffer ingests).

    Args:
        env: A reset-able single-cop env.
        cop: The cop policy callable.
        thief: The thief policy callable.
        seed: The spawn seed (reproducible episode).
        cfg: The loaded config.

    Returns:
        The padded episode dict matching the ``CentralizedReplayBuffer`` schema.
    """
    obs, info = env.reset(seed=seed)
    steps: list[dict] = []
    terminated = False
    while not terminated:
        state = env.state()
        mask = info["action_mask"]
        joint = {"cop_0": cop(state, mask["cop_0"], cfg), "thief": thief(state, mask["thief"], cfg)}
        nxt_obs, reward, terminated, info = env.step(joint)
        next_mask = [bool(b) for b in info["action_mask"]["cop_0"]]
        steps.append(
            {
                "obs": obs,
                "nxt": nxt_obs,
                "state": state,
                "a": int(joint["cop_0"]),
                "r": reward["cop_0"],
                "done": terminated,
                "mask": next_mask,
            }
        )
        obs = nxt_obs
    return smoke._pad_episode(steps, env, cfg)


def _train_eval_learner(cfg: dict, seed: int) -> TabularQLearner:
    """Train one throwaway 2x2 tabular cop learner for the artifact eval episode."""
    return solve.train_seed(cfg, seed, TabularQLearner(cfg))


def _greedy_eval_episode(cfg: dict, learner: TabularQLearner) -> tuple[CopsRobbersEnv, bool, int]:
    """Greedy-eval the trained cop from a fixed diagonal 2x2 start (capture demo).

    Reuses the smoke ``rollout`` (cop=learner greedy, thief=heuristic) so no
    geometry/transition logic is duplicated; the diagonal start guarantees a
    single-step interceptable capture for the artifact sub-game record.
    """
    start: tuple[Pos, Pos] = ((0, 0), (1, 1))
    env = make_env(cfg, h=2, w=2, num_cops=1)
    _ep, capture, steps = smoke.rollout(
        env,
        learner,
        cfg,
        0,
        0.0,
        smoke.Random(0),
        train=False,
        start_state=solve._start_state(*start),
    )
    return env, capture, steps


def build_p3_artifacts(cfg: dict, seeds, game_id: str) -> tuple[dict, dict]:
    """Build a minimal sub-game record + a god-view render dict from a greedy run.

    Trains the throwaway tabular learner on the first seed, plays one greedy 2x2
    capture episode, and derives the report-only :class:`Scorer` points. Returns
    a ``(subgame_record, render_state)`` pair (the render dict carries NO
    GlobalState — the sanctioned spectator seam).

    Args:
        cfg: The loaded config.
        seeds: The smoke seeds (the first drives the artifact episode).
        game_id: The sub-game identifier string.

    Returns:
        ``(record, god_view)`` plain JSON-serializable dicts.
    """
    seed = int(next(iter(seeds)))
    learner = _train_eval_learner(cfg, seed)
    env, capture, steps = _greedy_eval_episode(cfg, learner)
    winner = "cop" if capture else "thief"
    scores = Scorer(cfg).score(winner)
    record = {
        "game_id": game_id,
        "grid": [2, 2],
        "winner": winner,
        "capture": bool(capture),
        "steps": int(steps),
        "scores": {"cop": int(scores["cop"]), "thief": int(scores["thief"])},
        "seed": seed,
    }
    return record, render_state(env.state(), cfg)
