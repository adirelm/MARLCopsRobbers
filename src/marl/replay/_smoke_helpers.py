"""Collection / training / eval helpers for the THROWAWAY 2x2 smoke (T3.3).

Keeps ``tabular_smoke.py`` <=150 LOC. These helpers roll the cop learner (epsilon
-greedy) against the heuristic thief on the 2x2 stage, push each whole episode
into the :class:`CentralizedReplayBuffer` (exercising the buffer end-to-end) and
feed the per-step transitions to the tabular learner. They are throwaway scaffolding
deleted in P4 alongside the learner. Nothing is hardcoded — dims come from config.
"""

from __future__ import annotations

from random import Random
from typing import TYPE_CHECKING

import numpy as np

from src.marl.data.heuristics import thief_expert
from src.marl.data.obs_encoder import encode_obs_batch, encode_state
from src.marl.data.schemas import SourceTag
from src.marl.env import _env_helpers as env_helpers
from src.marl.env.actions import Action
from src.marl.replay.episode_buffer import CentralizedReplayBuffer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.marl.env.cops_robbers_env import CopsRobbersEnv
    from src.marl.replay.tabular_smoke import TabularQLearner


def _force_start(env: CopsRobbersEnv, start_state, cfg: dict) -> tuple[dict, dict]:
    """Override the env spawn with ``start_state`` and rebuild obs/mask dicts."""
    env._state = start_state
    env._memory = env_helpers.fresh_memory(1)
    env_helpers.update_memory(env._memory, start_state, cfg)
    obs = env_helpers.build_obs_dict(start_state, env._memory, 1, cfg)
    info = {"action_mask": env_helpers.build_mask_dict(start_state, 1, cfg)}
    return obs, info


def _cop_action(learner: TabularQLearner, s_key: tuple, mask, rng: Random, epsilon: float) -> Action:
    """Return the cop's epsilon-greedy action over its legal mask."""
    legal = [bool(b) for b in mask]
    if rng.random() < epsilon:
        choices = [i for i, ok in enumerate(legal) if ok]
        return Action(rng.choice(choices))
    return learner.greedy_action(s_key, "cop_0", legal)


def rollout(  # noqa: PLR0913 - explicit collection knobs for the throwaway smoke
    env: CopsRobbersEnv,
    learner: TabularQLearner,
    cfg: dict,
    seed: int,
    epsilon: float,
    rng: Random,
    train: bool,
    start_state=None,
) -> tuple[dict, bool, int]:
    """Roll one 2x2 episode (cop=learner, thief=heuristic); optionally train.

    Returns the padded episode dict (for the buffer), the capture flag, and the
    number of real steps taken. When ``train`` is True each transition updates the
    tabular learner online (standard Q-learning) using the cop reward. An optional
    ``start_state`` overrides the seeded spawn (used by the fixed-start eval).
    """
    # Local import breaks the tabular_smoke -> _smoke_solve -> _smoke_helpers cycle (DRY).
    from src.marl.replay.tabular_smoke import state_key  # noqa: PLC0415

    obs, info = env.reset(seed=seed)
    if start_state is not None:
        obs, info = _force_start(env, start_state, cfg)
    steps: list[dict] = []
    capture = False
    terminated = False
    while not terminated:
        state = env.state()
        s_key = state_key(state)
        mask = info["action_mask"]["cop_0"]
        cop_a = _cop_action(learner, s_key, mask, rng, epsilon if train else 0.0)
        joint = {"cop_0": cop_a, "thief": thief_expert(state, cfg)}
        nxt_obs, reward, terminated, info = env.step(joint, eval_mode=not train)
        nxt_state = env.state()
        next_mask = [bool(b) for b in info["action_mask"]["cop_0"]]
        if train:
            learner.update(
                s_key, "cop_0", int(cop_a), reward["cop_0"], state_key(nxt_state), next_mask, done=terminated
            )
        steps.append(
            {
                "obs": obs,
                "nxt": nxt_obs,
                "state": state,
                "a": int(cop_a),
                "r": reward["cop_0"],
                "done": terminated,
                "mask": next_mask,
            }
        )
        obs = nxt_obs
        capture = bool(info.get("capture"))
    return _pad_episode(steps, env, cfg), capture, len(steps)


def _pad_episode(steps: list[dict], env: CopsRobbersEnv, cfg: dict) -> dict:
    """Build a buffer-schema episode dict (N=1 cop axis) padded to ``t_max``."""
    t = len(steps)
    t_max = cfg["game"]["max_moves"]
    a_cop = cfg["env"]["actions"]["a_cop"]
    imgs = [encode_obs_batch([s["obs"]["cop_0"]])[0][0] for s in steps]
    imgs.append(encode_obs_batch([steps[-1]["nxt"]["cop_0"]])[0][0])
    scal = [encode_obs_batch([s["obs"]["cop_0"]])[1][0] for s in steps]
    scal.append(encode_obs_batch([steps[-1]["nxt"]["cop_0"]])[1][0])
    states = [encode_state(s["state"], cfg) for s in steps]
    states.append(encode_state(env.state(), cfg))
    c, w = cfg["env"]["obs_channels"], 2 * cfg["env"]["view_radius_max"] + 1
    ns = cfg["env"]["obs_scalars"]
    ep = {
        "obs": np.zeros((t_max + 1, 1, c, w, w), np.float32),
        "scalars": np.zeros((t_max + 1, 1, ns), np.float32),
        "global_state": np.zeros((t_max + 1, encode_state(env.state(), cfg).shape[0]), np.float32),
        "actions": np.zeros((t_max, 1), np.int64),
        "reward": np.zeros((t_max, 1), np.float32),
        "done": np.zeros((t_max,), bool),
        "filled": np.zeros((t_max,), bool),
        "next_legal_mask": np.zeros((t_max, 1, a_cop), bool),
        "hidden_seed": np.int64(0),
    }
    for i in range(t + 1):
        ep["obs"][i, 0], ep["scalars"][i, 0], ep["global_state"][i] = imgs[i], scal[i], states[i]
    for i in range(t):
        ep["actions"][i, 0], ep["reward"][i, 0] = steps[i]["a"], steps[i]["r"]
        ep["done"][i], ep["filled"][i] = steps[i]["done"], True
        ep["next_legal_mask"][i, 0] = np.asarray(steps[i]["mask"], bool)
    return ep


def make_buffer(cfg: dict, seed: int) -> CentralizedReplayBuffer:
    """Build the N=1 cop replay buffer for the smoke run (dims from config)."""
    env_cfg = cfg["env"]
    return CentralizedReplayBuffer(
        capacity=cfg["replay"]["buffer_episodes"],
        t_max=cfg["game"]["max_moves"],
        n_agents=1,
        obs_channels=env_cfg["obs_channels"],
        w_v=2 * env_cfg["view_radius_max"] + 1,
        obs_scalars=env_cfg["obs_scalars"],
        state_dim=3 * (2 * env_cfg["view_radius_max"] + 1) ** 2 + 2,
        n_actions=env_cfg["actions"]["a_cop"],
        seed=seed,
    )


def source_for(episode_idx: int) -> SourceTag:
    """Alternate RANDOM/LIVE_CTDE provenance so the buffer mixes sources."""
    return SourceTag.RANDOM if episode_idx % 2 == 0 else SourceTag.LIVE_CTDE
