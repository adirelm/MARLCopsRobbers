"""Construction + schedule helpers for :class:`SelfPlayTrainer` (keeps it <=150 LOC).

Builds the per-role replay buffers and learners and computes the per-round linear
ε anneal. The cop learner is the QMIX/VDN :class:`CopLearner` unless ``algo.name``
is ``"iql"`` (the §7.2 baseline → :class:`IqlLearner`); the thief is always the
single-agent adversarial :class:`ThiefLearner`. Both buffers store the env's
uniform ``a_cop``-wide legality mask (the thief learner slices it). Dims are
config-derived (no hardcode).
"""

from __future__ import annotations

from src.marl.learner.learners import CopLearner, IqlLearner, ThiefLearner
from src.marl.nets.agent_net import RecurrentQNet
from src.marl.replay.episode_buffer import CentralizedReplayBuffer


def make_role_buffer(cfg: dict, n_agents: int, seed: int) -> CentralizedReplayBuffer:
    """Build a role replay buffer (state on the T+1 axis; a_cop-wide mask)."""
    env = cfg["env"]
    w_v = 2 * int(env["view_radius_max"]) + 1
    return CentralizedReplayBuffer(
        capacity=int(cfg["replay"]["buffer_episodes"]),
        t_max=int(cfg["game"]["max_moves"]),
        n_agents=int(n_agents),
        obs_channels=int(env["obs_channels"]),
        w_v=w_v,
        obs_scalars=int(env["obs_scalars"]),
        state_dim=3 * w_v**2 + 2,
        n_actions=int(env["actions"]["a_cop"]),
        seed=int(seed),
    )


def build_cop_learner(cfg: dict, n_agents: int, net: RecurrentQNet | None = None):
    """Return the cop learner: :class:`CopLearner` (QMIX/VDN) or IQL baseline.

    Passes an OPTIONAL pre-built ``net`` (an OLoRA-wrapped encoder for the
    finetune path) straight through to the learner's injection seam.
    """
    if cfg["algo"]["name"] == "iql":
        return IqlLearner(cfg, n_agents=int(n_agents), role="cop", net=net)
    return CopLearner(cfg, n_agents=int(n_agents), net=net)


def make_thief_learner(cfg: dict, net: RecurrentQNet | None = None) -> ThiefLearner:
    """Return the single-agent adversarial thief learner (optional injected net)."""
    return ThiefLearner(cfg, net=net)


def linear_epsilon(cfg: dict, env_steps: int) -> float:
    """Linear ε anneal ``eps_start -> eps_end`` over ``training.eps_anneal_env_steps``."""
    tr = cfg["training"]
    start, end = float(tr["eps_start"]), float(tr["eps_end"])
    frac = min(1.0, env_steps / max(1, int(tr["eps_anneal_env_steps"])))
    return start + (end - start) * frac
