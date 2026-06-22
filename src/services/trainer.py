"""SelfPlayTrainer — the alternating best-response CTDE training loop (T4.6).

One round: pick the training role (``training_role``), build the trainee's LIVE
acting policy + a FROZEN opponent drawn from the other role's pool, collect
``selfplay.episodes_per_round`` self-play episodes, store BOTH role episodes into
their buffers, and run ``selfplay.update_ratio`` learner updates per episode on
the trainee only. The trainee is then snapshotted into its pool (best-response
history). ε anneals linearly across consumed env steps. The cop buffer/net/learner
use the fixed ``N=2`` width (1-cop stages pad with the ``active`` mask). Compute
thread caps are applied on construction so a full run can never freeze the host.
"""

from __future__ import annotations

from random import Random

import torch

from src.marl.data.schemas import SourceTag
from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.services._trainer_helpers import (
    build_cop_learner,
    linear_epsilon,
    make_role_buffer,
    make_thief_learner,
)
from src.services.episode_pad import pad_episode
from src.services.policy import RecurrentPolicy
from src.services.rollout import collect_episode
from src.services.selfplay import OpponentPool, training_role
from src.utils.compute import apply_compute_limits

_COP_SLOTS = 2  # cop buffer/net/learner fixed agent-axis width (1-cop stages pad via `active`)


class SelfPlayTrainer:
    """Alternating best-response trainer: collect -> store(both) -> update trainee."""

    def __init__(  # noqa: PLR0913 — cfg + seed + the 3 stage dims + 2 optional injected nets
        self,
        cfg: dict,
        seed: int,
        h: int,
        w: int,
        num_cops: int,
        cop_net: object = None,
        thief_net: object = None,
    ) -> None:
        """Build the cop/thief learners, buffers, and opponent pools for a stage.

        Args:
            cfg: Loaded config (``algo`` / ``selfplay`` / ``training`` / dims).
            seed: Master seed (torch + the episode/exploration RNGs + pools).
            h: Board rows for this curriculum stage.
            w: Board columns for this curriculum stage.
            num_cops: Real cop count this stage (acting width; <= ``_COP_SLOTS``).
            cop_net: OPTIONAL pre-built cop net to inject (a carried/OLoRA-wrapped
                net for curriculum transfer / finetune); ``None`` builds dense.
            thief_net: OPTIONAL pre-built thief net to inject; ``None`` builds dense.
        """
        apply_compute_limits(cfg)
        torch.manual_seed(int(seed))
        self._cfg = cfg
        self._env = CopsRobbersEnv(cfg, h=h, w=w, num_cops=int(num_cops))
        self._num_cops = int(num_cops)
        self._rng = Random(int(seed))
        self._mask_w = int(cfg["env"]["actions"]["a_cop"])
        self._batch = int(cfg["algo"]["batch_episodes"])
        selfplay = cfg["selfplay"]
        self._eps_per_round = int(selfplay["episodes_per_round"])
        self._update_ratio = int(selfplay["update_ratio"])
        self._window_k = int(selfplay["window_k"])
        self._cop = build_cop_learner(cfg, _COP_SLOTS, net=cop_net)
        self._thief = make_thief_learner(cfg, net=thief_net)
        self._cop_buf = make_role_buffer(cfg, _COP_SLOTS, int(seed))
        self._thief_buf = make_role_buffer(cfg, 1, int(seed) + 1)
        self._min_replay = int(cfg["replay"]["min_replay_episodes"])  # warmup before the 1st update
        self._cop_pool = OpponentPool("cop", cfg, self._num_cops, int(seed) + 2)
        self._thief_pool = OpponentPool("thief", cfg, 1, int(seed) + 3)
        self._env_steps = 0

    def _policies(self, role: str) -> tuple:
        """Return ``(cop_policy, thief_policy, trainee_learner)`` for a role round."""
        if role == "cop":
            return RecurrentPolicy(self._cop.online_net, self._num_cops), self._thief_pool.sample(), self._cop
        return self._cop_pool.sample(), RecurrentPolicy(self._thief.online_net, 1), self._thief

    def _store(self, out: dict) -> None:
        """Pad + ingest BOTH role episodes from one rollout into their buffers."""
        cop_ep = pad_episode(out["cop"], _COP_SLOTS, self._mask_w, self._cfg)
        thief_ep = pad_episode(out["thief"], 1, self._mask_w, self._cfg)
        self._cop_buf.add_episode(cop_ep, SourceTag.SELF_PLAY)
        self._thief_buf.add_episode(thief_ep, SourceTag.SELF_PLAY)

    def _round(self, round_idx: int) -> dict:
        """Run one best-response round; return ``{round, role, loss, capture_rate}``."""
        role = training_role(round_idx, self._window_k)
        cop_pol, thief_pol, trainee = self._policies(role)
        buf = self._cop_buf if role == "cop" else self._thief_buf
        captures, losses = 0, []
        for _ in range(self._eps_per_round):
            eps = linear_epsilon(self._cfg, self._env_steps)
            episode_seed = self._rng.randrange(2**31)
            out = collect_episode(self._env, cop_pol, thief_pol, self._cfg, episode_seed, eps, self._rng)
            self._store(out)
            self._env_steps += len(out["cop"])
            captures += int(out["capture"])
            if len(buf) >= self._min_replay:  # honor the configured replay warmup
                losses += [trainee.update(buf.sample(self._batch))["loss"] for _ in range(self._update_ratio)]
        (self._cop_pool if role == "cop" else self._thief_pool).add(trainee.online_net)
        return {
            "round": round_idx,
            "role": role,
            "loss": (sum(losses) / len(losses)) if losses else 0.0,  # 0.0 during warmup (no updates)
            "capture_rate": captures / self._eps_per_round,
        }

    def train_stage(self, rounds: int | None = None) -> list[dict]:
        """Run best-response rounds (default ``selfplay.rounds``); return per-round history."""
        total = int(self._cfg["selfplay"]["rounds"]) if rounds is None else int(rounds)
        return [self._round(i) for i in range(total)]

    def cop_net(self):
        """Return the trained cop online net (carried to the next curriculum stage)."""
        return self._cop.online_net

    def thief_net(self):
        """Return the trained thief online net (carried to the next curriculum stage)."""
        return self._thief.online_net
