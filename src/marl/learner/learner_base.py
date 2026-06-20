"""CTDE QmixLearner base — the exact Double-DQN / BPTT / active-mask update (T4.2).

The centralized BPTT learner shared by the VDN/QMIX cop team (P4b Stage 3). It
owns ALL masking so the P4a mixers stay mask-unaware: per-agent utilities are
multiplied by the episode-constant ``active`` mask BEFORE the mixer, on BOTH the
online and the frozen-target path. The team reward is the active-MEAN over cops
(``dec_pomdp`` shared reward, NEVER a sum); the target is Double-DQN (online
argmax, target value); the loss is a filled-masked Huber; the target net + mixer
are HARD-synced every ``algo.target_update_interval`` updates. Every hyperparameter
is read from config (``algo.*``) / dims — no literals. Global state is consumed
here only and never crosses the MCP boundary.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from src.marl.learner._learner_helpers import (
    bptt_unroll,
    flatten_obs,
    gather_chosen,
    masked_argmax,
    masked_huber,
    to_tensors,
)
from src.marl.mixers.base_mixer import BaseMixer
from src.marl.nets.agent_net import RecurrentQNet


class QmixLearner:
    """Centralized BPTT Double-DQN learner for one cop team behind a mixer."""

    def __init__(self, cfg: dict, role: str, n_agents: int, mixer: BaseMixer | None) -> None:
        """Build the online/target nets + mixers and the two-group AdamW optimizer.

        Args:
            cfg: The loaded config (reads ``algo.*`` hyperparameters + dims).
            role: ``"cop"`` or ``"thief"`` (sets the Q-head action width).
            n_agents: Same-role agent count (cop team size / 1 for thief).
            mixer: The credit-assignment mixer (``VdnMixer``/``QmixMixer``), or
                ``None`` for the IQL branch (no mixer path).
        """
        algo = cfg["algo"]
        self.n_agents = int(n_agents)
        self.gamma = float(algo["gamma"])
        self.huber_delta = float(algo["huber_delta"])
        self.grad_clip_norm = float(algo["grad_clip_norm"])
        self.target_update_interval = int(algo["target_update_interval"])
        self.double_q = bool(algo["double_q"])
        self._updates = 0
        self.online_net = RecurrentQNet(cfg, role, n_agents)
        self.target_net = self.online_net.clone_target()
        self.mixer = mixer
        self.target_mixer = self._frozen_copy(mixer)
        self.optimizer = torch.optim.AdamW(self._param_groups(algo), weight_decay=float(algo["weight_decay"]))

    @staticmethod
    def _frozen_copy(mixer: BaseMixer | None) -> BaseMixer | None:
        """Return a grad-free deepcopy of ``mixer`` (the frozen target mixer)."""
        if mixer is None:
            return None
        target = copy.deepcopy(mixer)
        return target.requires_grad_(False)

    def _param_groups(self, algo: dict) -> list[dict]:
        """Return AdamW param-groups: net @ ``lr_agent`` (+ mixer @ ``lr_mixer``).

        The mixer group is appended ONLY when the mixer carries trainable
        params: a parameter-free mixer (``VdnMixer``) would otherwise create an
        empty AdamW group, so it is skipped.
        """
        groups = [{"params": list(self.online_net.parameters()), "lr": float(algo["lr_agent"])}]
        if self._mixer_params():
            groups.append({"params": self._mixer_params(), "lr": float(algo["lr_mixer"])})
        return groups

    def _mixer_params(self) -> list[Tensor]:
        """Return the mixer's trainable params (empty for a param-free mixer)."""
        if self.mixer is None:
            return []
        return [p for p in self.mixer.parameters() if p.requires_grad]

    @property
    def device(self) -> torch.device:
        """Return the device the online net's parameters live on."""
        return self.online_net.head.weight.device

    @staticmethod
    def _active_mean(per_agent: Tensor, active: Tensor) -> Tensor:
        """Active-MEAN of ``[B,T,N]`` over cops (active ``[B,1,N]``); NEVER a sum."""
        return (per_agent * active).sum(-1) / active.sum(-1).clamp(min=1.0)

    def _team_reward(self, batch: dict) -> Tensor:
        """Return the active-MEAN team reward ``[B, T]`` (NEVER a sum over cops)."""
        data = to_tensors(batch, self.device)
        return self._active_mean(data["reward"], data["active"].float().unsqueeze(1))

    def _unroll(self, net: RecurrentQNet, data: dict) -> Tensor:
        """BPTT-unroll ``net`` over the flattened obs (T+1) -> q ``[B, T+1, N, A]``."""
        obs = flatten_obs(data["obs"], data["scalars"])
        b, n = obs.shape[0], obs.shape[2]
        return bptt_unroll(net, obs, net.initial_hidden(b, n))

    def _mix(self, mixer: BaseMixer, q_agents: Tensor, state: Tensor) -> Tensor:
        """Run ``mixer`` over a ``[B, T, N]`` slice -> joint ``Q_tot [B, T, 1]``."""
        b, t, n = q_agents.shape
        flat = mixer(q_agents.reshape(b * t, n), state.reshape(b * t, -1))
        return flat.view(b, t, 1)

    def _compute_target(self, data: dict, q_online_next: Tensor, q_target_next: Tensor) -> Tensor:
        """Return the centralized Double-DQN bootstrap ``Q_tot_next [B, T, 1]``.

        Online net picks ``a*`` (legal argmax over the next-step q ``[B,T,N,A]``);
        the frozen target net values it; the inactive slots are zeroed before the
        target mixer. Overridden by IQL for the per-agent eq4 (no mixer) target.
        """
        argmax_src = q_online_next if self.double_q else q_target_next
        a_star = masked_argmax(argmax_src, data["next_legal_mask"])
        q_next = gather_chosen(q_target_next, a_star) * data["active"].float().unsqueeze(1)
        return self._mix(self.target_mixer, q_next, data["global_state"][:, 1:])

    def update(self, batch: dict) -> dict:
        """Run one BPTT Double-DQN update; return ``{loss, grad_norm, q_tot}``.

        Args:
            batch: A sampled replay batch (buffer-sample shapes; numpy or tensor).

        Returns:
            Telemetry: scalar ``loss``, total ``grad_norm``, and mean ``q_tot``.
        """
        data = to_tensors(batch, self.device)
        t = data["actions"].shape[1]
        active = data["active"].float().unsqueeze(1)  # [B, 1, N]
        q_online = self._unroll(self.online_net, data)
        with torch.no_grad():
            q_target = self._unroll(self.target_net, data)
        q_chosen = gather_chosen(q_online[:, :t], data["actions"]) * active
        q_tot = self._mix(self.mixer, q_chosen, data["global_state"][:, :t])
        q_next_tot = self._compute_target(data, q_online[:, 1:].detach(), q_target[:, 1:]).detach()
        r_team = self._active_mean(data["reward"], active)
        y = (r_team + self.gamma * (1.0 - data["done"].float()) * q_next_tot.squeeze(-1)).unsqueeze(-1)
        loss = masked_huber(q_tot - y, data["filled"].unsqueeze(-1), self.huber_delta)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm_(self._clip_params(), self.grad_clip_norm)
        self.optimizer.step()
        self._maybe_sync()
        loss_v, q_v = float(loss.detach()), float(q_tot.mean().detach())
        return {"loss": loss_v, "grad_norm": float(grad_norm), "q_tot": q_v}

    def _clip_params(self) -> list[Tensor]:
        """Return every trainable parameter (net + mixer) for grad clipping."""
        params = list(self.online_net.parameters())
        if self.mixer is not None:
            params += list(self.mixer.parameters())
        return params

    def _maybe_sync(self) -> None:
        """HARD-sync target net + mixer from online every ``target_update_interval``."""
        self._updates += 1
        if self._updates % self.target_update_interval != 0:
            return
        self.target_net.load_state_dict(self.online_net.state_dict())
        if self.mixer is not None and self.target_mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
