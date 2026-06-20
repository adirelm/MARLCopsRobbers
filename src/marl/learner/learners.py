"""Role learners — Cop (mixed), IQL (per-agent), Thief (single-agent) (T4.2).

Three thin subclasses of :class:`QmixLearner` (P4b Stage 4):

* :class:`CopLearner` — the cooperative cop team behind a mixer; the mixer is
  ``QmixMixer`` when ``algo.name == "qmix"`` (the graded PRIMARY) else
  ``VdnMixer`` (the ablation arm). It reuses the centralized base ``update``.
* :class:`IqlLearner` — the §7.2 non-stationarity BASELINE: independent
  per-agent recurrent Double-DQN (eq4), NO mixer, NO global state, a SINGLE
  AdamW group at ``lr_agent``. It overrides ``update`` for the per-agent target
  and masks the loss by ``filled`` AND ``active``.
* :class:`ThiefLearner` — the adversary: a SINGLE-agent IQL learner with the
  thief head (``a_thief`` = 4); it slices the ``a_cop``-wide env mask down to
  its head width before argmax. It NEVER enters any mixer (the cop/thief
  decomposition boundary, ADR-D1-B/ADR-D3-A).
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from src.marl.learner._learner_helpers import gather_chosen, masked_argmax, masked_huber, to_tensors
from src.marl.learner.learner_base import QmixLearner
from src.marl.mixers.qmix_mixer import QmixMixer
from src.marl.mixers.vdn_mixer import VdnMixer
from src.marl.nets.agent_net import RecurrentQNet
from src.marl.nets.dims import action_dim


class CopLearner(QmixLearner):
    """Cooperative cop-team learner behind the QMIX (or VDN ablation) mixer."""

    def __init__(self, cfg: dict, n_agents: int, net: RecurrentQNet | None = None) -> None:
        """Wire the mixer chosen by ``algo.name`` and defer to the base learner.

        Args:
            cfg: The loaded config (``algo.name`` selects qmix/vdn; dims).
            n_agents: The cop-team size (mixer per-agent input count).
            net: OPTIONAL pre-built online net (an OLoRA-wrapped encoder for the
                finetune path); ``None`` builds a dense net internally.
        """
        mixer = QmixMixer(cfg, n_agents) if cfg["algo"]["mixer"]["type"] == "qmix" else VdnMixer()
        super().__init__(cfg, role="cop", n_agents=n_agents, mixer=mixer, net=net)


class IqlLearner(QmixLearner):
    """Independent per-agent Double-DQN baseline — no mixer, no global state."""

    def __init__(self, cfg: dict, n_agents: int, role: str = "cop", net: RecurrentQNet | None = None) -> None:
        """Build the no-mixer IQL learner (single AdamW group at ``lr_agent``).

        Args:
            cfg: The loaded config (reads ``algo.*`` + dims).
            n_agents: The per-agent learner's agent-axis width.
            role: The role driving the Q-head action width (cop/thief).
            net: OPTIONAL pre-built online net (OLoRA-wrapped for finetune); ``None``
                builds a dense net internally.
        """
        super().__init__(cfg, role=role, n_agents=n_agents, mixer=None, net=net)

    def _slice_legal(self, legal: Tensor) -> Tensor:
        """Return the legality mask unchanged (the thief subclass slices it)."""
        return legal

    def _per_agent_target(self, data: dict, q_online_next: Tensor, q_target_next: Tensor) -> Tensor:
        """Return the per-agent eq4 bootstrap ``Q_i_target(next, a*) [B, T, N]``.

        ``a*`` is the online net's legal argmax over the next step; the frozen
        target net values it. No mixer, no global state (the IQL contract).
        """
        argmax_src = q_online_next if self.double_q else q_target_next
        a_star = masked_argmax(argmax_src, self._slice_legal(data["next_legal_mask"]))
        return gather_chosen(q_target_next, a_star)

    def update(self, batch: dict) -> dict:
        """Run one per-agent BPTT Double-DQN update (eq4); return telemetry.

        Args:
            batch: A sampled replay batch (buffer-sample shapes; numpy or tensor).

        Returns:
            Telemetry: scalar ``loss``, total ``grad_norm``, and mean per-agent Q.
        """
        data = to_tensors(batch, self.device)
        t = data["actions"].shape[1]
        active = data["active"].float().unsqueeze(1)  # [B, 1, N]
        q_online = self._unroll(self.online_net, data)
        with torch.no_grad():
            q_target = self._unroll(self.target_net, data)
        q_chosen = gather_chosen(q_online[:, :t], data["actions"])
        q_next = self._per_agent_target(data, q_online[:, 1:].detach(), q_target[:, 1:]).detach()
        done = data["done"].float().unsqueeze(-1)
        y = data["reward"] + self.gamma * (1.0 - done) * q_next
        mask = data["filled"].float().unsqueeze(-1) * active  # filled AND active
        loss = masked_huber(q_chosen - y, mask, self.huber_delta)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm_(self._clip_params(), self.grad_clip_norm)
        self.optimizer.step()
        self._maybe_sync()
        q_mean = float(((q_chosen * mask).sum() / mask.sum().clamp(min=1.0)).detach())
        return {"loss": float(loss.detach()), "grad_norm": float(grad_norm), "q_tot": q_mean}


class ThiefLearner(IqlLearner):
    """Single-agent adversarial Double-DQN (IQL-style, no mixer, ``a_thief``=4)."""

    def __init__(self, cfg: dict, net: RecurrentQNet | None = None) -> None:
        """Build the single-agent thief learner with the 4-action thief head.

        Args:
            cfg: The loaded config (``env.actions.a_thief`` head width; dims).
            net: OPTIONAL pre-built online net (OLoRA-wrapped for finetune);
                ``None`` builds a dense net internally.
        """
        self._a_thief = action_dim(cfg, "thief")
        super().__init__(cfg, n_agents=1, role="thief", net=net)

    def _slice_legal(self, legal: Tensor) -> Tensor:
        """Slice the ``a_cop``-wide env legality mask down to the thief head width."""
        return legal[..., : self._a_thief]
