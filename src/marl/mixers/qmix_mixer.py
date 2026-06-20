"""QMIX monotone hypernetwork mixer (T4.1, P4a Stage 3).

QMIX (BRIEF eq5) factorises ``Q_tot`` as a monotone function of the per-agent
``Q_i``, conditioned on the global state via hypernetworks. The first-layer
weights ``w1`` and second-layer weights ``w2`` are passed through ``softplus``
so they are non-negative — this enforces ``∂Q_tot/∂Q_i ≥ 0`` (monotonicity),
which in turn guarantees IGM: the decentralized per-agent argmax equals the
joint that maximizes ``Q_tot``. The bias ``b1`` and state-value ``V(s)`` are
sign-free. embed_dim/state_dim come from config — no hardcode. PURE net: no
training loop, optimizer, or replay use (that is P4b).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.functional import elu, softplus

from src.marl.mixers.base_mixer import BaseMixer
from src.marl.nets.dims import state_dim


class QmixMixer(BaseMixer):
    """Monotone state-conditioned hypernetwork mixer (eq5)."""

    def __init__(self, cfg: dict, n_agents: int) -> None:
        """Build the hypernetworks producing the monotone two-layer mixer.

        Args:
            cfg: The loaded config (reads ``algo.mixer.embed_dim`` and the env
                dims for :func:`src.marl.nets.dims.state_dim`).
            n_agents: Number of per-agent ``Q_i`` inputs the mixer combines.
        """
        super().__init__()
        self.n_agents = int(n_agents)
        self.embed_dim = int(cfg["algo"]["mixer"]["embed_dim"])
        s_dim = state_dim(cfg)
        self.hyper_w1 = nn.Linear(s_dim, self.n_agents * self.embed_dim)
        self.hyper_b1 = nn.Linear(s_dim, self.embed_dim)
        self.hyper_w2 = nn.Linear(s_dim, self.embed_dim)
        self.value = nn.Sequential(
            nn.Linear(s_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, q_agents: Tensor, state: Tensor) -> Tensor:
        """Combine per-agent ``Q_i`` into a monotone joint ``Q_tot``.

        Args:
            q_agents: Per-agent Q values ``[B, n]`` (one chosen-action Q each).
            state: The global state ``[B, state_dim]`` conditioning the mixer.

        Returns:
            The joint value ``q_tot`` shaped ``[B, 1]``.
        """
        batch = state.shape[0]
        w1 = softplus(self.hyper_w1(state)).view(batch, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(batch, 1, self.embed_dim)
        w2 = softplus(self.hyper_w2(state)).view(batch, self.embed_dim, 1)
        v = self.value(state).view(batch, 1, 1)
        hidden = elu(torch.bmm(q_agents.unsqueeze(1), w1) + b1)
        return (torch.bmm(hidden, w2) + v).view(batch, 1)
