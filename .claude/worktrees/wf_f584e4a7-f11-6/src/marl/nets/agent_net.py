"""Shared recurrent agent Q-net (T4.0b, P4a Stage 1).

One ``RecurrentQNet`` per role (cop/thief), weight-shared across same-role
agents (BRIEF eq 8: ``z_t = f_phi(z_{t-1}, o_t)``). The CALLER passes the
already-flattened local obs ``[B, n, obs_dim]`` (the learner flattens the
image+scalars into ``obs_dim``); the net then concatenates an agent-id one-hot
(so same-role agents stay distinguishable under shared weights), runs a flat-MLP
encoder, a per-agent ``GRUCell``, then a plain role Q-head. All dims come from
:mod:`src.marl.nets.dims` — no literals. This is a PURE net: no training loop,
optimizer, or replay use (that is P4b).
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from src.marl.nets.dims import action_dim, hidden_dim, obs_dim


def _build_encoder(in_dim: int, widths: list[int]) -> nn.Sequential:
    """Build a flat-MLP encoder of ``Linear -> ReLU`` blocks (OLoRA-wrappable).

    Args:
        in_dim: Encoder input width (obs_dim + n_agents agent-id one-hot).
        widths: The ``nets.encoder_hidden`` Linear widths (e.g. ``[64, 64]``).

    Returns:
        An ``nn.Sequential`` of stacked ``nn.Linear`` + ``nn.ReLU`` layers.
    """
    layers: list[nn.Module] = []
    prev = in_dim
    for width in widths:
        layers.append(nn.Linear(prev, width))
        layers.append(nn.ReLU())
        prev = width
    return nn.Sequential(*layers)


class RecurrentQNet(nn.Module):
    """Weight-shared recurrent Q-net for one role over ``n_agents`` agents."""

    def __init__(self, cfg: dict, role: str, n_agents: int) -> None:
        """Build the encoder, per-agent GRU cell, and role Q-head.

        Args:
            cfg: The loaded config (reads ``nets.encoder_hidden``, dims).
            role: ``"cop"`` or ``"thief"`` (sets the Q-head action width).
            n_agents: Number of same-role agents sharing these weights; also the
                width of the prepended agent-id one-hot.
        """
        super().__init__()
        self.n_agents = int(n_agents)
        self._hidden = hidden_dim(cfg)
        widths = [int(width) for width in cfg["nets"]["encoder_hidden"]]
        enc_in = obs_dim(cfg) + self.n_agents
        self.encoder = _build_encoder(enc_in, widths)
        self.gru = nn.GRUCell(widths[-1], self._hidden)
        self.head = nn.Linear(self._hidden, action_dim(cfg, role))

    def _agent_ids(self, batch: int, n: int, device: torch.device) -> Tensor:
        """Return a ``[B,n,n_agents]`` per-agent one-hot identity block."""
        eye = torch.eye(self.n_agents, device=device)
        return eye[:n].unsqueeze(0).expand(batch, n, self.n_agents)

    def forward(self, obs: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        """Run one recurrent step over all agents in the merged ``(B*n)`` batch.

        Args:
            obs: Local observations ``[B, n, obs_dim]``.
            h: Previous hidden state ``[B, n, H]``.

        Returns:
            A ``(q, h')`` pair with ``q`` shaped ``[B, n, A]`` (A = role actions)
            and ``h'`` the next hidden state ``[B, n, H]``.
        """
        batch, n, _ = obs.shape
        ids = self._agent_ids(batch, n, obs.device)
        x = torch.cat([obs, ids], dim=-1).reshape(batch * n, -1)
        enc = self.encoder(x)
        h_next = self.gru(enc, h.reshape(batch * n, self._hidden))
        q = self.head(h_next)
        return q.view(batch, n, -1), h_next.view(batch, n, self._hidden)

    def initial_hidden(self, batch: int, n: int) -> Tensor:
        """Return a zero hidden state ``[B, n, H]`` on the net's parameter device."""
        device = self.head.weight.device
        return torch.zeros(batch, n, self._hidden, device=device)

    def clone_target(self) -> RecurrentQNet:
        """Return a detached deep copy for use as a frozen target net.

        Returns:
            A ``RecurrentQNet`` with identical parameter values, all with
            ``requires_grad=False`` (the learner owns soft/hard updates, P4b).
        """
        target = copy.deepcopy(self)
        return target.requires_grad_(False)
