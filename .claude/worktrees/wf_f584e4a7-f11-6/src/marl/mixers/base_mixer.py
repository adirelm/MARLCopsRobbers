"""Mixer ABC — the swappable credit-assignment seam (T4.0a, P4a Stage 2).

A mixer maps per-agent utilities ``q_agents[B,n]`` and the global state
``state[B,state_dim]`` to a joint ``q_tot[B,1]``. VDN (eq6) ignores the state;
QMIX (T4.1) consumes it through a monotone hypernetwork; IQL drops the mixer
entirely (a learner branch in P4b). This base fixes the forward contract so the
learner can swap implementations behind one interface. PURE net: no training.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class BaseMixer(nn.Module, ABC):
    """Abstract mixer combining per-agent Q values into a joint Q_tot."""

    @abstractmethod
    def forward(self, q_agents: Tensor, state: Tensor) -> Tensor:
        """Combine per-agent utilities into a single joint value.

        Args:
            q_agents: Per-agent Q values ``[B, n]`` (one chosen-action Q each).
            state: The global state ``[B, state_dim]`` (ignored by some mixers).

        Returns:
            The joint value ``q_tot`` shaped ``[B, 1]``.
        """
        raise NotImplementedError
