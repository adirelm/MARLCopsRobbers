"""VDN mixer — Value-Decomposition Networks (T4.0a, P4a Stage 2).

The simplest credit-assignment mixer (BRIEF eq6): ``Q_tot = Σ_i Q_i``. It is a
linear, state-independent decomposition, so the global ``state`` is ignored. A
zeroed (inactive-cop) ``Q_i`` contributes 0 to the sum, which is how the P4b
learner handles the N=2 active mask. PURE net: no parameters, no training.
"""

from __future__ import annotations

from torch import Tensor

from src.marl.mixers.base_mixer import BaseMixer


class VdnMixer(BaseMixer):
    """Sum per-agent Q values into Q_tot (eq6); the global state is ignored."""

    def forward(self, q_agents: Tensor, state: Tensor) -> Tensor:
        """Return ``Q_tot = Σ_i Q_i`` shaped ``[B, 1]``.

        Args:
            q_agents: Per-agent Q values ``[B, n]``.
            state: The global state ``[B, state_dim]`` (unused by VDN).

        Returns:
            The summed joint value ``[B, 1]``.
        """
        return q_agents.sum(dim=1, keepdim=True)
