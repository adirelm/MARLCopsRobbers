"""Alternating best-response self-play: role schedule + opponent pool (T4.6, #7).

The self-play REGIME is alternating best response: one role trains for
``selfplay.window_k`` consecutive rounds against a FROZEN opponent drawn from a
pool, then the roles swap. The :class:`OpponentPool` is seeded by the Manhattan
heuristic (a competent first opponent — random nets vs random nets learn nothing)
and accumulates frozen net snapshots (FIFO over ``selfplay.pool_size``), so the
trainee faces a mix of the heuristic and recent historical selves. ``window_k`` is
ONLY the frozen-opponent window length, never an update ratio.
"""

from __future__ import annotations

from random import Random

from src.marl.nets.agent_net import RecurrentQNet
from src.services.heuristic_policy import HeuristicPolicy
from src.services.policy import RecurrentPolicy


def training_role(round_idx: int, window_k: int) -> str:
    """Return the role that trains in round ``round_idx`` (the other is frozen).

    Alternates in blocks of ``window_k`` rounds: cop for the first block, thief
    for the next, and so on (best-response window, eq-free scheduling).

    Args:
        round_idx: Zero-based self-play round index within a stage.
        window_k: Frozen-opponent window length (``selfplay.window_k``).

    Returns:
        ``"cop"`` or ``"thief"`` — the role whose learner updates this round.
    """
    return "cop" if (round_idx // max(1, int(window_k))) % 2 == 0 else "thief"


class OpponentPool:
    """Frozen opponents for one role: a heuristic seed + FIFO net snapshots."""

    def __init__(self, role: str, cfg: dict, n_agents: int, seed: int) -> None:
        """Seed the pool with the Manhattan heuristic for ``role``.

        Args:
            role: The opponent's role (``"cop"`` / ``"thief"``).
            cfg: Loaded config (reads ``selfplay.pool_size``).
            n_agents: The opponent role's agent-axis width.
            seed: Seed for the snapshot sampler RNG.
        """
        self._role = role
        self._n = int(n_agents)
        self._size = int(cfg["selfplay"]["pool_size"])
        self._rng = Random(seed)
        self._snaps: list = [HeuristicPolicy(role, cfg, n_agents)]

    def add(self, net: RecurrentQNet) -> None:
        """Snapshot a FROZEN copy of ``net`` as a future opponent (FIFO-capped)."""
        self._snaps.append(RecurrentPolicy(net.clone_target(), self._n))
        if len(self._snaps) > self._size + 1:  # keep the heuristic seed at index 0
            self._snaps.pop(1)

    def sample(self) -> object:
        """Return a reset frozen opponent policy (heuristic or a net snapshot)."""
        opponent = self._rng.choice(self._snaps)
        opponent.reset()
        return opponent

    def __len__(self) -> int:
        """Return the number of opponents held (heuristic + net snapshots)."""
        return len(self._snaps)
