"""Centralized episodic replay buffer for CTDE / BPTT (T3.1).

Stores WHOLE padded episodes in a pre-allocated numpy ring so the recurrent
agent net can unroll with correct hidden-state continuity (BRIEF eq 9). The
global state ``s`` lives ONLY here (train-time; never crosses the MCP boundary).
Each stored episode carries a :class:`~src.marl.data.schemas.SourceTag`
provenance label so a sampled batch can mix EXPERT / SELF_PLAY / RANDOM /
LIVE_CTDE data. ``filled`` masks padded post-terminal steps so BPTT ignores pad.

N convention (two SEPARATE buffers): the cop buffer uses ``n_agents = 2`` (1-cop
stages pad the agent axis with ``filled``/zeros); the thief buffer uses
``n_agents = 1``.
"""

from __future__ import annotations

import numpy as np

from src.marl.data.schemas import SourceTag


class CentralizedReplayBuffer:
    """Pre-allocated numpy ring of padded whole episodes (one per role)."""

    def __init__(  # noqa: PLR0913 — one kwarg per fixed tensor dim is intentional
        self,
        capacity: int,
        t_max: int,
        n_agents: int,
        obs_channels: int,
        w_v: int,
        obs_scalars: int,
        state_dim: int,
        n_actions: int,
        seed: int,
    ) -> None:
        """Allocate the ring storage and seed the sampling RNG.

        Args:
            capacity: Maximum number of whole episodes held (``replay.buffer_episodes``).
            t_max: Padded per-episode horizon (``game.max_moves``).
            n_agents: Agent-axis width (cop buffer 2, thief buffer 1).
            obs_channels: Egocentric image channel count ``C``.
            w_v: Egocentric window width ``2 * view_radius_max + 1``.
            obs_scalars: Aliasing-memory scalar count per agent.
            state_dim: Stage-invariant encoded global-state width (77).
            n_actions: Action-space size for the legality mask (``a_cop``).
            seed: Seed for the ``np.random.default_rng`` sampler.
        """
        self._capacity = int(capacity)
        self._t_max = int(t_max)
        self._n_agents = int(n_agents)
        self._n_actions = int(n_actions)
        self._rng = np.random.default_rng(seed)
        self._size = 0
        self._cursor = 0
        n, t = self._capacity, self._t_max
        a = self._n_agents
        # obs/scalars live on the T+1 axis (matching global_state) so the stored
        # terminal next-obs survives for P4's recurrent target unroll over obs[1..T].
        self._obs = np.zeros((n, t + 1, a, obs_channels, w_v, w_v), dtype=np.float32)
        self._scalars = np.zeros((n, t + 1, a, obs_scalars), dtype=np.float32)
        self._global_state = np.zeros((n, t + 1, state_dim), dtype=np.float32)
        self._actions = np.zeros((n, t, a), dtype=np.int64)
        self._reward = np.zeros((n, t, a), dtype=np.float32)
        self._done = np.zeros((n, t), dtype=bool)
        self._filled = np.zeros((n, t), dtype=bool)
        self._next_legal_mask = np.zeros((n, t, a, n_actions), dtype=bool)
        # Episode-constant per-slot occupancy mask (NOT time-varying): which of the
        # N agent slots hold a real agent (vs a zero-filled phantom from widening a
        # k<N episode to the buffer's full N width). Distinct from per-step `filled`.
        self._active = np.zeros((n, a), dtype=bool)
        self._hidden_seed = np.zeros((n,), dtype=np.int64)
        self._source_tag: list[SourceTag] = [SourceTag.RANDOM] * n

    def __len__(self) -> int:
        """Return the number of episodes currently stored."""
        return self._size

    def add_episode(self, episode: dict, source_tag: SourceTag) -> None:
        """Write one padded whole episode into the ring at the write cursor.

        Per-step fields shorter than ``t_max`` are zero-padded; ``filled`` marks
        the real steps so padded post-terminal steps are ignored downstream.

        Args:
            episode: Mapping with keys ``obs (T+1,N,C,w,w)`` (the +1 frame is the
                terminal next-obs), ``scalars (T+1,N,scalars)``, ``global_state
                (T+1,state_dim)``, ``actions (T,N)``, ``reward (T,N)``, ``done
                (T,)``, ``filled (T,)``, ``next_legal_mask (T,N,n_actions)``,
                ``active (N,)`` (episode-constant per-slot occupancy mask),
                ``hidden_seed`` scalar.
            source_tag: Provenance label stored alongside this episode.

        Raises:
            ValueError: If any per-agent field's agent axis != ``n_agents`` or
                ``active`` is mis-shaped (see :meth:`_validate_agent_axes`); the
                producer widens a k-cop episode to N (no silent broadcast).
        """
        self._validate_agent_axes(episode)
        active = np.asarray(episode["active"], dtype=bool)
        i = self._cursor
        self._zero_slot(i)
        t = min(int(episode["filled"].shape[0]), self._t_max)
        a = self._n_agents
        self._obs[i, : t + 1] = episode["obs"][: t + 1, :a]
        self._scalars[i, : t + 1] = episode["scalars"][: t + 1, :a]
        self._global_state[i, : t + 1] = episode["global_state"][: t + 1]
        self._actions[i, :t] = episode["actions"][:t, :a]
        self._reward[i, :t] = episode["reward"][:t, :a]
        self._done[i, :t] = episode["done"][:t]
        self._filled[i, :t] = episode["filled"][:t]
        self._next_legal_mask[i, :t] = episode["next_legal_mask"][:t, :a]
        self._active[i] = active
        self._hidden_seed[i] = np.int64(episode["hidden_seed"])
        self._source_tag[i] = source_tag
        self._cursor = (i + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def _validate_agent_axes(self, episode: dict) -> None:
        """Fail loud if any per-agent field's agent axis != n_agents (no broadcast)."""
        n = self._n_agents
        for key in ("obs", "scalars", "actions", "reward", "next_legal_mask"):
            if np.asarray(episode[key]).shape[1] != n:
                raise ValueError(f"episode[{key!r}] agent axis != n_agents ({n})")
        if np.asarray(episode["active"]).shape != (n,):
            raise ValueError(f"episode['active'] must be shape ({n},)")

    def _zero_slot(self, i: int) -> None:
        """Reset every per-step field at ring index ``i`` (clears stale pad)."""
        self._obs[i] = 0.0
        self._scalars[i] = 0.0
        self._global_state[i] = 0.0
        self._actions[i] = 0
        self._reward[i] = 0.0
        self._done[i] = False
        self._filled[i] = False
        self._next_legal_mask[i] = False
        self._active[i] = False

    def sample(self, batch_size: int) -> dict:
        """Sample ``batch_size`` episodes with replacement into a batched dict.

        Args:
            batch_size: Number of episodes to draw.

        Returns:
            A dict of batched arrays plus a ``source_tag`` list of length ``B``
            carrying provenance. ``obs``/``scalars``/``global_state`` are
            ``(B, T+1, ...)`` (the +1 holds the terminal next-obs); per-step
            fields are ``(B, T, ...)``; ``active`` is ``(B, n_agents)``.

        Raises:
            ValueError: If the buffer is empty.
        """
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        idx = self._rng.integers(0, self._size, size=int(batch_size))
        return {
            "obs": self._obs[idx],
            "scalars": self._scalars[idx],
            "global_state": self._global_state[idx],
            "actions": self._actions[idx],
            "reward": self._reward[idx],
            "done": self._done[idx],
            "filled": self._filled[idx],
            "next_legal_mask": self._next_legal_mask[idx],
            "active": self._active[idx],
            "hidden_seed": self._hidden_seed[idx],
            "source_tag": [self._source_tag[int(j)] for j in idx],
        }
