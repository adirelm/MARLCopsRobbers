"""AgentController — server-side per-session GRU hidden state for one role (T5.5, eq 8).

The bridge between an MCP tool handler and the trained policy. It holds ONE
:class:`~src.services.policy.RecurrentPolicy` per ``session_id`` (a sub-game): the
policy carries the recurrent hidden state ``z_t`` across the ~25 ``request_move``
ticks and is reset on ``new_sub_game`` (a fresh ``z_0``). Acting is GREEDY (ε=0,
decentralized execution). Policy internals never escape — ``act`` returns only the
chosen action int, so no value/logit/hidden can leak through a tool return. The
controller is built via ``sdk.build_policy`` (the single acting seam, §4).
"""

from __future__ import annotations

from random import Random

import numpy as np

from src.marl.env.types import Observation


class AgentController:
    """Per-session recurrent acting controller for one MCP server's role."""

    def __init__(self, sdk: object, role: str, net: object, n_agents: int = 1) -> None:
        """Bind the role's trained net + the SDK acting seam.

        Args:
            sdk: A ``MarlSDK`` (uses ``build_policy``; no global state crosses here).
            role: ``"cop"`` or ``"thief"``.
            net: The trained role agent net (dense or OLoRA/bundle-loaded).
            n_agents: Agents per session (1 — one hidden stream per agent/session).
        """
        self._sdk = sdk
        self._role = role
        self._net = net
        self._n = int(n_agents)
        self._sessions: dict[str, dict] = {}
        self._rng = Random(0)  # greedy eval; rng is required by act() but unused at ε=0

    def new_session(self, session_id: str) -> None:
        """Start/reset a sub-game session — a fresh policy (z_0) + an empty tick cache."""
        self._sessions[session_id] = {
            "policy": self._sdk.build_policy(self._role, self._net, self._n),
            "cache": {},  # (tick -> action) for retry idempotency
            "last_tick": -1,
        }

    def end_session(self, session_id: str) -> None:
        """Drop a session's hidden state (``end_sub_game``); a no-op if unknown."""
        self._sessions.pop(session_id, None)

    def has_session(self, session_id: str) -> bool:
        """Return whether ``session_id`` has an active hidden-state stream."""
        return session_id in self._sessions

    def act(self, session_id: str, tick: int, image: list, scalars: list, legal_mask: list) -> int:
        """Advance ``z_t`` one tick (idempotently) and return the GREEDY legal action int.

        A retried ``(session_id, tick)`` returns the CACHED action WITHOUT re-running
        the net (so a lost-response retry never double-advances the GRU state); a
        tick that regresses below the last advanced tick is rejected. ``tick`` is the
        agent's OWN step counter — never opponent/global state.

        Args:
            session_id: The active sub-game session (must exist).
            tick: The monotonic per-session step index (idempotency key).
            image: The agent's LOCAL egocentric image ``(C, W, W)`` (nested lists).
            scalars: The agent's aliasing-memory scalars.
            legal_mask: The env legal mask (a_cop-wide; the policy slices per role).

        Returns:
            The chosen action index (NO value/logit/hidden ever returned).

        Raises:
            KeyError: If ``session_id`` was never started via :meth:`new_session`.
            ValueError: If ``tick`` regresses below the last advanced tick.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"unknown session {session_id!r}: call new_sub_game first")
        if tick in session["cache"]:
            return session["cache"][tick]  # idempotent retry — do NOT advance z_t
        if tick < session["last_tick"]:
            raise ValueError(f"tick {tick} regresses below last={session['last_tick']} for {session_id!r}")
        obs: Observation = {
            "image": np.asarray(image, dtype=np.float32),
            "scalars": np.asarray(scalars, dtype=np.float32),
        }
        action = int(session["policy"].act([obs], [legal_mask], 0.0, self._rng)[0])
        session["cache"][tick] = action
        session["last_tick"] = tick
        return action
