"""Recurrent acting policy — GRU hidden carry + ε-greedy legal action (T4.6 / eq 8).

Wraps a trained :class:`~src.marl.nets.agent_net.RecurrentQNet` for ACTING: it
carries the per-agent GRU hidden state ``z_t`` across a sub-game's ticks (eq 8),
encodes each agent's LOCAL :class:`~src.marl.env.types.Observation` into the net's
obs vector (image-flat ++ scalars, matching the learner's ``flatten_obs``), and
returns one legal ε-greedy action per agent (illegal actions masked to ``-inf``
before the argmax). This is the SINGLE acting seam reused by the self-play
rollout, ``SDK.build_policy``, and the MCP ``AgentController`` — none of which
ever sees the global state. Acting runs under ``torch.no_grad``.
"""

from __future__ import annotations

from random import Random

import numpy as np
import torch

from src.marl.data.obs_encoder import encode_obs_batch
from src.marl.env.actions import Action
from src.marl.env.types import Observation
from src.marl.learner._learner_helpers import masked_argmax
from src.marl.nets.agent_net import RecurrentQNet


class RecurrentPolicy:
    """Carry per-agent GRU hidden state and pick legal ε-greedy actions for one role."""

    def __init__(self, net: RecurrentQNet, n_agents: int) -> None:
        """Bind an acting net and zero the per-agent hidden state.

        Args:
            net: A trained ``RecurrentQNet`` for this role (acted on under
                ``no_grad``; its weights are never mutated here).
            n_agents: Number of same-role agents this policy drives in lockstep
                (e.g. 1 or 2 cops); may be ``<=`` the net's agent-id width.
        """
        self._net = net
        self._n = int(n_agents)
        self.reset()

    def reset(self) -> None:
        """Zero the hidden state ``z_0`` (call on episode / ``new_sub_game`` reset)."""
        self._hidden = self._net.initial_hidden(1, self._n)

    def _encode(self, obs_list: list[Observation]) -> torch.Tensor:
        """Encode local observations into the net's ``[1, n, obs_dim]`` input."""
        images, scalars = encode_obs_batch(obs_list)
        flat = images.reshape(images.shape[0], -1)
        vec = np.concatenate([flat, scalars], axis=-1)
        return torch.from_numpy(vec).float().unsqueeze(0)

    def act(
        self,
        obs_list: list[Observation],
        legal_masks: list,
        epsilon: float,
        rng: Random,
        state: object = None,
    ) -> list[Action]:
        """Advance the hidden state one tick and return one legal action per agent.

        Args:
            obs_list: One LOCAL Observation per agent (length ``n_agents``).
            legal_masks: One env legality mask per agent (the uniform ``a_cop``
                width); each is sliced to this net's head width so a thief never
                considers the cop-only barrier slot (mirrors ``ThiefLearner``).
            epsilon: Exploration probability; with prob ε an agent picks a uniform
                LEGAL action, else the legal-masked greedy argmax.
            rng: The exploration RNG (seeded for reproducibility).
            state: Accepted for a uniform policy interface (the heuristic opponent
                needs it) but IGNORED here — a net policy acts on local obs only.

        Returns:
            One :class:`Action` per agent (always legal for this role).
        """
        a_dim = self._net.head.out_features
        masks = [list(mask)[:a_dim] for mask in legal_masks]
        obs = self._encode(obs_list)
        with torch.no_grad():
            q, self._hidden = self._net(obs, self._hidden)
        legal = torch.as_tensor(np.asarray(masks, dtype=bool)).unsqueeze(0)
        greedy = masked_argmax(q, legal)[0]
        actions: list[Action] = []
        for i, mask in enumerate(masks):
            if rng.random() < epsilon:
                actions.append(Action(rng.choice([j for j, ok in enumerate(mask) if ok])))
            else:
                actions.append(Action(int(greedy[i])))
        return actions
