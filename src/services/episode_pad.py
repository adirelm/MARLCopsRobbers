"""Pad per-role step records into the CentralizedReplayBuffer episode schema (T4.6).

Generalizes the P3 throwaway ``_smoke_helpers._pad_episode`` to N agent slots
with an episode-constant ``active`` occupancy mask: a role's real agents fill the
leading slots and the remaining slots are zero-filled phantoms (``active=False``),
so a 1-cop stage still matches the cop buffer's fixed ``N=2`` width. ``obs`` /
``scalars`` / ``global_state`` live on the ``T+1`` axis (index ``T`` holds the
terminal next-frame for the recurrent target unroll); per-step fields are ``T``.
Global state ``s`` is encoded train-time only here and never leaves the buffer.
"""

from __future__ import annotations

import numpy as np

from src.marl.data.obs_encoder import encode_obs_batch, encode_state


def pad_episode(steps: list[dict], n_slots: int, n_actions: int, cfg: dict) -> dict:
    """Build one padded buffer episode (agent axis ``n_slots``) from step records.

    Args:
        steps: Per-step records (one role) each with ``obs``/``nxt`` (per-agent
            Observation lists), ``state``/``nxt_state`` (GlobalState), ``acts``/
            ``rews`` (per-agent), ``nmask`` (per-agent next legal masks), ``done``.
        n_slots: The buffer's fixed agent-axis width for this role (cop 2, thief 1).
        n_actions: Action-space width for the legality mask (``a_cop`` / ``a_thief``).
        cfg: Loaded config (reads ``game.max_moves`` + ``env.*`` obs dims).

    Returns:
        A buffer-schema episode dict (``add_episode`` contract): real steps marked
        in ``filled``, leading ``n_real`` slots marked in ``active``.
    """
    t = len(steps)
    t_max = int(cfg["game"]["max_moves"])
    n_real = len(steps[0]["obs"])
    c = int(cfg["env"]["obs_channels"])
    w = 2 * int(cfg["env"]["view_radius_max"]) + 1
    ns = int(cfg["env"]["obs_scalars"])
    state_dim = encode_state(steps[0]["state"], cfg).shape[0]
    ep = {
        "obs": np.zeros((t_max + 1, n_slots, c, w, w), np.float32),
        "scalars": np.zeros((t_max + 1, n_slots, ns), np.float32),
        "global_state": np.zeros((t_max + 1, state_dim), np.float32),
        "actions": np.zeros((t_max, n_slots), np.int64),
        "reward": np.zeros((t_max, n_slots), np.float32),
        "done": np.zeros((t_max,), bool),
        "filled": np.zeros((t_max,), bool),
        "next_legal_mask": np.zeros((t_max, n_slots, n_actions), bool),
        "active": np.zeros(n_slots, bool),
        "hidden_seed": np.int64(0),
    }
    ep["active"][:n_real] = True
    for i, step in enumerate(steps):
        imgs, scal = encode_obs_batch(step["obs"])
        ep["obs"][i, :n_real], ep["scalars"][i, :n_real] = imgs, scal
        ep["global_state"][i] = encode_state(step["state"], cfg)
        ep["actions"][i, :n_real], ep["reward"][i, :n_real] = step["acts"], step["rews"]
        ep["done"][i], ep["filled"][i] = step["done"], True
        for j in range(n_real):
            ep["next_legal_mask"][i, j] = np.asarray(step["nmask"][j], bool)
    last = steps[-1]
    imgs, scal = encode_obs_batch(last["nxt"])
    ep["obs"][t, :n_real], ep["scalars"][t, :n_real] = imgs, scal
    ep["global_state"][t] = encode_state(last["nxt_state"], cfg)
    return ep
