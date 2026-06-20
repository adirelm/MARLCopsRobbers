"""Pure-tensor helpers for the centralized BPTT learners (T4.2, P4b Stage 2).

Four side-effect-free functions shared by every cop/thief learner update:

* :func:`masked_huber` — filled-masked-mean Huber loss (no grad through mask).
* :func:`bptt_unroll` — roll the recurrent Q-net over ``T+1`` steps, carrying
  the GRU hidden state, to produce ``q[B, T+1, N, A]``.
* :func:`gather_chosen` — gather the chosen-action Q-value per ``[B, T, N]``.
* :func:`masked_argmax` — argmax restricted to legal actions (additive ``-inf``
  on illegal entries) — used for Double-DQN online action selection.

These are deliberately net-agnostic and config-free: every dimension flows in
through the tensor shapes, so there are no literals to hoist into config.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor
from torch.nn import functional


def frozen_copy(module: object) -> object:
    """Return a grad-free deepcopy of an ``nn.Module`` (a frozen target).

    Args:
        module: Any ``nn.Module`` (or ``None``) to snapshot.

    Returns:
        A ``requires_grad_(False)`` deepcopy, or ``None`` when ``module`` is
        ``None`` (the parameter-free / no-mixer branch).
    """
    if module is None:
        return None
    return copy.deepcopy(module).requires_grad_(False)


def masked_huber(td_error: Tensor, mask: Tensor, delta: float) -> Tensor:
    """Return the filled-masked-mean Huber loss of a TD error.

    Computes ``huber(td_error, delta)`` elementwise, zeroes the masked
    positions, sums, then divides by ``mask.sum()`` so the result is the mean
    over *unmasked* (filled) entries. The mask is detached to a stop-gradient
    multiplier: no gradient flows through it back to the caller.

    Args:
        td_error: The temporal-difference error tensor (any shape).
        mask: A 0/1 (or bool) tensor broadcastable to ``td_error``; ``1`` keeps
            an entry, ``0`` drops it from both the loss and its gradient.
        delta: The Huber transition point (``algo.huber_delta`` from config).

    Returns:
        A scalar tensor: the mean Huber loss over the unmasked entries.
    """
    keep = mask.detach().to(td_error.dtype)
    per_elem = functional.huber_loss(td_error, torch.zeros_like(td_error), delta=delta, reduction="none")
    return (per_elem * keep).sum() / keep.sum().clamp(min=1.0)


def bptt_unroll(net: object, obs: Tensor, h0: Tensor) -> Tensor:
    """Roll the recurrent Q-net over ``T+1`` steps carrying the hidden state.

    Args:
        net: A ``RecurrentQNet`` (``forward(obs[B,N,obs_dim], h) -> (q, h')``).
        obs: Per-step local observations ``[B, T+1, N, obs_dim]``.
        h0: The initial hidden state ``[B, N, H]``.

    Returns:
        The stacked Q-values ``[B, T+1, N, A]`` with the GRU hidden state
        carried forward across all ``T+1`` timesteps.
    """
    steps = obs.shape[1]
    hidden = h0
    out: list[Tensor] = []
    for step in range(steps):
        q_step, hidden = net.forward(obs[:, step], hidden)
        out.append(q_step)
    return torch.stack(out, dim=1)


def gather_chosen(q: Tensor, actions: Tensor) -> Tensor:
    """Gather the Q-value of the chosen action per agent and timestep.

    Args:
        q: Q-values ``[B, T, N, A]``.
        actions: Chosen action indices ``[B, T, N]`` (integer).

    Returns:
        The chosen-action Q-values ``[B, T, N]``.
    """
    return q.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)


def masked_argmax(q: Tensor, legal: Tensor) -> Tensor:
    """Return the argmax over actions restricted to the legal ones.

    Adds ``-inf`` to illegal action positions (on a copy, leaving ``q``
    untouched) before taking the argmax over the last axis, guaranteeing the
    returned index is always legal.

    Args:
        q: Q-values ``[..., A]``.
        legal: A boolean legality mask ``[..., A]`` (``True`` == legal).

    Returns:
        The legal-argmax indices, shaped like ``q`` without its last axis.
    """
    neg_inf = torch.finfo(q.dtype).min
    masked = q.masked_fill(~legal.bool(), neg_inf)
    return masked.argmax(dim=-1)


def flatten_obs(image: Tensor, scalars: Tensor) -> Tensor:
    """Flatten the egocentric image + scalars into the net's ``obs_dim`` vector.

    Args:
        image: Egocentric image ``[B, T+1, N, C, W, W]``.
        scalars: Aliasing-memory scalars ``[B, T+1, N, obs_scalars]``.

    Returns:
        The per-step flattened local obs ``[B, T+1, N, obs_dim]`` matching the
        :func:`src.marl.nets.dims.obs_dim` contract (image flat ++ scalars).
    """
    b, t, n = image.shape[:3]
    flat_img = image.reshape(b, t, n, -1)
    return torch.cat([flat_img, scalars], dim=-1)


def net_param_group(net: object, lr: float) -> dict:
    """Return the AdamW group of a net's TRAINABLE params at learning rate ``lr``.

    Filters on ``requires_grad`` so a frozen-base OLoRA-wrapped net contributes
    only its ``{A, B, head}`` adapters (the frozen ``W0``/``bias`` are excluded);
    a plain dense net contributes every parameter (the unchanged P4b path).

    Args:
        net: A ``RecurrentQNet`` (plain or OLoRA-wrapped).
        lr: The agent-net learning rate (``algo.lr_agent`` from config).

    Returns:
        A single AdamW param-group dict ``{"params": [...], "lr": lr}``.
    """
    return {"params": [p for p in net.parameters() if p.requires_grad], "lr": lr}


def to_tensors(batch: dict, device: torch.device) -> dict:
    """Convert a (possibly numpy) replay batch into a dict of tensors on ``device``.

    Float fields keep ``float32``; ``actions`` become ``int64``; the boolean
    masks (``done``/``filled``/``next_legal_mask``/``active``) become ``bool``.
    Non-array provenance keys (e.g. ``source_tag``) are dropped.

    Args:
        batch: The sampled replay batch (numpy arrays or tensors).
        device: Target torch device for every produced tensor.

    Returns:
        A new dict of the converted tensors.
    """
    long_keys = {"actions"}
    bool_keys = {"done", "filled", "next_legal_mask", "active"}
    float_keys = ("obs", "scalars", "global_state", "reward")
    out: dict[str, Tensor] = {}
    for key in (*float_keys, *long_keys, *bool_keys):
        if key not in batch:  # global_state is optional (IQL has no centralized state)
            continue
        dtype = torch.int64 if key in long_keys else torch.bool if key in bool_keys else torch.float32
        out[key] = torch.as_tensor(batch[key], dtype=dtype, device=device)
    return out
