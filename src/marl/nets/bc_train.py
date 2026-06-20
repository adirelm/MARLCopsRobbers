"""Behavior-cloning supervised pretraining of the recurrent agent net (T4.4).

BC is the "pre-trained model" OLoRA later adapts (ADR-D4-1). It is a SINGLE-STEP
supervised problem: feed the role's flattened local obs through
:meth:`~src.marl.nets.agent_net.RecurrentQNet.forward` with a zero hidden state,
read the Q-head outputs AS LOGITS, and minimize cross-entropy against the greedy
expert label. CopNet is built ``n_agents=2`` so the encoder input width (which
includes the agent-id one-hot) is FIXED across the curriculum ladder; ThiefNet
is ``n_agents=1``. Validation accuracy is measured on the held-out EPISODE-level
split (:func:`src.marl.data.bc_split.episode_split`) — never on adjacent records.

The CI smoke runs this on a tiny vendored set (overfit: loss down + beats the
random floor). The full per-grid ``bc.val_acc_gate_by_grid`` gate (read by
:func:`gate_for`) is checked by the slow full-BC script, not the unit suite (a
tiny set under-trains). The privileged-expert vs local-obs ceilings that justify
those per-grid gates are derived in docs/ANALYSIS.md §0.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from src.marl.data.bc_split import episode_split
from src.marl.nets.agent_net import RecurrentQNet

# CopNet fixes the encoder input width across the ladder via a 2-wide agent-id
# one-hot (ADR-D4); ThiefNet is single-agent. (Not a tunable — a net-shape contract.)
_N_AGENTS_BY_ROLE = {"cop": 2, "thief": 1}

Dataset = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def gate_for(cfg: dict, grid: tuple[int, int]) -> float:
    """Return the per-grid BC val-acc gate ``bc.val_acc_gate_by_grid[min(h, w)]``.

    Args:
        cfg: The loaded config (reads ``bc.val_acc_gate_by_grid``).
        grid: The ``(h, w)`` board size; the gate is keyed on ``min(h, w)``.

    Returns:
        The float val-acc threshold required to attach OLoRA at this grid.
    """
    return float(cfg["bc"]["val_acc_gate_by_grid"][min(grid)])


def _flatten(obs: np.ndarray, scalars: np.ndarray) -> Tensor:
    """Flatten ``(N, C, W, W)`` image + ``(N, scalars)`` into ``[N, 1, obs_dim]``.

    Mirrors :func:`src.marl.learner._learner_helpers.flatten_obs` (image flat ++
    scalars), with a singleton agent axis so the net runs one agent per record.

    Args:
        obs: The egocentric image batch ``(N, C, W, W)`` float32.
        scalars: The aliasing-memory scalars ``(N, obs_scalars)`` float32.

    Returns:
        A ``[N, 1, obs_dim]`` float32 tensor matching the net's encoder input.
    """
    n = obs.shape[0]
    flat_img = torch.from_numpy(obs.reshape(n, -1)).float()
    flat_sc = torch.from_numpy(scalars).float()
    return torch.cat([flat_img, flat_sc], dim=-1).unsqueeze(1)


def _val_accuracy(net: RecurrentQNet, obs: Tensor, labels: Tensor) -> float:
    """Return the greedy argmax accuracy of ``net`` on a held-out obs/label split."""
    if obs.shape[0] == 0:
        return 0.0
    net.eval()
    with torch.no_grad():
        h0 = net.initial_hidden(obs.shape[0], 1)
        logits, _h = net(obs, h0)
        preds = logits[:, 0, :].argmax(dim=-1)
    return float((preds == labels).float().mean().item())


def bc_fit(
    cfg: dict, grid: tuple[int, int], role: str, dataset: Dataset
) -> tuple[RecurrentQNet, list[float], float]:
    """Train a role net by BC, returning the net, per-epoch losses, and val-acc.

    Args:
        cfg: The loaded config (reads ``bc.epochs`` / ``bc.lr`` + net dims).
        grid: The ``(h, w)`` the dataset was generated on (provenance only here).
        role: ``"cop"`` (built ``n_agents=2``) or ``"thief"`` (``n_agents=1``).
        dataset: An ``(obs, scalars, actions, episode_ids)`` array tuple.

    Returns:
        A ``(net, losses, val_acc)`` triple: the trained
        :class:`RecurrentQNet`, the per-epoch CE losses, and the held-out
        episode-level validation accuracy.
    """
    obs_arr, scalars_arr, actions_arr, episode_ids = dataset
    train_idx, val_idx = episode_split(episode_ids, cfg)
    x = _flatten(obs_arr, scalars_arr)
    y = torch.from_numpy(actions_arr).long()
    net = RecurrentQNet(cfg, role, _N_AGENTS_BY_ROLE[role])
    opt = torch.optim.AdamW(net.parameters(), lr=float(cfg["bc"]["lr"]))
    loss_fn = nn.CrossEntropyLoss()
    x_tr, y_tr = x[train_idx], y[train_idx]
    losses: list[float] = []
    for _epoch in range(int(cfg["bc"]["epochs"])):
        net.train()
        opt.zero_grad()
        h0 = net.initial_hidden(x_tr.shape[0], 1)
        logits, _h = net(x_tr, h0)
        loss = loss_fn(logits[:, 0, :], y_tr)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    val_acc = _val_accuracy(net, x[val_idx], y[val_idx])
    return net, losses, val_acc


def bc_train(cfg: dict, grid: tuple[int, int], role: str, dataset: Dataset) -> tuple[RecurrentQNet, float]:
    """Train a role net by BC and return ``(net, val_acc)`` (the spec contract).

    Thin wrapper over :func:`bc_fit` dropping the loss history.

    Args:
        cfg: The loaded config.
        grid: The ``(h, w)`` the dataset was generated on.
        role: ``"cop"`` or ``"thief"``.
        dataset: An ``(obs, scalars, actions, episode_ids)`` array tuple.

    Returns:
        A ``(net, val_acc)`` pair — the trained net and its held-out val accuracy.
    """
    net, _losses, val_acc = bc_fit(cfg, grid, role, dataset)
    return net, val_acc
