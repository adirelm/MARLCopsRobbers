"""Checkpoint split — full training state vs the export-only agent net (T4.6/T4.8).

A FULL checkpoint holds every role net's ``state_dict`` for resuming training. The
EXPORT is the MCP-shippable artifact: the agent net ``state_dict`` ONLY, plus a
shape sidecar JSON (``obs_dim`` / ``hidden`` / ``actions`` / ``view_radius`` /
``obs_channels``). The mixer, opponent heads, optimizer state, and the global
state are NEVER exported — the cop/thief decomposition boundary and the
local-obs-only MCP boundary. The sidecar lets the server rebuild the net shape
without importing any training code.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.marl.nets.agent_net import RecurrentQNet
from src.marl.nets.dims import action_dim, hidden_dim, obs_dim


def shape_sidecar(cfg: dict, role: str) -> dict:
    """Return the export shape sidecar (dims only — no weights, no global state)."""
    return {
        "obs_dim": obs_dim(cfg),
        "hidden": hidden_dim(cfg),
        "actions": action_dim(cfg, role),
        "view_radius": int(cfg["env"]["view_radius_max"]),
        "obs_channels": int(cfg["env"]["obs_channels"]),
    }


def export_agent_weights(net: RecurrentQNet, path: str | Path, cfg: dict, role: str) -> Path:
    """Save ONLY the agent net ``state_dict`` + a shape sidecar (no train-only structure).

    Args:
        net: The trained role agent net (its ``state_dict`` carries no mixer).
        path: Destination ``.pt`` path (a sibling ``<path>.shape.json`` is written).
        cfg: Loaded config (drives the sidecar dims).
        role: ``"cop"`` / ``"thief"`` (sets the sidecar action width).

    Returns:
        The sidecar JSON path written alongside the weights.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), path)
    sidecar = path.with_suffix(path.suffix + ".shape.json")
    sidecar.write_text(json.dumps(shape_sidecar(cfg, role), indent=2), encoding="utf-8")
    return sidecar


def save_full_checkpoint(path: str | Path, nets: dict[str, RecurrentQNet]) -> Path:
    """Save the FULL training state — every role net's ``state_dict`` — for resuming."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({role: net.state_dict() for role, net in nets.items()}, path)
    return path
