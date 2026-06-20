"""Tests for the checkpoint export/full split + shape sidecar (T4.6/T4.8).

Verifies the export carries ONLY the agent net (no mixer/global state), the
sidecar dims are config-derived, and both checkpoints round-trip. torch seeded.
"""

from __future__ import annotations

import json

import torch

from src.marl.nets.agent_net import RecurrentQNet
from src.marl.nets.dims import obs_dim
from src.services.checkpoints import export_agent_weights, save_full_checkpoint, shape_sidecar

SEED = 7


def test_export_writes_weights_and_sidecar(cfg, tmp_path):
    """export_agent_weights writes the .pt + a sidecar with config-derived dims."""
    torch.manual_seed(SEED)
    net = RecurrentQNet(cfg, "cop", 2)
    path = tmp_path / "cop.pt"
    sidecar = export_agent_weights(net, path, cfg, "cop")
    assert path.exists() and sidecar.exists()
    meta = json.loads(sidecar.read_text())
    assert meta["obs_dim"] == obs_dim(cfg)
    assert meta["actions"] == cfg["env"]["actions"]["a_cop"]


def test_export_is_agent_net_only_and_round_trips(cfg, tmp_path):
    """The exported state_dict is exactly the agent net (no mixer) and reloads."""
    torch.manual_seed(SEED)
    net = RecurrentQNet(cfg, "thief", 1)
    path = tmp_path / "thief.pt"
    export_agent_weights(net, path, cfg, "thief")
    state = torch.load(path, weights_only=True)
    assert set(state) == set(net.state_dict())
    assert not any("mixer" in key for key in state)
    RecurrentQNet(cfg, "thief", 1).load_state_dict(state)  # must not raise


def test_shape_sidecar_thief_action_width(cfg):
    """The thief sidecar reports the 4-action thief head (not a_cop)."""
    assert shape_sidecar(cfg, "thief")["actions"] == cfg["env"]["actions"]["a_thief"]


def test_save_full_checkpoint_round_trips_both_roles(cfg, tmp_path):
    """A full checkpoint holds both role nets' state_dicts."""
    torch.manual_seed(SEED)
    nets = {"cop": RecurrentQNet(cfg, "cop", 2), "thief": RecurrentQNet(cfg, "thief", 1)}
    path = save_full_checkpoint(tmp_path / "full.pt", nets)
    loaded = torch.load(path, weights_only=True)
    assert set(loaded) == {"cop", "thief"}
