"""Tests for the seeded ablation sweep + SDK policy/export round-trip (T4.7/T4.8).

The sweep runs tiny qmix/vdn stages through the SDK and appends one JSONL record
per run; the export -> load -> build_policy round-trip reconstructs an acting
policy from saved agent weights. torch seeded.
"""

from __future__ import annotations

import copy
import json

import torch

from src.marl.nets.agent_net import RecurrentQNet
from src.sdk.sdk import MarlSDK
from src.services.sweep import run_sweep

SEED = 7


def _tiny(cfg: dict) -> dict:
    """Shrink self-play/replay so a swept stage trains in a fraction of a second."""
    c = copy.deepcopy(cfg)
    c["selfplay"]["episodes_per_round"] = 2
    c["selfplay"]["update_ratio"] = 1
    c["selfplay"]["rounds"] = 1
    c["algo"]["batch_episodes"] = 2
    c["replay"]["buffer_episodes"] = 16
    return c


def test_run_sweep_appends_one_jsonl_record_per_run(cfg, tmp_path):
    """Sweeping 2 algorithms x 1 seed writes 2 reproducible JSONL records."""
    c = _tiny(cfg)
    out = tmp_path / "runs" / "ablation.jsonl"
    records = run_sweep(MarlSDK(c), c, ["qmix", "vdn"], [SEED], stage_idx=0, out_path=out)
    assert len(records) == 2
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["algorithm"] == "qmix"
    assert first["grid"] == [2, 2]
    assert "final_capture_rate" in first


def test_export_load_build_policy_round_trip(cfg, tmp_path):
    """export_weights -> load_weights reconstructs the net; build_policy makes it act."""
    torch.manual_seed(SEED)
    sdk = MarlSDK(cfg)
    net = RecurrentQNet(cfg, "cop", 2)
    sidecar = sdk.export_weights(net, "cop", tmp_path / "cop.pt")
    assert sidecar.exists()
    loaded = sdk.load_weights("cop", tmp_path / "cop.pt", n_agents=2)
    originals = net.state_dict()
    assert all(torch.equal(originals[k], v) for k, v in loaded.state_dict().items())
    policy = sdk.build_policy("cop", loaded, n_agents=1)
    assert hasattr(policy, "act") and hasattr(policy, "reset")
