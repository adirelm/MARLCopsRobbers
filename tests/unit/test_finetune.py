"""Tests for curriculum transfer + the OLoRA finetune entry (T4.6/T4.5).

Runs TINY 2-stage curricula (shrunk self-play/replay) so the carry-net-forward
loop and the OLoRA attach (encoder wrapped, GRU frozen) are exercised fast.
Distinct from test_curriculum.py, which tests the env-level promotion ladder.
"""

from __future__ import annotations

import copy

import torch

from src.marl.nets.agent_net import RecurrentQNet
from src.marl.nets.olora_linear import OLoRALinear
from src.sdk.sdk import MarlSDK
from src.services.finetune import finetune_curriculum, stage_params

SEED = 7


def _tiny(cfg: dict) -> dict:
    """Shrink self-play/replay so a curriculum stage trains in a fraction of a second."""
    c = copy.deepcopy(cfg)
    c["selfplay"]["episodes_per_round"] = 2
    c["selfplay"]["update_ratio"] = 1
    c["selfplay"]["rounds"] = 1
    c["algo"]["batch_episodes"] = 2
    c["replay"]["buffer_episodes"] = 16
    return c


def _base_nets(cfg: dict):
    """Build fresh cop (N=2) + thief (N=1) base nets."""
    torch.manual_seed(SEED)
    return RecurrentQNet(cfg, "cop", 2), RecurrentQNet(cfg, "thief", 1)


def test_stage_params_resolves_curriculum_ladder(cfg):
    """stage_params reads (h, w, num_cops) from the curriculum config."""
    assert stage_params(cfg, 0) == (2, 2, cfg["env"]["curriculum"]["num_cops_by_stage"][0])


def test_finetune_curriculum_carries_nets_across_stages(cfg):
    """Two stages run in order and the SAME net object is carried + trained forward."""
    c = _tiny(cfg)
    cop, thief = _base_nets(c)
    out = finetune_curriculum(c, SEED, [0, 1], cop, thief, rounds_per_stage=1)
    assert [s["stage"] for s in out["history"]] == [0, 1]
    assert out["cop_net"] is cop  # carried in place across stages
    assert isinstance(out["thief_net"], RecurrentQNet)


def test_sdk_finetune_olora_wraps_encoder_and_freezes_gru(cfg):
    """SDK.finetune(olora=True) wraps the encoder Linears and freezes the GRU."""
    c = _tiny(cfg)
    cop, thief = _base_nets(c)
    out = MarlSDK(c).finetune(SEED, [0], cop, thief, olora=True, rounds_per_stage=1)
    assert any(isinstance(m, OLoRALinear) for m in out["cop_net"].encoder)
    assert all(not p.requires_grad for p in out["cop_net"].gru.parameters())


def test_sdk_finetune_without_olora_stays_dense(cfg):
    """SDK.finetune(olora=False) trains the dense carried nets (no wrapping)."""
    c = _tiny(cfg)
    cop, thief = _base_nets(c)
    out = MarlSDK(c).finetune(SEED, [0], cop, thief, olora=False, rounds_per_stage=1)
    assert not any(isinstance(m, OLoRALinear) for m in out["cop_net"].encoder)
