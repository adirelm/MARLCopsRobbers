"""Unit tests for the OLoRA attach bundle save/load (T4.5, P4c Stage 3).

Validates the attach contract: ``base_sha`` is a stable content hash of the
unwrapped base net; a saved bundle round-trips so a reloaded wrapped net's
forward matches the trained source EXACTLY (the rebase is reconstructed from
the base's INITIAL QR, then the TRAINED A/B + head are restored on top); and a
TAMPERED ``base_sha`` is rejected at load (``ValueError``). torch is seeded.
"""

from __future__ import annotations

import json

import pytest
import torch

from src.marl.nets.agent_net import RecurrentQNet
from src.marl.nets.olora_linear import OLoRALinear, wrap_encoder
from src.marl.olora_bundle import base_sha, load_bundle, save_bundle

SEED = 7
FN_TOL = 1e-5


def _base_net(cfg: dict) -> RecurrentQNet:
    """Build a seeded cop base net (``n_agents=2`` per the ladder contract)."""
    torch.manual_seed(SEED)
    return RecurrentQNet(cfg, role="cop", n_agents=2)


def _train_adapters(net: RecurrentQNet) -> None:
    """Perturb every trainable A/B + head so the wrapped net differs from init."""
    with torch.no_grad():
        for module in net.encoder:
            if isinstance(module, OLoRALinear):
                module.A.add_(torch.randn_like(module.A) * 0.3)
                module.B.add_(torch.randn_like(module.B) * 0.3)
        net.head.weight.add_(torch.randn_like(net.head.weight) * 0.3)


def test_base_sha_is_deterministic_and_input_sensitive(cfg: dict) -> None:
    """base_sha is stable for equal nets and changes when a base weight changes."""
    net_a = _base_net(cfg)
    net_b = _base_net(cfg)
    assert base_sha(net_a) == base_sha(net_b)
    with torch.no_grad():
        net_b.head.weight.add_(1.0)
    assert base_sha(net_a) != base_sha(net_b)


def test_bundle_round_trips_forward(cfg: dict, tmp_path) -> None:
    """A saved trained wrapped net reloads to an identical forward output."""
    base = _base_net(cfg)
    sha = base_sha(base)
    wrapped = wrap_encoder(base, cfg)
    _train_adapters(wrapped)
    obs = torch.randn(4, 2, RecurrentQNet(cfg, "cop", 2).encoder[0].in_features - 2)
    h0 = wrapped.initial_hidden(4, 2)
    expected, _ = wrapped(obs, h0)
    path = tmp_path / "bundle.pt"
    save_bundle(path, wrapped, sha)
    reloaded = load_bundle(path, _base_net(cfg))
    got, _ = reloaded(obs, h0)
    assert torch.linalg.norm(got - expected) < FN_TOL


def test_load_rejects_tampered_base_sha(cfg: dict, tmp_path) -> None:
    """A bundle whose base_sha does not match the load-time base is rejected."""
    base = _base_net(cfg)
    wrapped = wrap_encoder(base, cfg)
    path = tmp_path / "bundle.pt"
    save_bundle(path, wrapped, "deadbeef-not-a-real-sha")
    with pytest.raises(ValueError, match="base_sha"):
        load_bundle(path, _base_net(cfg))


def test_load_reconstructs_rebase_from_base_not_saved_w0(cfg: dict, tmp_path) -> None:
    """Load re-derives W0 from the base's INITIAL QR (no W0 in the bundle file)."""
    base = _base_net(cfg)
    sha = base_sha(base)
    wrapped = wrap_encoder(base, cfg)
    _train_adapters(wrapped)
    path = tmp_path / "bundle.pt"
    save_bundle(path, wrapped, sha)
    payload = torch.load(path, weights_only=True)
    assert set(payload.keys()) == {"base_sha", "rank", "scale", "adapters", "head_state"}
    for entry in payload["adapters"].values():
        assert set(entry.keys()) == {"A", "B"}  # no W0/bias persisted


def test_bundle_self_describes_rank_scale_and_loads_with_them(cfg: dict, tmp_path) -> None:
    """The bundle persists its OWN rank/scale; load rebuilds with them, not config.

    Trains at a NON-default rank/scale (2 / 4.0, vs config 4 / 8.0) so a loader
    that re-read the live config would reconstruct the wrong rebase and diverge.
    """
    cfg2 = {**cfg, "olora": {**cfg["olora"], "rank": 2, "scale": 4.0}}
    base = _base_net(cfg2)
    sha = base_sha(base)
    wrapped = wrap_encoder(base, cfg2)
    _train_adapters(wrapped)
    obs = torch.randn(3, 2, RecurrentQNet(cfg2, "cop", 2).encoder[0].in_features - 2)
    h0 = wrapped.initial_hidden(3, 2)
    expected, _ = wrapped(obs, h0)
    path = tmp_path / "b.pt"
    save_bundle(path, wrapped, sha)
    payload = torch.load(path, weights_only=True)
    assert payload["rank"] == 2
    assert payload["scale"] == 4.0
    reloaded = load_bundle(path, _base_net(cfg2))  # no cfg passed -> must use the bundle's own r/scale
    got, _ = reloaded(obs, h0)
    assert torch.linalg.norm(got - expected) < FN_TOL


def test_save_bundle_rejects_unwrapped_net(cfg: dict, tmp_path) -> None:
    """save_bundle on an unwrapped net raises (no silent delta-free bundle)."""
    base = _base_net(cfg)
    with pytest.raises(ValueError, match="wrapped"):
        save_bundle(tmp_path / "b.pt", base, base_sha(base))


def test_bundle_base_sha_is_serializable_string(cfg: dict, tmp_path) -> None:
    """The persisted base_sha is a plain JSON-serializable hex string."""
    base = _base_net(cfg)
    sha = base_sha(base)
    assert isinstance(sha, str)
    json.dumps({"base_sha": sha})  # must not raise
    path = tmp_path / "bundle.pt"
    save_bundle(path, wrap_encoder(base, cfg), sha)
    assert torch.load(path, weights_only=True)["base_sha"] == sha
