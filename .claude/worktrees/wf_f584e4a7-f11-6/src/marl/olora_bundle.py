"""OLoRA attach bundle: content-hashed save/load with a base-mismatch guard (T4.5).

The attach artifact that hands a BC-pretrained, OLoRA-wrapped encoder to the P4d
finetune loop. A bundle persists ONLY ``{base_sha, rank, scale, adapters:{name:
(A,B)}, head_state}`` — never the frozen base ``W0``/``bias``. ``load_bundle``
reconstructs the rebase by RE-RUNNING the initial QR of the supplied base net
([7] eqs 3-5: ``B=Q[:, :r]``, ``A=R[:r, :]``, ``W0 = W_pre - scale*(B@A)``) and
then restores the TRAINED adapters + head on top, so the learned delta is never
erased. The bundle is SELF-DESCRIBING — it carries the ``rank``/``scale`` it was
trained with and the loader rebuilds with THOSE (not the live config), so a
config drift between save and load can never silently corrupt the reconstruction.
``base_sha`` is a deterministic content hash (sorted keys, float64 bytes) of the
unwrapped base state_dict; load REJECTS a bundle whose stored ``base_sha`` does
not match the base net passed in (a wrong-base attach is a silent-corruption
hazard).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from src.marl.nets.agent_net import RecurrentQNet
from src.marl.nets.olora_linear import OLoRALinear, wrap_encoder


def base_sha(net: RecurrentQNet) -> str:
    """Return a deterministic sha256 over the unwrapped base net state_dict.

    Hashes each parameter/buffer in SORTED key order, cast to a FIXED dtype
    (float64) so the digest is invariant to load-time dtype quirks. Used to pin
    a bundle to the exact base it was attached on.

    Args:
        net: The unwrapped base :class:`RecurrentQNet` (BC-pretrained).

    Returns:
        A hex sha256 string over the sorted, dtype-fixed base tensors.
    """
    digest = hashlib.sha256()
    for key in sorted(net.state_dict()):
        digest.update(key.encode("utf-8"))
        tensor = net.state_dict()[key].detach().to(torch.float64).cpu().contiguous()
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _adapters(net: RecurrentQNet) -> dict[str, dict[str, torch.Tensor]]:
    """Collect the trained ``{A, B}`` of every wrapped encoder Linear by name."""
    out: dict[str, dict[str, torch.Tensor]] = {}
    for idx, module in enumerate(net.encoder):
        if isinstance(module, OLoRALinear):
            out[str(idx)] = {"A": module.A.detach().clone(), "B": module.B.detach().clone()}
    return out


def _hparams(net: RecurrentQNet) -> tuple[int, float]:
    """Read the ``(rank, scale)`` off the net's first wrapped encoder Linear."""
    for module in net.encoder:
        if isinstance(module, OLoRALinear):
            return int(module.A.shape[0]), float(module.scale)
    raise ValueError("save_bundle requires an OLoRA-wrapped net (no OLoRALinear adapters found)")


def save_bundle(path: str | Path, net: RecurrentQNet, base_sha: str) -> None:
    """Persist the self-describing bundle ``{base_sha, rank, scale, adapters, head}``.

    Args:
        path: Destination file path (a ``torch.save`` payload).
        net: The OLoRA-WRAPPED net whose encoder Linears are
            :class:`OLoRALinear` (trained adapters + head are saved).
        base_sha: The content hash of the base this bundle was attached on.

    Raises:
        ValueError: If ``net`` is not OLoRA-wrapped (no adapters to save — a
            silent delta-free bundle would otherwise result).
    """
    rank, scale = _hparams(net)
    payload = {
        "base_sha": base_sha,
        "rank": rank,
        "scale": scale,
        "adapters": _adapters(net),
        "head_state": {k: v.detach().clone() for k, v in net.head.state_dict().items()},
    }
    torch.save(payload, Path(path))


def _load_adapters(net: RecurrentQNet, adapters: dict[str, dict[str, torch.Tensor]]) -> None:
    """Copy saved ``A, B`` into each wrapped encoder Linear (in place, no grad)."""
    with torch.no_grad():
        for idx, module in enumerate(net.encoder):
            if isinstance(module, OLoRALinear):
                entry = adapters[str(idx)]
                module.A.copy_(entry["A"])
                module.B.copy_(entry["B"])


def load_bundle(path: str | Path, base_net: RecurrentQNet) -> RecurrentQNet:
    """Wrap ``base_net`` and restore a saved bundle's adapters + head onto it.

    Re-runs the INITIAL QR rebase from ``base_net`` (via
    :func:`src.marl.nets.olora_linear.wrap_encoder`) using the bundle's OWN
    ``rank``/``scale`` (NOT the live config, so config drift cannot corrupt the
    reconstruction), NOT from the saved adapters, then overlays the trained
    ``A, B`` + head from the bundle.

    Args:
        path: The bundle file written by :func:`save_bundle`.
        base_net: The unwrapped base net the bundle was attached on (mutated:
            its encoder Linears are replaced with :class:`OLoRALinear`).

    Returns:
        The wrapped ``base_net`` carrying the bundle's trained adapters + head.

    Raises:
        ValueError: If the bundle's ``base_sha`` does not match ``base_net``.
    """
    payload = torch.load(Path(path), weights_only=True)
    if payload["base_sha"] != base_sha(base_net):
        raise ValueError("OLoRA bundle base_sha does not match the supplied base net")
    olora = {"rank": int(payload["rank"]), "scale": float(payload["scale"]), "target_layers": ["encoder"]}
    wrapped = wrap_encoder(base_net, {"olora": olora})
    _load_adapters(wrapped, payload["adapters"])
    wrapped.head.load_state_dict(payload["head_state"])
    return wrapped
