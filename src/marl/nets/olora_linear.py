"""OLoRA-wrapped Linear layer + encoder attach helper (T4.3, P4c Stage 1).

``OLoRALinear`` duck-types ``nn.Linear`` (exposes ``in_features`` /
``out_features`` and ``forward(x)``) so it drops into the encoder
``nn.Sequential`` transparently. It registers the re-based base ``W0`` and the
``bias`` as FROZEN buffers (LoRA convention: bias is part of the frozen base)
and the orthonormal adapters ``A, B`` as the ONLY trainable parameters. The
forward is ``F.linear(x, W0 + scale*(B@A), bias)`` ([7] eqs 3-5), function-
preserving at init. ``wrap_encoder`` replaces every encoder ``nn.Linear`` with
an ``OLoRALinear`` and FREEZES the GRU cell (the transferred recurrent backbone),
leaving the trainable set exactly ``{A, B}`` (+ the un-wrapped Q-head, which the
finetune still adapts) — i.e. only ``{A, B, head, mixer}`` train (spec contract).
"""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional

from src.marl.nets.agent_net import RecurrentQNet
from src.marl.nets.olora_init import olora_init


class OLoRALinear(nn.Module):
    """OLoRA adapter over a frozen pretrained ``nn.Linear`` (duck-typed)."""

    def __init__(self, pretrained: nn.Linear, rank: int, scale: float) -> None:
        """Factor ``pretrained`` and register frozen base + trainable adapters.

        Args:
            pretrained: The source ``nn.Linear`` to wrap (its weight/bias are
                copied; the source is never referenced after construction).
            rank: Adapter rank ``r`` (guarded ``<= min(m, n) // 2``).
            scale: OLoRA ``alpha`` scaling on ``B@A``.
        """
        super().__init__()
        self.in_features = pretrained.in_features
        self.out_features = pretrained.out_features
        self.scale = float(scale)
        weight = pretrained.weight.detach().clone()
        w0, b, a = olora_init(weight, rank, scale)
        self.register_buffer("W0", w0)
        if pretrained.bias is not None:
            self.register_buffer("bias", pretrained.bias.detach().clone())
        else:
            self.bias = None
        self.B = nn.Parameter(b.clone())
        self.A = nn.Parameter(a.clone())

    def forward(self, x: Tensor) -> Tensor:
        """Return ``F.linear(x, W0 + scale*(B@A), bias)`` (frozen base + adapters)."""
        weight = self.W0 + self.scale * (self.B @ self.A)
        return functional.linear(x, weight, self.bias)  # type: ignore[arg-type]


def wrap_encoder(net: RecurrentQNet, cfg: dict) -> RecurrentQNet:
    """Attach OLoRA to ``net``: wrap the encoder Linears, freeze the GRU.

    Mutates ``net`` in place. Only the encoder ``nn.Sequential`` Linear children
    are wrapped as :class:`OLoRALinear` (``olora.rank`` / ``olora.scale`` from
    config); the GRU cell is FROZEN (the transferred recurrent backbone), and the
    Q-head is left plain + trainable. The post-attach trainable set is therefore
    exactly ``{A, B}`` (+ head), so the learner trains only ``{A, B, head, mixer}``.

    Args:
        net: A built ``RecurrentQNet`` whose ``encoder`` is an ``nn.Sequential``.
        cfg: Loaded config (reads ``olora.rank``, ``olora.scale`` and
            ``olora.target_layers`` — only ``["encoder"]`` scope is supported).

    Returns:
        The same ``net`` with its encoder Linears wrapped (returned for chaining).

    Raises:
        ValueError: If ``olora.target_layers`` is anything but ``["encoder"]``.
    """
    olora = cfg["olora"]
    target = list(olora.get("target_layers", ["encoder"]))
    if target != ["encoder"]:
        raise ValueError(f"wrap_encoder supports only target_layers=['encoder'], got {target!r}")
    rank = int(olora["rank"])
    scale = float(olora["scale"])
    for idx, module in enumerate(net.encoder):
        if isinstance(module, nn.Linear):
            net.encoder[idx] = OLoRALinear(module, rank, scale)
    for param in net.gru.parameters():
        param.requires_grad_(False)  # freeze the transferred recurrent backbone
    return net
