"""OLoRA thin-QR initialization (T4.3, P4c Stage 1).

Orthonormal Low-Rank Adaptation of a PRETRAINED weight via QR decomposition
(cite [7] Buyukakyuz 2024, eqs 3-5): factor the pretrained ``W_pre`` with a
reduced QR, keep the leading ``rank`` columns/rows as the adapters ``B, A``,
and re-base ``W0 = W_pre - scale*(B@A)`` so the adapted weight
``W0 + scale*(B@A)`` reproduces ``W_pre`` exactly at init (function-preserving).
Unlike random-init LoRA, ``B`` carries the pretrained weight's orthonormal
column basis (the "O" in OLoRA). All values are passed in — no config/literals.
"""

from __future__ import annotations

import torch
from torch import Tensor


def olora_init(weight: Tensor, rank: int, scale: float) -> tuple[Tensor, Tensor, Tensor]:
    """Factor a pretrained weight into a re-based base ``W0`` plus adapters ``B, A``.

    Computes a reduced QR of the pretrained ``weight`` ([7] eq 3), slices the
    leading ``rank`` orthonormal columns into ``B`` and rows into ``A`` (eq 4),
    and re-bases ``W0 = weight - scale*(B@A)`` (eq 5) so ``W0 + scale*(B@A)``
    equals ``weight`` at init.

    Args:
        weight: The pretrained ``nn.Linear`` weight ``[m, n]`` (out, in).
        rank: Adapter rank ``r``; must satisfy ``1 <= rank <= min(m, n) // 2``.
        scale: The OLoRA ``alpha`` scaling applied to ``B@A``.

    Returns:
        A ``(W0, B, A)`` tuple: re-based base ``[m, n]``, orthonormal-column
        ``B`` ``[m, r]``, and ``A`` ``[r, n]``.

    Raises:
        ValueError: If ``rank`` is not in ``[1, min(m, n) // 2]`` (a non-positive
            rank silently slices the whole QR basis, so it is rejected too).
    """
    m, n = weight.shape
    max_rank = min(m, n) // 2
    if rank < 1 or rank > max_rank:
        raise ValueError(f"OLoRA rank {rank} must be in [1, min(m,n)//2={max_rank}] for {m}x{n}")
    q, r_mat = torch.linalg.qr(weight, mode="reduced")
    b = q[:, :rank].contiguous()
    a = r_mat[:rank, :].contiguous()
    w0 = weight - scale * (b @ a)
    return w0, b, a
