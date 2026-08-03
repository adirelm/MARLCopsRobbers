"""Curriculum finetune — carry the agent nets across stages (T4.6 + the T4.5 loop).

``finetune_curriculum`` runs self-play stage by stage (e.g. 3x3 -> 4x4 -> 5x5),
CARRYING the trained cop/thief nets forward into the next stage's
:class:`~src.services.trainer.SelfPlayTrainer` (the encoder + global-state widths
are stage-invariant by design, so the same net transfers). When the carried nets
are OLoRA-wrapped (the finetune path), only the adapters + head + mixer train at
each stage while the frozen base + GRU transfer unchanged. ``stage_params`` reads
a curriculum stage's ``(h, w, num_cops)`` from config — the single stage resolver
reused by the SDK training entry (distinct from the env-level promotion ladder
:class:`src.marl.env.curriculum.Curriculum`, which walks by capture rate).
"""

from __future__ import annotations

from src.marl.nets.agent_net import RecurrentQNet
from src.services.trainer import SelfPlayTrainer


def stage_params(cfg: dict, stage_idx: int) -> tuple[int, int, int]:
    """Return ``(h, w, num_cops)`` for curriculum stage ``stage_idx`` (config-derived)."""
    curriculum = cfg["env"]["curriculum"]
    h, w = curriculum["stages"][stage_idx]
    return int(h), int(w), int(curriculum["num_cops_by_stage"][stage_idx])


def finetune_curriculum(  # noqa: PLR0913 — cfg + seed + stages + 2 carried nets + rounds are distinct
    cfg: dict,
    seed: int,
    stage_indices: list[int],
    cop_net: RecurrentQNet,
    thief_net: RecurrentQNet,
    rounds_per_stage: int | None = None,
) -> dict:
    """Run self-play across curriculum stages, carrying the nets forward.

    Args:
        cfg: Loaded config (the algorithm should already be selected upstream).
        seed: Master seed (each stage's trainer is seeded from it).
        stage_indices: Ordered curriculum stage indices (e.g. ``[1, 2, 3]``).
        cop_net: The starting cop net (carried + mutated across stages; may be
            OLoRA-wrapped for the parameter-efficient finetune path).
        thief_net: The starting thief net (carried across stages).
        rounds_per_stage: Rounds per stage (default ``selfplay.rounds``).

    Returns:
        ``{"history": [{stage, grid, rounds}, ...], "cop_net", "thief_net"}`` —
        the per-stage round histories and the final carried nets.
    """
    history = []
    cop, thief = cop_net, thief_net
    for idx in stage_indices:
        h, w, num_cops = stage_params(cfg, idx)
        trainer = SelfPlayTrainer(cfg, seed, h, w, num_cops, cop_net=cop, thief_net=thief)
        rounds = trainer.train_stage(rounds_per_stage)
        cop, thief = trainer.cop_net(), trainer.thief_net()
        history.append({"stage": idx, "grid": [h, w], "rounds": rounds})
    return {"history": history, "cop_net": cop, "thief_net": thief}
