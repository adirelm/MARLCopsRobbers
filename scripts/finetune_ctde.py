"""Full OLoRA-wrapped CTDE curriculum-finetune run (slow; routes through the SDK).

Pipeline: BC-pretrain the cop + thief encoders on the small grid, persist each
UNWRAPPED BC base + its content hash, then OLoRA-attach and finetune up the
curriculum ladder (3x3 -> 4x4 -> 5x5) via ``MarlSDK.finetune``, saving an OLoRA
bundle per role (reconstructable from the saved base + bundle). The finetune
orchestration lives in the SDK / services (this script is a thin wrapper). It is
the SLOW deferred T4.5 loop — run manually, NOT in the fast test suite:
``uv run python scripts/finetune_ctde.py``.
"""

from __future__ import annotations

from pathlib import Path

from src.marl.data.bc_dataset import build_bc_dataset
from src.marl.nets.bc_train import bc_train
from src.marl.olora_bundle import base_sha, save_bundle
from src.sdk.sdk import MarlSDK
from src.services.checkpoints import save_full_checkpoint
from src.utils.config_loader import load_config

# Finetune the curriculum tail (skip the 2x2 warmup): 3x3 -> 4x4 -> 5x5.
_FINETUNE_STAGES = [1, 2, 3]


def _bc_base(cfg: dict, grid: tuple[int, int], role: str, seed: int):
    """BC-pretrain a role base net on ``grid`` and return the trained net."""
    obs, scalars, actions, episode_ids, _manifest = build_bc_dataset(
        cfg, grid, int(cfg["bc"]["n_pairs"]), seed, role=role
    )
    net, val_acc = bc_train(cfg, grid, role, (obs, scalars, actions, episode_ids))
    print(f"[bc] {role} {grid} val_acc={val_acc:.3f}")
    return net


def main(cfg: dict | None = None) -> dict:
    """Run the BC -> OLoRA-attach -> curriculum-finetune pipeline; save the bundles."""
    cfg = cfg or load_config()
    seed = int(cfg["training"]["seeds"][0])
    grid = tuple(cfg["bc"]["pretrain_grids"][-1])
    base_dir = Path(cfg["paths"]["base_checkpoint_dir"])
    bundle_dir = Path(cfg["paths"]["olora_bundle_dir"])
    cop_base = _bc_base(cfg, grid, "cop", seed)
    thief_base = _bc_base(cfg, grid, "thief", seed)
    shas = {"cop": base_sha(cop_base), "thief": base_sha(thief_base)}
    save_full_checkpoint(base_dir / "bc_base.pt", {"cop": cop_base, "thief": thief_base})
    out = MarlSDK(cfg).finetune(seed, _FINETUNE_STAGES, cop_base, thief_base, olora=True)
    save_bundle(bundle_dir / "cop.bundle.pt", out["cop_net"], shas["cop"])
    save_bundle(bundle_dir / "thief.bundle.pt", out["thief_net"], shas["thief"])
    print(f"[finetune] stages={[s['stage'] for s in out['history']]} -> {bundle_dir}")
    return out


if __name__ == "__main__":
    main()
