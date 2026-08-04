"""Long-running pipeline orchestration — the service layer behind the manual scripts.

Keeps the slow BC→OLoRA→curriculum-finetune pipeline and the ablation sweep OUT of
`scripts/`, so those launchers import ONLY `src.sdk` (ADR-0002: the SDK is the single
business-logic surface; enforced by `tests/architecture/test_v3_gates.py`).

Input: a loaded config + the run axes (seed / algorithms / stage).
Output: the trained artifacts written to the configured dirs + the run records.
Setup: none beyond a valid config — both entry points are pure orchestration over
existing services and are safe to call from a thin script or the SDK.
"""

from __future__ import annotations

from pathlib import Path

from src.marl.data.bc_dataset import build_bc_dataset
from src.marl.nets.bc_train import bc_train
from src.marl.olora_bundle import base_sha, save_bundle
from src.services.checkpoints import save_full_checkpoint
from src.services.sweep import run_sweep


def _bc_base(cfg: dict, grid: tuple[int, int], role: str, seed: int) -> object:
    """BC-pretrain one role's base net on ``grid``; return the trained net."""
    obs, scalars, actions, episode_ids, _manifest = build_bc_dataset(
        cfg, grid, int(cfg["bc"]["n_pairs"]), seed, role=role
    )
    net, val_acc = bc_train(cfg, grid, role, (obs, scalars, actions, episode_ids))
    print(f"[bc] {role} {grid} val_acc={val_acc:.3f}")
    return net


def bc_olora_pipeline(sdk: object, cfg: dict) -> dict:
    """BC-pretrain both roles, then OLoRA-finetune up the curriculum; save the bundles.

    Args:
        sdk: The :class:`~src.sdk.sdk.MarlSDK` providing the ``finetune`` seam.
        cfg: Loaded config (BC grids, seeds, checkpoint/bundle dirs).

    Returns:
        The finetune result dict (``history`` / ``cop_net`` / ``thief_net``).
    """
    seed = int(cfg["training"]["seeds"][0])
    grid = tuple(cfg["bc"]["pretrain_grids"][-1])
    base_dir = Path(cfg["paths"]["base_checkpoint_dir"])
    bundle_dir = Path(cfg["paths"]["olora_bundle_dir"])
    cop_base = _bc_base(cfg, grid, "cop", seed)
    thief_base = _bc_base(cfg, grid, "thief", seed)
    shas = {"cop": base_sha(cop_base), "thief": base_sha(thief_base)}
    save_full_checkpoint(base_dir / "bc_base.pt", {"cop": cop_base, "thief": thief_base})
    stages = [int(s) for s in cfg["bc"]["finetune_stages"]]  # which curriculum rungs to train
    out = sdk.finetune(seed, stages, cop_base, thief_base, olora=True)
    save_bundle(bundle_dir / "cop.bundle.pt", out["cop_net"], shas["cop"])
    save_bundle(bundle_dir / "thief.bundle.pt", out["thief_net"], shas["thief"])
    print(f"[finetune] stages={[s['stage'] for s in out['history']]} -> {bundle_dir}")
    return out


def ablation_sweep(sdk: object, cfg: dict, algorithms: list[str], stage_idx: int = 0) -> list[dict]:
    """Sweep ``algorithms`` x ``training.seeds`` at one stage; append the run records.

    Args:
        sdk: The SDK exposing ``train(algorithm, seed, stage_idx)``.
        cfg: Loaded config (seeds + the runs dir).
        algorithms: The arms to sweep (e.g. ``["qmix", "vdn", "iql"]``).
        stage_idx: Curriculum stage index to train at.

    Returns:
        One record per run (also appended to ``paths.runs_dir/ablation.jsonl``).
    """
    out = Path(cfg["paths"]["runs_dir"]) / "ablation.jsonl"
    seeds = list(cfg["training"]["seeds"])
    records = run_sweep(sdk, cfg, algorithms, seeds, stage_idx, out)
    print(f"[sweep] {len(records)} runs ({algorithms} x {len(seeds)} seeds) -> {out}")
    return records
