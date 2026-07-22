"""Full OLoRA-wrapped CTDE curriculum-finetune run (slow; SDK-only launcher).

BC-pretrains the cop + thief encoders, persists each unwrapped base + its content hash,
then OLoRA-attaches and finetunes up the curriculum ladder (3x3 -> 4x4 -> 5x5), saving an
OLoRA bundle per role. All orchestration lives in the service layer behind
``MarlSDK.run_bc_olora_pipeline`` (ADR-0002) — this file only parses config and calls it.
Slow deferred T4.5 loop; run manually: ``uv run python scripts/finetune_ctde.py``.
"""

from __future__ import annotations

from src.sdk import MarlSDK
from src.utils.config_loader import load_config


def main(cfg: dict | None = None) -> dict:
    """Run the BC -> OLoRA-attach -> curriculum-finetune pipeline; save the bundles."""
    cfg = cfg or load_config()
    return MarlSDK(cfg).run_bc_olora_pipeline()


if __name__ == "__main__":
    main()
