"""The slow-pipeline service layer behind the manual scripts (ADR-0002 SDK-only launchers).

Drives both entry points with stub SDKs so the orchestration contract — which stages are
finetuned, what gets written where, what the sweep records — is asserted without running
real training.
"""

from __future__ import annotations

import json

from src.services import pipelines


class _StubSdk:
    """Records the calls the pipelines make instead of training."""

    def __init__(self) -> None:
        self.finetune_calls: list[tuple] = []
        self.train_calls: list[tuple] = []

    def finetune(self, seed, stage_indices, cop_net, thief_net, olora=True):
        self.finetune_calls.append((seed, tuple(stage_indices), olora))
        return {"history": [{"stage": s} for s in stage_indices], "cop_net": cop_net, "thief_net": thief_net}

    def train(self, algorithm, seed, stage_idx):
        self.train_calls.append((algorithm, seed, stage_idx))
        return [{"round": 0, "role": "cop", "loss": 0.1, "capture_rate": 0.5}]


def test_ablation_sweep_covers_every_arm_and_seed_and_appends(tmp_path, cfg):
    """Every (algorithm, seed) pair trains once and one JSONL record lands per run."""
    cfg["paths"]["runs_dir"] = str(tmp_path)
    cfg["training"]["seeds"] = [7, 17]
    sdk = _StubSdk()

    records = pipelines.ablation_sweep(sdk, cfg, ["qmix", "iql"], stage_idx=1)

    assert sdk.train_calls == [("qmix", 7, 1), ("qmix", 17, 1), ("iql", 7, 1), ("iql", 17, 1)]
    assert len(records) == 4
    written = (tmp_path / "ablation.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(written) == 4
    assert json.loads(written[0])["algorithm"] == "qmix"


def test_bc_olora_pipeline_finetunes_the_curriculum_tail_and_saves_bundles(tmp_path, cfg, monkeypatch):
    """BC bases are built per role, then OLoRA-finetuned over stages 1-3 and bundled."""
    cfg["paths"]["base_checkpoint_dir"] = str(tmp_path / "base")
    cfg["paths"]["olora_bundle_dir"] = str(tmp_path / "bundles")
    built: list[str] = []
    saved: list[str] = []

    monkeypatch.setattr(
        pipelines, "_bc_base", lambda cfg_, grid, role, seed: built.append(role) or f"net-{role}"
    )
    monkeypatch.setattr(pipelines, "base_sha", lambda net: f"sha-{net}")
    monkeypatch.setattr(pipelines, "save_full_checkpoint", lambda path, nets: saved.append(str(path)))
    monkeypatch.setattr(pipelines, "save_bundle", lambda path, net, sha: saved.append(str(path)))
    sdk = _StubSdk()

    out = pipelines.bc_olora_pipeline(sdk, cfg)

    assert built == ["cop", "thief"]
    assert sdk.finetune_calls == [(int(cfg["training"]["seeds"][0]), (1, 2, 3), True)]
    assert [s["stage"] for s in out["history"]] == [1, 2, 3]
    assert any("bc_base.pt" in p for p in saved)
    assert sum("bundle.pt" in p for p in saved) == 2
