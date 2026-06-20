"""Tests for the SelfPlayTrainer best-response loop (T4.6).

Runs a TINY self-play stage on 2x2 (a few episodes/round, a 16-episode buffer,
the compute thread caps from the SDK config) so the real collect->store->update
loop is exercised end to end fast. torch is seeded via the trainer's master seed.
"""

from __future__ import annotations

import copy
import math

from src.services.trainer import SelfPlayTrainer


def _tiny(cfg: dict) -> dict:
    """Shrink the self-play / replay knobs so a real stage runs in well under a second."""
    c = copy.deepcopy(cfg)
    c["selfplay"]["episodes_per_round"] = 2
    c["selfplay"]["update_ratio"] = 1
    c["selfplay"]["rounds"] = 2
    c["algo"]["batch_episodes"] = 2
    c["replay"]["buffer_episodes"] = 16
    return c


def test_train_stage_alternates_roles_and_returns_history(cfg):
    """window_k=1 alternates cop/thief; each round reports a finite loss + capture rate."""
    trainer = SelfPlayTrainer(_tiny(cfg), seed=7, h=2, w=2, num_cops=1)
    history = trainer.train_stage(rounds=2)
    assert [h["role"] for h in history] == ["cop", "thief"]
    for record in history:
        assert math.isfinite(record["loss"])
        assert 0.0 <= record["capture_rate"] <= 1.0


def test_train_stage_default_rounds_from_config(cfg):
    """train_stage() with no arg runs selfplay.rounds rounds."""
    c = _tiny(cfg)
    c["selfplay"]["rounds"] = 1
    trainer = SelfPlayTrainer(c, seed=3, h=2, w=2, num_cops=1)
    assert len(trainer.train_stage()) == 1


def test_buffers_fill_with_both_roles_each_round(cfg):
    """Both role buffers receive episodes every round (store-both), regardless of trainee."""
    trainer = SelfPlayTrainer(_tiny(cfg), seed=11, h=2, w=2, num_cops=1)
    trainer.train_stage(rounds=1)
    assert len(trainer._cop_buf) > 0
    assert len(trainer._thief_buf) > 0
