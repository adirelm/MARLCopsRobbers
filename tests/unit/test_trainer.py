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
    c["replay"]["min_replay_episodes"] = 2  # small warmup so the tiny stage still exercises updates
    return c


def test_warmup_skips_updates_until_min_replay_episodes(cfg):
    """With min_replay above the episodes collected, NO learner update runs (loss == 0.0)."""
    c = _tiny(cfg)
    c["replay"]["min_replay_episodes"] = 999  # never reached in a 2-episode round -> warmup only
    history = SelfPlayTrainer(c, seed=7, h=2, w=2, num_cops=1).train_stage(rounds=2)
    assert all(record["loss"] == 0.0 for record in history)  # collect-only, no update during warmup


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


def _count_thief_pool_draws(trainer: SelfPlayTrainer, rounds: tuple[int, ...]) -> tuple[int, list]:
    """Return (#thief_pool.sample() draws, [frozen opponent per round]) over cop rounds."""
    calls = {"n": 0}
    real = trainer._thief_pool.sample

    def counting():
        calls["n"] += 1
        return real()

    trainer._thief_pool.sample = counting  # type: ignore[method-assign]
    opps = [trainer._policies("cop", r)[1] for r in rounds]
    return calls["n"], opps


def test_window_k_freezes_opponent_within_a_window(cfg):
    """window_k>1 holds ONE frozen opponent for the whole window (codex F4).

    Rounds 0 and 1 of a window_k=2 window draw the opponent pool exactly ONCE and
    reuse the SAME frozen opponent object — the documented "frozen for window_k
    rounds" semantics, previously violated by per-round re-sampling.
    """
    c = _tiny(cfg)
    c["selfplay"]["window_k"] = 2
    trainer = SelfPlayTrainer(c, seed=5, h=2, w=2, num_cops=1)
    draws, opps = _count_thief_pool_draws(trainer, rounds=(0, 1))  # one window
    assert draws == 1  # sampled ONCE for the window
    assert opps[0] is opps[1]  # same frozen opponent held across the window


def test_window_k_one_samples_per_round(cfg):
    """window_k=1 (the GRADED setting) samples the opponent every round (per-round)."""
    trainer = SelfPlayTrainer(_tiny(cfg), seed=5, h=2, w=2, num_cops=1)  # cfg window_k == 1
    draws, _ = _count_thief_pool_draws(trainer, rounds=(0, 1))
    assert draws == 2  # each round is its own window -> per-round sampling
