"""Unit smokes for the role learners (CopLearner / IqlLearner / ThiefLearner).

P4b Stage 4 — FAST deterministic gates on the role-specific learner branches:

* CopLearner builds the RIGHT mixer per ``algo.mixer.type`` (qmix -> QmixMixer, vdn ->
  VdnMixer) and trains through it (centralized base path),
* IqlLearner has EXACTLY ONE AdamW param-group (``lr_agent`` only — no
  ``lr_mixer``) and NO mixer attribute (the per-agent eq4 branch drops it),
* IqlLearner overfits ONE tiny episode (per-agent masked loss decreases),
* ThiefLearner is single-agent with A=``a_thief``=4 (q/mask widths are 4, the
  env's ``a_cop`` mask is sliced to the thief head before argmax).

torch is seeded for determinism.
"""

from __future__ import annotations

import copy

import torch

from src.marl.learner.learners import CopLearner, IqlLearner, ThiefLearner
from src.marl.mixers.qmix_mixer import QmixMixer
from src.marl.mixers.vdn_mixer import VdnMixer
from src.marl.nets.dims import action_dim
from tests.unit._learner_fixtures import make_batch

SEED = 7


def _seeded() -> None:
    """Seed torch so each learner build/update is deterministic."""
    torch.manual_seed(SEED)


def test_cop_learner_builds_qmix_mixer_when_mixer_type_qmix(cfg: dict) -> None:
    """algo.mixer.type == 'qmix' -> CopLearner wires a QmixMixer (online + frozen target)."""
    _seeded()
    learner = CopLearner(cfg, n_agents=2)
    assert isinstance(learner.mixer, QmixMixer)
    assert isinstance(learner.target_mixer, QmixMixer)
    assert not any(p.requires_grad for p in learner.target_mixer.parameters())


def test_cop_learner_builds_vdn_mixer_when_mixer_type_vdn(cfg: dict) -> None:
    """algo.mixer.type == 'vdn' -> CopLearner wires a VdnMixer (the ablation toggle, ADR-D3-B)."""
    _seeded()
    vdn_cfg = copy.deepcopy(cfg)
    vdn_cfg["algo"]["mixer"]["type"] = "vdn"
    learner = CopLearner(vdn_cfg, n_agents=2)
    assert isinstance(learner.mixer, VdnMixer)


def test_cop_learner_update_runs_through_mixer(cfg: dict) -> None:
    """A CopLearner update returns telemetry and a finite loss (mixer path runs)."""
    _seeded()
    learner = CopLearner(cfg, n_agents=2)
    out = learner.update(make_batch(b=2, t=3, n=2, active=[True, True], seed=1))
    assert {"loss", "grad_norm", "q_tot"} <= set(out)
    assert torch.isfinite(torch.tensor(out["loss"]))


def test_iql_learner_has_single_param_group_no_mixer(cfg: dict) -> None:
    """IqlLearner: NO mixer attr + exactly ONE AdamW group at lr_agent (no lr_mixer)."""
    _seeded()
    learner = IqlLearner(cfg, n_agents=2)
    assert learner.mixer is None
    assert learner.target_mixer is None
    groups = learner.optimizer.param_groups
    assert len(groups) == 1  # single group: no separate lr_mixer group
    assert groups[0]["lr"] == cfg["algo"]["lr_agent"]


def test_iql_learner_overfits_one_episode(cfg: dict) -> None:
    """Repeating the per-agent IQL update on ONE fixed episode drives the loss down."""
    _seeded()
    learner = IqlLearner(cfg, n_agents=2)
    batch = make_batch(b=1, t=3, n=2, active=[True, True], seed=2)
    first = learner.update(batch)["loss"]
    for _ in range(60):
        last = learner.update(batch)["loss"]
    assert last < first * 0.5


def test_iql_telemetry_keys_present(cfg: dict) -> None:
    """The IQL branch still returns the {loss, grad_norm, q_tot} telemetry dict."""
    _seeded()
    learner = IqlLearner(cfg, n_agents=1)
    out = learner.update(make_batch(b=2, t=2, n=1, active=[True], seed=4))
    assert {"loss", "grad_norm", "q_tot"} <= set(out)


def test_thief_learner_uses_action_dim_four(cfg: dict) -> None:
    """ThiefLearner is single-agent with A == a_thief == 4 (q-head width is 4)."""
    _seeded()
    learner = ThiefLearner(cfg)
    assert learner.n_agents == 1
    assert learner.mixer is None
    assert learner.online_net.head.out_features == action_dim(cfg, "thief")
    assert action_dim(cfg, "thief") == 4


def test_thief_learner_slices_cop_width_mask(cfg: dict) -> None:
    """ThiefLearner consumes an a_cop(5)-wide env mask, slicing it to a_thief(4)."""
    _seeded()
    learner = ThiefLearner(cfg)
    # make_batch emits a 5-wide (a_cop) next_legal_mask + 5-wide actions; the
    # thief learner must slice both to its 4-action head without an index error.
    batch = make_batch(b=2, t=2, n=1, active=[True], seed=5)
    batch["actions"] = (batch["actions"] % action_dim(cfg, "thief")).astype(batch["actions"].dtype)
    out = learner.update(batch)
    assert torch.isfinite(torch.tensor(out["loss"]))
