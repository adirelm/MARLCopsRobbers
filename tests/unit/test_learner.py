"""Unit smokes for the CTDE QmixLearner base (T4.2, P4b Stage 3).

FAST deterministic gates on the exact Double-DQN / BPTT / active-mask update:

* overfit-one-episode — repeating the update on one tiny fixed N=1 episode
  drives the loss markedly down (the math actually learns),
* masked-loss — perturbing a ``filled=0`` pad step's target leaves loss + grad
  unchanged (pad steps contribute nothing),
* hard-target-sync — target params are frozen between syncs and copied exactly
  at ``target_update_interval`` updates,
* telemetry — ``update`` returns ``{loss, grad_norm, q_tot}``,
* ∂Q_tot/∂q_inactive == 0 — an ``active=[True, False]`` slot's q never moves
  Q_tot (zeroed before the mixer),
* active-MEAN reward — two equal cop rewards give ``r_team`` == the shared
  value, not 2x (mean over active cops, never a sum).

torch is seeded for determinism.
"""

from __future__ import annotations

import copy

import torch

from src.marl.learner._learner_helpers import to_tensors
from src.marl.learner.learner_base import QmixLearner
from src.marl.mixers.qmix_mixer import QmixMixer
from src.marl.mixers.vdn_mixer import VdnMixer
from src.marl.nets.dims import state_dim
from tests.unit._learner_fixtures import make_batch

SEED = 7


def _cop_learner(cfg: dict, n_agents: int) -> QmixLearner:
    """Build a VDN-mixer QmixLearner for ``n_agents`` cops (seeded)."""
    torch.manual_seed(SEED)
    return QmixLearner(cfg, role="cop", n_agents=n_agents, mixer=VdnMixer())


def test_update_returns_telemetry_keys(cfg: dict) -> None:
    """update returns a dict carrying exactly loss / grad_norm / q_tot."""
    learner = _cop_learner(cfg, n_agents=1)
    batch = make_batch(b=2, t=3, n=1, active=[True], seed=1)
    out = learner.update(batch)
    assert {"loss", "grad_norm", "q_tot"} <= set(out.keys())
    for key in ("loss", "grad_norm", "q_tot"):
        assert isinstance(float(out[key]), float)


def test_overfit_one_episode_loss_decreases(cfg: dict) -> None:
    """Repeating the update on ONE fixed N=1 episode drives the loss down."""
    learner = _cop_learner(cfg, n_agents=1)
    batch = make_batch(b=1, t=3, n=1, active=[True], seed=2)
    first = learner.update(batch)["loss"]
    for _ in range(60):
        last = learner.update(batch)["loss"]
    assert last < first * 0.5  # markedly lower after overfitting one episode


def test_masked_pad_step_contributes_zero_gradient(cfg: dict) -> None:
    """A filled=0 pad step contributes 0: perturbing its target leaves loss flat."""
    learner = _cop_learner(cfg, n_agents=1)
    # t=3 but only the first 2 steps are real (step 2 is a pad: filled=0).
    batch = make_batch(b=1, t=3, n=1, active=[True], seed=3, filled=[True, True, False])
    base = learner.update(copy.deepcopy(batch))["loss"]
    learner2 = _cop_learner(cfg, n_agents=1)
    bumped = copy.deepcopy(batch)
    bumped["reward"][:, 2, :] += 99.0  # perturb ONLY the padded step's reward
    after = learner2.update(bumped)["loss"]
    assert abs(base - after) < 1e-6


def test_hard_target_sync_freezes_then_copies(cfg: dict) -> None:
    """Target params stay frozen between syncs and copy exactly at the interval."""
    learner = _cop_learner(cfg, n_agents=1)
    learner.target_update_interval = 3  # sync on the 3rd update
    batch = make_batch(b=1, t=2, n=1, active=[True], seed=4)
    before = next(learner.target_net.parameters()).clone()
    learner.update(copy.deepcopy(batch))
    learner.update(copy.deepcopy(batch))
    mid = next(learner.target_net.parameters())
    assert torch.equal(before, mid)  # frozen for the first two updates
    learner.update(copy.deepcopy(batch))  # 3rd update triggers the hard sync
    synced = next(learner.target_net.parameters())
    online = next(learner.online_net.parameters())
    assert torch.equal(synced, online)  # copied online -> target at the interval
    assert not torch.equal(synced, before)  # and it actually changed


def test_inactive_slot_q_does_not_move_q_tot(cfg: dict) -> None:
    """∂Q_tot/∂q_inactive == 0: zeroing an inactive slot's q never moves Q_tot."""
    learner = _cop_learner(cfg, n_agents=2)
    batch = make_batch(b=1, t=2, n=2, active=[True, False], seed=5)
    out_a = learner.update(copy.deepcopy(batch))["q_tot"]
    learner2 = _cop_learner(cfg, n_agents=2)
    bumped = copy.deepcopy(batch)
    # Drive the inactive slot (idx 1) obs to a wildly different value; Q_tot
    # must be identical because its q_i is zeroed before the mixer.
    bumped["obs"][:, :, 1] += 50.0
    bumped["scalars"][:, :, 1] += 50.0
    out_b = learner2.update(bumped)["q_tot"]
    assert abs(out_a - out_b) < 1e-5


def test_team_reward_is_active_mean_not_sum(cfg: dict) -> None:
    """Two equal active cop rewards give r_team == the shared value (mean, not sum)."""
    learner = _cop_learner(cfg, n_agents=2)
    batch = make_batch(b=4, t=2, n=2, active=[True, True], reward=3.0, seed=6)
    r_team = learner._team_reward(to_tensors(batch, learner.device))
    # Active-MEAN over two cops sharing reward 3.0 == 3.0 (NOT 6.0).
    assert torch.allclose(r_team, torch.full_like(r_team, 3.0))


def test_team_reward_ignores_inactive_slot(cfg: dict) -> None:
    """An inactive (phantom) cop's zero reward never dilutes the active mean."""
    learner = _cop_learner(cfg, n_agents=2)
    # Only slot 0 active; slot 1 is a zero-reward phantom -> mean over actives = 1.0.
    batch = make_batch(b=2, t=2, n=2, active=[True, False], reward=1.0, seed=8)
    batch["reward"][:, :, 1] = 0.0  # phantom slot carries no reward
    r_team = learner._team_reward(to_tensors(batch, learner.device))
    assert torch.allclose(r_team, torch.ones_like(r_team))


def test_double_q_decoupling_uses_online_argmax(cfg: dict) -> None:
    """Double-DQN bootstraps Q_target at the ONLINE argmax; toggling double_q changes it.

    Pins the convergence-critical online/target decoupling: a* is chosen by the
    online net (double_q=True) vs the target net (False); with online != target
    the two bootstraps differ, so a future net-source swap is caught.
    """
    torch.manual_seed(SEED)
    learner = QmixLearner(cfg, role="cop", n_agents=2, mixer=QmixMixer(cfg, 2))
    with torch.no_grad():  # perturb online so its argmax can diverge from target's
        for p in learner.online_net.parameters():
            p.add_(torch.randn_like(p) * 0.7)
    data = to_tensors(make_batch(b=4, t=3, n=2, active=[True, True], seed=9), learner.device)
    q_online_next = learner._unroll(learner.online_net, data)[:, 1:]
    q_target_next = learner._unroll(learner.target_net, data)[:, 1:]
    learner.double_q = True
    ddqn = learner._compute_target(data, q_online_next, q_target_next)
    learner.double_q = False
    vanilla = learner._compute_target(data, q_online_next, q_target_next)
    assert not torch.allclose(ddqn, vanilla)


def test_qmix_inactive_slot_has_zero_q_gradient(cfg: dict) -> None:
    """QMIX (primary mixer): a zeroed inactive cop q_i gets exactly zero gradient."""
    torch.manual_seed(SEED)
    mixer = QmixMixer(cfg, 2)
    q = torch.randn(3, 2, requires_grad=True)
    active = torch.tensor([1.0, 0.0])  # cop 1 inactive
    state = torch.randn(3, state_dim(cfg))
    mixer(q * active, state).sum().backward()
    assert torch.all(q.grad[:, 1] == 0.0)  # inactive slot: zero gradient
    assert torch.any(q.grad[:, 0] != 0.0)  # active slot: real gradient
