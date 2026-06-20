"""CI tests for the BC supervised pretraining loop (T4.4, P4c Stage 2).

Runs on the TINY vendored fixtures (tests/fixtures/bc_mini_*.npz) so the default
``uv run pytest`` stays fast. Asserts the overfit smoke: the cross-entropy loss
decreases and the held-out val-acc BEATS the per-role random floor by a margin
(cop 1/5, thief 1/4) — NOT the full per-grid gate (a tiny set under-trains). Also
pins ``gate_for`` to the per-grid config value and the thief role pairing. torch
is seeded for determinism.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.marl.data.bc_npz import load_npz
from src.marl.nets.agent_net import RecurrentQNet
from src.marl.nets.bc_train import _val_accuracy, bc_fit, bc_train, gate_for
from src.marl.nets.dims import obs_dim

SEED = 4455
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_COP_FLOOR = 1.0 / 5.0  # a_cop = 5
_THIEF_FLOOR = 1.0 / 4.0  # a_thief = 4
_MARGIN = 0.10  # beats-random margin for the overfit smoke


def _load(grid: tuple[int, int], role: str):
    """Load a vendored fixture's (obs, scalars, actions, episode_ids) arrays."""
    name = f"bc_mini_{grid[0]}x{grid[1]}_{role}.npz"
    obs, scalars, actions, episode_ids, _manifest = load_npz(_FIXTURES / name)
    return obs, scalars, actions, episode_ids


@pytest.mark.parametrize(
    ("grid", "role", "floor"),
    [
        ((2, 2), "cop", _COP_FLOOR),
        ((2, 2), "thief", _THIEF_FLOOR),
        ((3, 3), "cop", _COP_FLOOR),
        ((3, 3), "thief", _THIEF_FLOOR),
    ],
)
def test_bc_train_beats_random_floor(cfg, grid, role, floor):
    """Val-acc beats the per-role random floor by a margin on the tiny set."""
    torch.manual_seed(SEED)
    _net, val_acc = bc_train(cfg, grid, role, _load(grid, role))
    assert val_acc >= floor + _MARGIN


def test_bc_fit_loss_decreases(cfg):
    """The CE loss at the last epoch is well below the first (overfit smoke)."""
    torch.manual_seed(SEED)
    _net, losses, _val = bc_fit(cfg, (3, 3), "cop", _load((3, 3), "cop"))
    assert len(losses) >= 2
    assert losses[-1] < losses[0]


def test_bc_fit_learns_separable_set_beyond_majority(cfg):
    """A balanced, separable synthetic set is learned to high val-acc.

    This is the teeth the real cop fixtures cannot provide: those are heavily
    class-imbalanced, so a mode-collapsed constant classifier already clears the
    random floor. A balanced separable set is NOT clearable by mode collapse
    (its majority baseline is 1/5), so high val-acc here proves the forward + CE
    + label + optimizer wiring genuinely learns the obs->action mapping.
    """
    torch.manual_seed(SEED)
    deep = {**cfg, "bc": {**cfg["bc"], "epochs": 200}}
    n_per, n_cls = 24, 5
    labels = np.repeat(np.arange(n_cls), n_per).astype(np.int64)
    np.random.default_rng(SEED).shuffle(labels)
    n = labels.shape[0]
    width = 2 * int(cfg["env"]["view_radius_max"]) + 1
    obs = np.zeros((n, int(cfg["env"]["obs_channels"]), width, width), np.float32)
    scal = np.zeros((n, int(cfg["env"]["obs_scalars"])), np.float32)
    for i, y in enumerate(labels):
        obs[i] = float(y) + 1.0  # class-distinct (linearly separable) pattern
    episode_ids = (np.arange(n) // 2).astype(np.int64)  # multi-episode -> non-trivial split
    _net, _losses, val_acc = bc_fit(deep, (3, 3), "cop", (obs, scal, labels, episode_ids))
    assert val_acc > 0.8  # >> the 1/5 mode-collapse baseline on a balanced set


def test_val_accuracy_empty_split_returns_zero(cfg):
    """_val_accuracy on an empty held-out split returns 0.0 (defensive guard)."""
    torch.manual_seed(SEED)
    net = RecurrentQNet(cfg, "cop", 2)
    empty = torch.zeros((0, 1, obs_dim(cfg)))
    assert _val_accuracy(net, empty, torch.zeros((0,), dtype=torch.long)) == 0.0


def test_bc_train_returns_recurrent_qnet_with_role_head(cfg):
    """bc_train returns a RecurrentQNet whose head width matches the role actions."""
    torch.manual_seed(SEED)
    net, _val = bc_train(cfg, (3, 3), "thief", _load((3, 3), "thief"))
    assert net.head.out_features == cfg["env"]["actions"]["a_thief"]


def test_cop_net_built_with_two_agents(cfg):
    """CopNet pretrains with n_agents=2 (fixed encoder input width across ladder)."""
    torch.manual_seed(SEED)
    net, _val = bc_train(cfg, (3, 3), "cop", _load((3, 3), "cop"))
    assert net.n_agents == 2


def test_thief_net_built_with_one_agent(cfg):
    """ThiefNet pretrains with n_agents=1."""
    torch.manual_seed(SEED)
    net, _val = bc_train(cfg, (3, 3), "thief", _load((3, 3), "thief"))
    assert net.n_agents == 1


def test_gate_for_reads_per_grid_config(cfg):
    """gate_for returns bc.val_acc_gate_by_grid[min(h, w)] verbatim."""
    assert gate_for(cfg, (2, 2)) == cfg["bc"]["val_acc_gate_by_grid"][2]
    assert gate_for(cfg, (3, 3)) == cfg["bc"]["val_acc_gate_by_grid"][3]
    # rectangular grid keys on min(h, w)
    assert gate_for(cfg, (3, 5)) == cfg["bc"]["val_acc_gate_by_grid"][3]


def test_thief_fixture_uses_thief_action_range(cfg):
    """The thief fixture's labels live in [0, a_thief) (role pairing sanity)."""
    _obs, _sc, actions, _eids = _load((3, 3), "thief")
    assert int(actions.max()) < cfg["env"]["actions"]["a_thief"]
