"""Unit tests for the QMIX monotone hypernetwork mixer (T4.1, P4a Stage 3).

Pure-tensor gates for the QMIX mixer: monotonicity (``∂Q_tot/∂Q_i ≥ 0`` via
softplus weights), IGM consistency (eq5 — decentralized argmax matches the
joint maximizing the mixer), a NEGATIVE control (a non-monotone signed-weight
variant FAILS monotonicity, so the test has teeth), and output shape ``[B,1]``.
No training loop. torch is seeded for determinism.
"""

from __future__ import annotations

import copy
import itertools

import pytest
import torch
from torch.nn.functional import elu, softplus

from src.marl.mixers.qmix_mixer import QmixMixer
from src.marl.nets.dims import state_dim

SEED = 7
N_AGENTS = 2


def _cfg() -> dict:
    """Return a minimal config with the QMIX/env keys the mixer reads."""
    return {
        "algo": {"mixer": {"embed_dim": 32}},
        "env": {"view_radius_max": 2},
    }


def _state(batch: int) -> torch.Tensor:
    """Return a random global state ``[batch, state_dim]`` (=77)."""
    return torch.randn(batch, state_dim(_cfg()))


def test_output_shape_is_b_one() -> None:
    """QMIX forward returns ``[B, 1]`` for an N=2 fixture."""
    torch.manual_seed(SEED)
    mixer = QmixMixer(_cfg(), N_AGENTS)
    q_agents = torch.randn(4, N_AGENTS)
    q_tot = mixer(q_agents, _state(4))
    assert q_tot.shape == (4, 1)


def test_monotonic_in_each_agent_q() -> None:
    """Perturbing one ``Q_i`` UP never decreases ``q_tot`` (δ>0, all i)."""
    torch.manual_seed(SEED)
    mixer = QmixMixer(_cfg(), N_AGENTS)
    q_agents = torch.randn(8, N_AGENTS)
    state = _state(8)
    delta = 0.5
    base = mixer(q_agents, state)
    for i in range(N_AGENTS):
        bumped = q_agents.clone()
        bumped[:, i] += delta
        assert torch.all(mixer(bumped, state) >= base - 1e-6)


def test_igm_decentralized_argmax_matches_joint() -> None:
    """IGM (eq5): per-agent argmax joint == argmax of mixer over joint set."""
    torch.manual_seed(SEED)
    mixer = QmixMixer(_cfg(), N_AGENTS)
    n_actions = 4
    state = _state(1)
    per_agent_q = torch.randn(N_AGENTS, n_actions)
    greedy = tuple(int(per_agent_q[i].argmax()) for i in range(N_AGENTS))
    best_joint, best_val = None, None
    with torch.no_grad():
        for joint in itertools.product(range(n_actions), repeat=N_AGENTS):
            chosen = torch.tensor([[per_agent_q[i, a] for i, a in enumerate(joint)]])
            val = float(mixer(chosen, state))
            if best_val is None or val > best_val:
                best_joint, best_val = joint, val
    assert best_joint == greedy


def _broken_forward(mixer: QmixMixer, q_agents: torch.Tensor, state: torch.Tensor, raw: str) -> torch.Tensor:
    """Mixer forward that SKIPS softplus on the ``raw`` weights ("w1"/"w2"/"both")."""
    batch = state.shape[0]
    w1 = mixer.hyper_w1(state)
    w2 = mixer.hyper_w2(state)
    w1 = w1 if raw in ("both", "w1") else softplus(w1)
    w2 = w2 if raw in ("both", "w2") else softplus(w2)
    w1 = w1.view(batch, mixer.n_agents, mixer.embed_dim)
    w2 = w2.view(batch, mixer.embed_dim, 1)
    b1 = mixer.hyper_b1(state).view(batch, 1, mixer.embed_dim)
    v = mixer.value(state).view(batch, 1, 1)
    hidden = elu(torch.bmm(q_agents.unsqueeze(1), w1) + b1)
    return (torch.bmm(hidden, w2) + v).view(batch, 1)


@pytest.mark.parametrize("raw", ["both", "w1", "w2"])
def test_negative_control_missing_softplus_breaks_monotonicity(raw: str) -> None:
    """Dropping softplus on w1, w2, OR both makes the mixer non-monotone (teeth).

    Proves the monotonicity gate would catch a regression that removed softplus
    from EITHER hypernet weight, not only from both at once.
    """
    torch.manual_seed(SEED)
    mixer = QmixMixer(_cfg(), N_AGENTS)
    q_agents = torch.randn(64, N_AGENTS)
    state = _state(64)
    base = _broken_forward(mixer, q_agents, state, raw)
    violated = False
    for i in range(N_AGENTS):
        bumped = q_agents.clone()
        bumped[:, i] += 0.5
        if torch.any(_broken_forward(mixer, bumped, state, raw) < base - 1e-6):
            violated = True
    assert violated, f"variant raw={raw!r} should be able to violate monotonicity"


def test_clone_is_independent() -> None:
    """A deep copy mixer forwards identically yet is a separate module."""
    torch.manual_seed(SEED)
    mixer = QmixMixer(_cfg(), N_AGENTS)
    twin = copy.deepcopy(mixer)
    q_agents = torch.randn(2, N_AGENTS)
    state = _state(2)
    assert torch.allclose(mixer(q_agents, state), twin(q_agents, state))
