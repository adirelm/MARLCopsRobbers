"""Unit tests for CTDE learner helpers (T4.2, P4b Stage 2).

FAST deterministic gates on the four pure tensor helpers used by the BPTT
update: masked_huber (filled-masked mean, no grad through mask), bptt_unroll
(GRU roll over T+1 steps matching a manual per-step roll), gather_chosen
(picks the chosen-action Q), masked_argmax (never returns an illegal index).
torch is seeded for determinism.
"""

from __future__ import annotations

import torch
from torch.nn import functional

from src.marl.learner._learner_helpers import (
    bptt_unroll,
    gather_chosen,
    masked_argmax,
    masked_huber,
)
from src.marl.nets.agent_net import RecurrentQNet
from src.marl.nets.dims import action_dim, hidden_dim, obs_dim

SEED = 7


def test_masked_huber_zeroes_masked_and_matches_torch() -> None:
    """Masked entries contribute 0; the value equals torch huber on unmasked."""
    torch.manual_seed(SEED)
    td = torch.tensor([[2.0, -3.0, 0.5], [10.0, 0.1, -7.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    delta = 1.0
    got = masked_huber(td, mask, delta)
    # Reference: torch huber on the unmasked entries only, mean over mask.sum().
    unmasked = td[mask.bool()]
    ref = (
        functional.huber_loss(unmasked, torch.zeros_like(unmasked), delta=delta, reduction="sum") / mask.sum()
    )
    assert torch.allclose(got, ref)


def test_masked_huber_no_grad_through_masked_entries() -> None:
    """Perturbing a masked TD entry leaves the loss (and its grad) unchanged."""
    td = torch.tensor([[2.0, 5.0, 0.5]], requires_grad=True)
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    loss = masked_huber(td, mask, 1.0)
    loss.backward()
    # The masked (index 1) entry must receive exactly zero gradient.
    assert td.grad is not None
    assert td.grad[0, 1] == 0.0
    assert td.grad[0, 0] != 0.0


def test_masked_huber_no_grad_flows_into_mask() -> None:
    """The mask is a stop-gradient multiplier (loss must not be a fn of it)."""
    td = torch.tensor([[2.0, -3.0]])
    mask = torch.tensor([[1.0, 1.0]], requires_grad=True)
    loss = masked_huber(td, mask, 1.0)
    assert not loss.requires_grad


def test_bptt_unroll_shape_and_matches_manual_roll(cfg: dict) -> None:
    """bptt_unroll output is [B,T+1,N,A] and equals a manual per-step roll."""
    torch.manual_seed(SEED)
    b, t, n = 2, 4, 2
    net = RecurrentQNet(cfg, "cop", n_agents=n)
    obs = torch.randn(b, t + 1, n, obs_dim(cfg))
    h0 = net.initial_hidden(b, n)
    rolled = bptt_unroll(net, obs, h0)
    assert rolled.shape == (b, t + 1, n, action_dim(cfg, "cop"))
    # Manual reference: carry hidden across the T+1 steps step-by-step.
    h = h0
    expected = []
    for step in range(t + 1):
        q_step, h = net.forward(obs[:, step], h)
        expected.append(q_step)
    expected_q = torch.stack(expected, dim=1)
    assert torch.allclose(rolled, expected_q)


def test_bptt_unroll_carries_hidden_not_reset(cfg: dict) -> None:
    """A later step depends on earlier obs (hidden is carried, not reset)."""
    torch.manual_seed(SEED)
    b, t, n = 1, 3, 1
    net = RecurrentQNet(cfg, "cop", n_agents=n)
    obs = torch.randn(b, t + 1, n, obs_dim(cfg))
    base = bptt_unroll(net, obs, net.initial_hidden(b, n))
    perturbed = obs.clone()
    perturbed[:, 0] = torch.randn(b, n, obs_dim(cfg))  # change ONLY step 0
    after = bptt_unroll(net, perturbed, net.initial_hidden(b, n))
    # The last step's q must move because step-0 fed the carried hidden.
    assert not torch.allclose(base[:, -1], after[:, -1])


def test_gather_chosen_picks_right_action_q() -> None:
    """gather_chosen selects q[..., a] for each [B,T,N] chosen action."""
    torch.manual_seed(SEED)
    b, t, n, a = 2, 3, 2, 4
    q = torch.randn(b, t, n, a)
    actions = torch.randint(0, a, (b, t, n))
    chosen = gather_chosen(q, actions)
    assert chosen.shape == (b, t, n)
    for bi in range(b):
        for ti in range(t):
            for ni in range(n):
                assert chosen[bi, ti, ni] == q[bi, ti, ni, actions[bi, ti, ni]]


def test_masked_argmax_never_returns_illegal() -> None:
    """The argmax index is always legal even when the true max is illegal."""
    # q's largest entry sits on an ILLEGAL action; masked_argmax must avoid it.
    q = torch.tensor([[[9.0, 1.0, 2.0, 3.0]]])
    legal = torch.tensor([[[False, True, True, True]]])
    idx = masked_argmax(q, legal)
    assert idx.shape == (1, 1)
    assert idx[0, 0] == 3  # the largest LEGAL action
    assert legal[0, 0, idx[0, 0]]


def test_masked_argmax_matches_plain_argmax_when_all_legal() -> None:
    """With every action legal, masked_argmax equals a plain argmax."""
    torch.manual_seed(SEED)
    q = torch.randn(3, 2, 5)
    legal = torch.ones(3, 2, 5, dtype=torch.bool)
    idx = masked_argmax(q, legal)
    assert torch.equal(idx, q.argmax(dim=-1))


def test_masked_argmax_does_not_mutate_input() -> None:
    """The additive -inf mask must not corrupt the caller's q tensor."""
    q = torch.tensor([[[5.0, 1.0]]])
    legal = torch.tensor([[[False, True]]])
    q_before = q.clone()
    masked_argmax(q, legal)
    assert torch.equal(q, q_before)


def test_helpers_are_pure_no_hidden_state(cfg: dict) -> None:
    """bptt_unroll twice on the same inputs is deterministic (no side state)."""
    torch.manual_seed(SEED)
    net = RecurrentQNet(cfg, "cop", n_agents=1)
    obs = torch.randn(2, 3, 1, obs_dim(cfg))
    h0 = net.initial_hidden(2, 1)
    assert torch.equal(bptt_unroll(net, obs, h0), bptt_unroll(net, obs, h0))
    assert hidden_dim(cfg) == h0.shape[-1]
