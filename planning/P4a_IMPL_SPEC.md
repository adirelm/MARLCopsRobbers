# P4a implementation spec — nets + mixers + QMIX (frozen, readiness-audit-approved)

> P4 is split into P4a–P4d (user-signed-off). **P4a = pure-tensor nets + mixers** (T4.0b, T4.0a, T4.1)
> — NO training loop (that's P4b). TDD, ≤150 LOC/file, ruff(+D), no-hardcode (all dims from config).
> torch is installed. Gate = forward-shape + hidden-carry + online/target copy + VDN-sum + QMIX
> softplus-monotonicity (∂Q_tot/∂Q_i ≥ 0) + IGM (eq5) + a non-monotone NEGATIVE control. All FAST/
> deterministic (seed torch). The full training/convergence is P4b+/the long local run, NOT here.

## Pinned decisions (from the P4 readiness audit)
- **obs_dim is config-derived, not hardcoded:** `obs_dim = obs_channels*(2*view_radius_max+1)^2 +
  obs_scalars = 5*25+6 = 131`. Add a `src/marl/nets/dims.py` helper.
- **IQL has NO mixer** (resolves the TODO↔PLAN conflict, P1 fix #2): P4a builds `base_mixer` (ABC) +
  `vdn_mixer` only. **Do NOT create `iql_mixer.py`** — the IQL path branches in the learner (P4b).
- **QMIX weights = softplus** (commits the abs/softplus contradiction, #3a). V(s) is sign-free.
- **encode_state stays the 77-float binary team-occupancy** (no per-cop plane): the QMIX mixer already
  receives per-agent Q_i (which distinguish cops), so the global state need not encode cop identity.
- **N=2 active-mask** is a P4b (learner/buffer) concern: P4a mixers operate on `q_agents[B,n]` as given;
  the learner will zero inactive cops' Q_i before the mixer (a zeroed Q_i contributes 0 to VDN sum and
  to the QMIX first layer `Σ w1_i·q_i`). The QMIX N=2 fixture test uses both-active.

## src/marl/nets/dims.py  (config-derived dims — no hardcode)
```python
def obs_dim(cfg: dict) -> int           # obs_channels*(2*view_radius_max+1)**2 + obs_scalars (=131)
def hidden_dim(cfg: dict) -> int        # nets.hidden_dim (=64)
def action_dim(cfg: dict, role: str) -> int   # env.actions.a_cop (5) for "cop", a_thief (4) for "thief"
def state_dim(cfg: dict) -> int         # 3*(2*view_radius_max+1)**2 + 2 (=77) — matches encode_state
```

## src/marl/nets/agent_net.py  (T4.0b — RecurrentQNet / GRUQNet)
```python
class RecurrentQNet(nn.Module):
    def __init__(self, cfg: dict, role: str, n_agents: int): ...
    def forward(self, obs: Tensor, h: Tensor) -> tuple[Tensor, Tensor]   # obs[B,n,obs_dim], h[B,n,H] -> q[B,n,A], h'[B,n,H]
    def initial_hidden(self, batch: int, n: int) -> Tensor               # zeros [B,n,H]
```
- Architecture (eq8): flatten `obs[B,n,obs_dim]`; prepend an **agent-id one-hot** of width `n_agents`
  (so same-role agents share weights but are distinguishable) → encoder input dim `obs_dim + n_agents`;
  **flat-MLP encoder** = `nets.encoder_hidden` Linear layers (ReLU), each an `nn.Linear` (OLoRA-wrappable
  in P4c) → `nn.GRUCell(enc_out, hidden_dim)` applied per agent over the merged `(B*n)` batch → a plain
  **role Q-head** `nn.Linear(hidden_dim, action_dim(cfg, role))`. All dims via dims.py — NO literals.
- `online`/`target`: provide `def clone_target(self) -> RecurrentQNet` (deep copy, `requires_grad_(False)`)
  and `def soft_update`/`hard_update` is the LEARNER's job (P4b) — P4a only needs the deep-copy clone.
- Tests (seed torch): `forward` returns `q.shape==(B,n,A)` + `h'.shape==(B,n,H)` for cop (A=5) AND thief
  (A=4); hidden carries (h' != h for nonzero input); 2-agent (n=2) weight-sharing with distinct agent-id
  one-hots gives generally-different q per agent; `clone_target` params equal + detached (requires_grad False).

## src/marl/mixers/base_mixer.py  (T4.0a — the swappable seam)
```python
class BaseMixer(nn.Module, ABC):
    @abstractmethod
    def forward(self, q_agents: Tensor, state: Tensor) -> Tensor   # q_agents[B,n], state[B,state_dim] -> q_tot[B,1]
```

## src/marl/mixers/vdn_mixer.py  (T4.0a)
```python
class VdnMixer(BaseMixer):   # eq6: Q_tot = Σ_i Q_i (state ignored)
    def forward(self, q_agents, state): return q_agents.sum(dim=1, keepdim=True)
```
- Test: `VdnMixer` output == `q_agents.sum(dim=1)` exactly for an N=2 fixture; ignores `state`.

## src/marl/mixers/qmix_mixer.py  (T4.1 — monotone hypernetwork)
```python
class QmixMixer(BaseMixer):
    def __init__(self, cfg: dict, n_agents: int): ...   # embed_dim=algo.mixer.embed_dim(32); state_dim from dims
    def forward(self, q_agents: Tensor, state: Tensor) -> Tensor   # [B,n],[B,77] -> [B,1]
```
- Standard QMIX: hypernets map `state` → `w1 = softplus(hyper_w1(state)).view(B,n,embed)`,
  `b1 = hyper_b1(state)`, `w2 = softplus(hyper_w2(state)).view(B,embed,1)`, `v = V(state)`;
  `hidden = elu(bmm(q_agents[B,1,n], w1) + b1)`; `q_tot = bmm(hidden, w2) + v` → `[B,1]`. **softplus** on
  w1/w2 (≥0 → monotone); V(s) sign-free. Generic over N. Gated behind `algo.name=="qmix"` at the call
  site (not inside the module). embed_dim/state_dim/n via dims/config — no hardcode.
- Tests (seed torch, N=2 fixture): **monotonicity** — perturbing one `Q_i` UP never decreases `q_tot`
  (numeric: `q_tot(q+δe_i) ≥ q_tot(q)` for δ>0, all i); **IGM (eq5)** — the decentralized greedy joint
  (argmax each `Q_i`) equals the joint maximizing the mixer output over a small enumerated action set;
  **NEGATIVE control** — a deliberately non-monotone mixer (e.g. raw signed weights, no softplus) FAILS
  the monotonicity assertion (proves the test has teeth); output shape `[B,1]`.

## Notes
- No iql_mixer.py (IQL = learner branch, P4b). No training loop / optimizer / replay use in P4a.
- Keep each file ≤150 LOC; agent_net may split a tiny `_encoder` helper if needed.
- After P4a: docs/TODO T4.0a wording — drop the `iql_mixer.py` mention (IQL branches in the learner).
