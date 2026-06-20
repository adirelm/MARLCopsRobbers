# P4b implementation spec — CTDE learners + N=2 active-mask (frozen, audit-approved)

> P4b = T4.2 (learners) + finalizing the N=2 agent-padding. TDD, ≤150 LOC/file, ruff(+D),
> no-hardcode (config via config_loader/dims). torch installed. Gate = FAST deterministic smokes
> (overfit-one-episode, masked-loss zero-grad, target-sync, ∂Q_tot/∂q_inactive=0, IQL-no-mixer,
> thief-mask-slice, import-boundary). The real convergence/sweep is P4d (long local run), NOT here.
> All contracts below are pinned from the codex+Claude P4b readiness (standard CTDE; no user trade-off).

## Pinned contracts
- **`active` mask is EPISODE-CONSTANT per slot** (NOT time-varying): the env has a fixed N per
  instance, so a whole episode is uniformly N=1 or N=2. A k-cop episode in an N-wide buffer has real
  cops in slots 0..k-1, slots k..N-1 zero-filled, `active[j] = (j < k)`. Distinct from `filled`
  (per-timestep episode length).
- **Team reward = active-MEAN over cops** of `reward[t]` (cops share an equal team reward in
  dec_pomdp, so mean == the shared value; NEVER sum — summing would double/triple it).
- **Learner owns the masking; mixers stay mask-unaware** (keep the P4a pure-net contract): the learner
  multiplies `q_i *= active` (broadcast) BEFORE the mixer, on BOTH the online and target paths. For
  VDN this is exact (sum ignores 0). For QMIX, a zeroed inactive `q_i` adds 0 through its `softplus(w1)`
  column; `b1`/`V(s)` are shared joint team-state terms (independent of N) — this is the standard
  PyMARL convention and is correct.
- **Thief** slices the length-`a_cop`(5) env mask to `a_thief`(4) before store/argmax; thief net head =
  `action_dim(cfg,"thief")=4`; thief buffer `n_actions=4`; thief is a SEPARATE single-agent Double-DQN,
  NOT in any mixer.
- **IQL** = per-agent independent Double-DQN target (eq4), NO mixer, NO global_state, a SINGLE AdamW
  group at `lr_agent` (no `lr_mixer`). No `iql_mixer.py` (fix the stale PRD:179 mention).

## src/marl/replay/episode_buffer.py  (ADD the active field — extends P3)
- Add `self._active = np.zeros((capacity, n_agents), dtype=bool)` (episode-constant per slot).
- `add_episode(episode, source_tag)`: require `episode["active"]` shape `(n_agents,)`; keep the
  existing agent-axis fail-loud guard (the producer now WIDENS to n_agents, so a correctly-widened
  episode passes; a missing/mis-shaped `active` raises). Write `self._active[i] = episode["active"]`.
- `sample(batch)`: add `"active": self._active[idx]` (shape `(B, n_agents)`).
- `_zero_slot`: `self._active[i] = False`.
- Tests: replace `test_agent_axis_mismatch_raises` with (a) a WIDENED N=2 episode carrying
  `active=[True, False]` round-trips (add→sample, active recovered), (b) a missing/mis-shaped `active`
  raises. Keep the rest green.

## src/marl/learner/  (split to stay ≤150 LOC/file)
### _learner_helpers.py
```python
def masked_huber(td_error: Tensor, mask: Tensor, delta: float) -> Tensor   # huber(td)*mask summed/mask.sum (filled-masked mean)
def bptt_unroll(net, obs: Tensor, h0: Tensor) -> Tensor                     # roll GRU over [B,T+1,N,obs_dim] -> q[B,T+1,N,A] (hidden carried)
def gather_chosen(q: Tensor, actions: Tensor) -> Tensor                     # q[B,T,N,A], actions[B,T,N] -> q_chosen[B,T,N]
def masked_argmax(q: Tensor, legal: Tensor) -> Tensor                       # additive -inf on ~legal, argmax over A
```
### learner_base.py :: QmixLearner  (base for VDN/QMIX cop learners)
```python
class QmixLearner:
    def __init__(self, cfg, role, n_agents, mixer: BaseMixer): ...   # online+target net (clone_target), online+target mixer (deepcopy frozen), AdamW 2 param-groups {net:lr_agent, mixer:lr_mixer}, gamma/huber_delta/target_update_interval from cfg
    def update(self, batch: dict) -> dict: ...                       # one BPTT update; returns {loss, grad_norm, q_tot}
    def _compute_target(self, batch, q_online_next, q_target_next) -> Tensor: ...  # centralized; overridden by IQL
```
- `update`: BPTT-unroll online + target nets over `obs` (T+1). `q_chosen = gather(q_online[:, :T], actions) * active`. `Q_tot = mixer(q_chosen[t], state[:, :T])` (reshape [B*T,N]+[B*T,state]→[B*T,1]→[B,T,1]).
  Double-DQN: `a* = masked_argmax(q_online[:,1:], next_legal_mask)`; `q_next = gather(q_target[:,1:], a*) * active`;
  `Q_tot_next = target_mixer(q_next[t], state[:,1:])`. `r_team = (reward * active).sum(-1)/active.sum(-1)` (active-mean over cops).
  `y = r_team + gamma*(1-done)*Q_tot_next.detach()`. `loss = masked_huber(Q_tot - y, filled, huber_delta)`.
  backward; clip to `algo.grad_clip_norm`; step; hard target sync (copy net+mixer params) every
  `target_update_interval` updates. Telemetry dict.
### learners.py :: CopLearner / ThiefLearner / IqlLearner
- `CopLearner(QmixLearner)`: role="cop", n_agents=cop count, mixer = QmixMixer or VdnMixer per
  `algo.mixer.type` / `algo.name`. (QMIX gated behind `algo.name=="qmix"`; VDN behind "vdn".)
- `IqlLearner(QmixLearner)`: role per-agent, NO mixer (skip the mixer path), single AdamW group
  (lr_agent only). Override `_compute_target` → per-agent eq4: `y_i = r_i + gamma*(1-done)*
  Q_i_target(obs_i[t+1], masked_argmax Q_i_online(obs_i[t+1]))`; per-agent loss masked by filled AND
  active. No global_state use.
- `ThiefLearner(IqlLearner)`: role="thief", n_agents=1, A=4 (slice the env mask to a_thief), separate
  net+buffer; it is the single-agent adversarial Double-DQN (IQL-style, no mixer).

## tests (FAST smokes — seed torch)
- tests/unit/test_learner.py: overfit-ONE-episode (repeat-update a tiny fixed episode → loss
  decreases toward ~0); masked-loss (a pad step `filled=0` contributes 0 gradient — perturb its
  target, grad unchanged); hard-target-sync (target frozen between syncs; copies at interval);
  Double-DQN target shape; telemetry keys present; **∂Q_tot/∂q_inactive == 0** (active=0 slot: zero
  grad + perturbing its q leaves Q_tot unchanged); **active-mean** reward (2 equal cop rewards → team
  == the shared value, not 2×); IQL single param-group (no lr_mixer; no mixer attr); ThiefLearner uses
  A=4 (mask sliced).
- tests/architecture/test_import_boundary.py (K3, scoped to existing modules): assert no module under
  src/marl/env or src/marl/nets (the runtime/exec path present at P4) imports learner/mixers/
  episode_buffer/GlobalState/services; note the MCP half is extended in P5. Give it teeth (a planted
  bad import would fail) without asserting nonexistent P5 modules.

## Docs to update (§1.4 spec-first)
- docs/TODO.md T4.2 + PLAN.md:462 episode tuple: add the `active` field; reframe the N=2 padding as the
  episode-constant active mask (NOT time-varying). docs/PRD.md:179: drop the `iql_mixer.py` mention
  (IQL branches in the learner). Note the producer (rollout, P4d) widens k-cop→N-wide + emits `active`.
