# P4b validate-after fixes (codex + Claude; the TD/Double-DQN math is CORRECT — proven)

> Both models verified the training math end-to-end (online-argmax/target-value decoupling, y
> detached, BPTT hidden carry, gamma*(1-done) terminal, active-mean reward, active-mask on both
> paths). Fixes = ~5 real-but-minor code defects + hardening the TEST blind spots so the
> convergence-critical invariants are pinned (mutations currently survive). TDD, ≤150 LOC, ruff(+D).

## OWNER A — buffer + helpers
FILES: src/marl/replay/episode_buffer.py + src/marl/learner/_learner_helpers.py +
tests/unit/test_episode_buffer_active.py + tests/unit/test_learner_helpers.py
- **A1 (HIGH — incomplete N-broadcast guard, codex):** add_episode validates ONLY obs.shape[1]; an
  N=1 `scalars`/`actions`/`reward`/`next_legal_mask`/`active` can still numpy-broadcast into an N=2
  slot, duplicating cop 0. Validate the agent-axis of EVERY per-agent field == n_agents (fail loud).
  Test: an episode with obs N=2 but actions N=1 raises (no silent broadcast).
- **A2 (major edge — masked_huber NaN):** `keep.sum()` denominator is unclamped → a fully-masked
  batch (filled.sum()==0) yields NaN. Clamp: `/ keep.sum().clamp(min=1)`. Test: masked_huber on an
  all-zero mask returns 0.0 (not NaN).

## OWNER B — learner_base + its tests
FILES: src/marl/learner/learner_base.py + tests/unit/test_learner.py
- **B1 (honor double_q, Claude L1):** `algo.double_q` is read but NEVER used (always Double-DQN). HONOR
  it: in the target, a* comes from the ONLINE next-q when `double_q` else from the TARGET next-q
  (vanilla DQN). Keeps the config truthful + enables the §7.2 DQN-vs-DoubleDQN ablation.
- **B2 (skip empty param-group, Claude L7):** `_param_groups` appends a mixer group whenever
  `self.mixer is not None`, but VdnMixer has NO params → an empty AdamW group. Only add the mixer group
  if it has trainable params (`any(p.requires_grad ...)`).
- **B3 (TEST teeth — the convergence-critical gaps):** add fast mutation-proof tests:
  (a) Double-DQN DECOUPLING — with online != target, assert a* is taken from the ONLINE next-q and the
  value from the TARGET next-q (factor a tiny `_ddqn_next_value` seam or spy the helper inputs); a
  swap must fail. (b) `(1-done)` TERMINAL masking — a terminal-step target == r_team (no bootstrap); a
  variant without (1-done) differs. (c) TARGET-PATH active-mask via the PRIMARY QMIX mixer (not VDN):
  ∂Q_tot/∂q_inactive == 0 (gradient, not just value). (d) masked-pad test asserts the pad step's
  `.grad` is exactly 0 (not merely loss-equality).

## OWNER C — learners (Cop/Iql/Thief) + its tests
FILES: src/marl/learner/learners.py + tests/unit/test_learners.py
- **C1 (mixer ablation toggle, codex):** CopLearner selects the mixer from `algo.name` only, ignoring
  `algo.mixer.type` → the documented VDN/QMIX ablation toggle is inert. Select the mixer from
  `algo.mixer.type` (qmix→QmixMixer, vdn→VdnMixer). Test: mixer.type drives the mixer class.
- **C2 (IQL no-global-state, codex):** IQL/Thief `update` calls the generic tensor builder that
  REQUIRES `global_state`, violating the "no global_state" IQL contract (a genuine no-state batch
  fails). Make global_state optional on the IQL path (don't require/materialize it). Test: an IQL
  update on a batch WITHOUT global_state succeeds.
- **C3 (IQL telemetry active-mask, Claude L3):** IQL `q_tot` telemetry averages q_chosen over ALL N
  incl inactive phantom slots — mask by active.
- **C4 (Thief slices actions too, Claude L4/L5):** ThiefLearner slices the legality MASK to a_thief but
  not `actions`; defensively slice/validate actions to <a_thief (the producer guarantees it, but make
  it loud). Fix the self-contradicting thief test (assert the LEARNER slices, don't pre-slice with %4).
- **C5 (IQL/Thief mutation-proof tests, Claude L8):** the IQL path is the weakest-tested high-risk
  code — add tests that would CATCH: IQL using a mixer, IQL using global_state, the per-agent eq4
  target wrong, the thief width != 4, the thief mask not sliced. Each mutation must fail a test.

## NOT FIXED (validated NOT defects / deferred)
- BPTT lens: CLEAN (proven). The unused `hidden_seed` buffer field — leave (reserved for P4d warm-start;
  add a one-line "reserved, h0=zeros until P4d" comment if convenient).
- Target-sync lens: CLEAN (proven — copies net+mixer, frozen between).
