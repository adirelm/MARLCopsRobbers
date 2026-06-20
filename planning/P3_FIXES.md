# P3 validate-after fixes (Claude 10-lens adversarial; codex quota-limited this round)

> 38 real-defect findings triaged. The two P4-BREAKING replay bugs are the priority (latent in
> P3 since the smoke is N=1 + every episode terminates, but they corrupt P4's CTDE learner). TDD:
> strengthen/add the failing test, then fix. Keep ≤150 LOC/file, ruff(+D), no-hardcode. The tabular
> smoke is THROWAWAY (deleted in P4) but correctness still matters.

## OWNER A — replay + smoke cluster
FILES: src/marl/replay/{episode_buffer,tabular_smoke,_smoke_helpers,_smoke_solve}.py +
tests/unit/{test_episode_buffer,test_tabular_smoke}.py
- **A1 (obs/scalars T→T+1, L1 major — P4 contract):** allocate `_obs`/`_scalars` on the **T+1** axis
  (match `_global_state`), and in `add_episode` copy `episode["obs"][:t+1, :a]` / `scalars[:t+1, :a]`
  (mirror the global_state line). Update `_zero_slot`/`sample`/docstring. The producer (_pad_episode)
  already builds T+1 obs — this keeps the terminal next-obs so P4's recurrent target unrolls obs[1..T].
  TEST: a full-length episode's terminal next-obs frame survives add→sample (mark obs[real_t], assert
  recoverable); fix the shape assertions to (B,T+1,N,...) for obs/scalars.
- **A2 (N-mismatch silent broadcast, L2/L8/L9/L10 major — P4 corruption):** at the top of
  `add_episode`, **fail loud** if the episode agent axis != `self._n_agents`:
  `if episode["obs"].shape[1] != self._n_agents: raise ValueError(...)` (and the producer must supply
  the full N width). This kills the numpy size-1→size-2 broadcast that copies the real cop into the
  phantom slot. TEST: feeding an N=1 episode into an N=2 buffer raises (not silent broadcast). Add a
  one-line code comment + a TODO that the full N=2 agent-padding convention (zero-filled phantom +
  per-agent active mask) is a P4 task when num_cops=2 episodes actually exist.
- **A3 (tabular Q-update terminal bootstrap, L7):** `TabularQLearner.update(..., done: bool)` — at a
  terminal transition the target is `reward` only (no `gamma*max Q'`). Thread `done` from the rollout.
  TEST: update with done=True → Q moves toward r (no bootstrap); a separate test with a NONZERO next-Q
  asserts the `gamma*max_legal Q'` term is actually applied (currently both update tests use a fresh
  next-key with Q'=0, so the bootstrap is untested).
- **A4 (eval_optimal genuine gate, L7/L10 major):** the thief FLEES, so naive "min-Manhattan" is the
  wrong metric — keep the **pursuit-optimal** target but enforce it from **EVERY 2×2 start** (not only
  diagonal manhattan==2), and add a NEGATIVE control: an untrained/zeroed Q-table must FAIL the
  optimal gate (proves the gate is not vacuous). Run the smoke's optimal gate over **ALL
  `training.seeds`** (5), not seeds[:2]. (tabular is cheap — fast even at 5×4000.)
- **A5 (config episodes, L7):** spec said 2000 but config has 4000 (needed for all-seed convergence);
  config is source of truth — leave 4000; the doc-reconcile is OWNER C.

## OWNER B — data cluster
FILES: src/marl/data/{heuristics,bc_dataset,obs_encoder}.py +
tests/unit/{test_heuristics,test_bc_dataset,test_obs_encoder}.py
- **B1 (heuristic boxed-in crash, L5 major):** in `_maybe_explore` (and any `rng.choice(legal)` path),
  guard the empty-legal-moves case (fully boxed actor) — return a safe default (the BFS/greedy fallback
  or a no-op move index) instead of `rng.choice([])` which raises. TEST: a fully-walled actor with
  epsilon>0 + rng returns a legal/no-op action, no crash.
- **B2 (misleading cop-capture test, L5 minor):** `test_cop_captures_optimally_on_2x2` reconstructs the
  thief at a FIXED cell each iter (thief never moves) — rename/clarify it tests the **stationary-thief**
  optimal chase (which IS min-Manhattan), and do NOT claim it captures the fleeing thief (greedy chaser
  vs equal-speed evader need not capture — expected). Add a no-path edge test: cop walled off from the
  thief → `cop_expert` returns a legal fallback (covers heuristics.py:81/97/128).
- **B3 (BC label contract, L6 major):** add the missing assertion that the dataset LABEL is the
  expert's **greedy (epsilon=0)** action while collection uses `bc.epsilon` (mutation-proof: a label
  taken with epsilon>0 would fail). DOCUMENT in the bc_dataset docstring that labels come from a
  PRIVILEGED full-GlobalState expert while inputs are LOCAL obs (an inherent BC imitation gap when the
  thief is beyond view radius; mitigated by the aliasing-memory scalars; revisited in P4/T4.4).
- **B4 (BC empty-dataset guard, L6 nit):** `build_bc_dataset(..., n_pairs=0)` (or any empty obs_list)
  must raise a clear ValueError, not leak np.stack's "need at least one array".
- **B5 (encode_state N=2, L3 major):** add a 2-cop test (both cops on the cop plane). NOTE (code): two
  co-located cops collapse to a single 1.0 (idempotent assignment) — acceptable for P3; add a one-line
  comment that a P4 multi-cop centralized state may need a count/per-cop plane.
- **B6 (encode_obs_batch order-preserving + channels-first, L4):** the current "pure stack" test uses N
  identical Observations (can't detect reordering) and a (3,5,5) shape that can't distinguish
  channels-first from channels-last (C==W==5). Use DISTINCT per-item values + assert ordering is
  preserved and the layout is channels-first (B,C,5,5) via a non-square-channel probe or explicit index.

## OWNER C — sdk + integration + docs
FILES: src/sdk/{sdk,_helpers}.py + tests/unit/test_sdk.py + tests/integration/test_p3_pipeline.py +
docs/schema/subgame.schema.json + docs (PRD/PLAN/TODO/planning/P3_IMPL_SPEC.md)
- **C1 (subgame validator stub, L9 major):** the self-contained `_validate` in test_p3_pipeline enforces
  required/enum/type but NOT `additionalProperties:false` — strengthen it to reject extra keys, AND add
  a NEGATIVE test (a record with a bad enum / missing field / extra key is REJECTED). Currently it only
  runs on the known-good record, so it can't catch a schema regression.
- **C2 (god-view leak assertion, L9 minor):** the `not isinstance(god, GlobalState)` assert is
  tautological (god is built as a dict). Strengthen to the recursive no-leak scan used by the env leak
  test (no GlobalState field/instance anywhere in the render_state dict) + assert JSON-serializable.
- **C3 (SDK docstring, L8 minor):** `run_p3_smoke`'s docstring claims it writes the sub-game JSON
  "validating vs subgame.schema.json" but no validation occurs — either make the docstring accurate OR
  call a lightweight validator; pick the accurate-docstring fix (no new src validator at P3).
- **C4 (doc reconciles):** in planning/P3_IMPL_SPEC.md fix `episodes: 2000`→`4000` and the
  `build_bc_dataset` return to a **4-tuple** (obs, scalars, actions, manifest) to match the test; fix the
  encode_state "centered/padded"→"top-left anchored, off-board masked" wording. Verify PRD/PLAN/TODO
  have NO remaining "BC pretrain val-acc≥0.9 / train→eval AT P3" milestone (FR-OLoRA mentioning BC
  val-acc as P4 evidence is fine; a P3 *milestone row* claiming it is not) — fix any that remain. Add a
  T3.1 note: replay N=2 agent-padding convention is finalized in P4.
- **C5 (optimality-gate wording, with A4):** reconcile the T3.3 DoD/PRD/PLAN "2×2 reaches optimal"
  wording to **pursuit-optimal capture from every 2×2 start over all training.seeds** (the thief flees,
  so naive min-Manhattan is wrong); match A4's implementation.

## NOT FIXED (validated NOT defects / deferred to P4)
- "cop_expert never captures the fleeing thief_expert" — EXPECTED for an equal-speed greedy pursuer vs
  optimal evader; not a bug (B2 fixes the misleading test + A4 fixes the eval metric).
- Full N=2 agent-axis padding + per-agent active mask — deferred to P4 (A2 adds the fail-loud guard now).
- SDK double-train of the throwaway learner (L8 minor) — acceptable for a throwaway P3 smoke; optional.
