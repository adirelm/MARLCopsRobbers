# A6 plan — VALIDATED fix list (post 27-agent evidence-bound validation)

> Round-2 (90-agent) review proposed 27 FIX items (`CANONICAL_FIXES_2.md`). A 27-agent
> validation pass (each quoting real file:line, defaulting to false-positive if unprovable)
> returned **16 confirmed · 9 partial · 2 false-positive**. THIS file is the authoritative,
> calibrated action list. Apply EXACTLY these; do not re-add the dropped over-reach. Keep
> PRD/PLAN/TODO/config mutually consistent. docs/adr/ is currently EMPTY — ADRs referenced by
> number (ADR-0005/0006/0007/0008/0011/0012/0013) live only in the PLAN index table; where a
> fix says "+ADR", also add the rationale to the PLAN ADR table (do NOT block on creating ADR
> files — that is a separate P0 task).

## DROPPED — false positives (make NO change except the tiny noted clarifications)
- **#8 peer-MCP/Gmail egress bypass — FALSE.** Plan already routes ALL egress through
  `ApiGatekeeper.execute` (PLAN:137,505,539; PRD:352-353; TODO:232,346; rate_limits.json:3).
  ONLY tiny real gap: state behavior when the FIFO overflow queue itself hits `max_queue:256`
  → "reject-with-error + log" (one line in PRD FR-API-1 AC + rate_limits.json note).
- **#17 self-grade in tracked repo — FALSE.** Numeric self-grade already routed to the
  git-ignored cover sheet + git-ignored `instructions/` audit doc (TODO:418; .gitignore:44-46).
  ONLY cosmetic: append "(to git-ignored cover sheet)" to TODO:403/418/419. Do not remove the gate.

## CONFIG (config/config.yaml + rate_limits.json + pyproject.toml)
- **#3a softplus:** line 130 commit to **softplus** (not "abs/softplus"); fix the PLAN:197
  "abs weights" + TODO:162 "abs/softplus" to all say softplus. (Resolves an internal contradiction.)
- **#3c DROP `include_opponent_utility`:** delete config.yaml:137 + the ADR-0007 (PLAN:503) +
  PRD OD4:480 references. The 4×4 2-cop scenario already supplies the K1 non-trivial-mixer
  evidence, so the lever is unneeded and removes the latent ADR-0006 boundary risk.
- **#5 conv→MLP:** delete `nets.conv_channels` (line 148); encoder is a flat MLP trunk
  (`encoder_hidden`, OLoRA-wrappable Linear) per L10 §5/eq8. (See PLAN/TODO edits.)
- **#4a Φ(terminal)=0:** add `reward.phi_terminal_zero: true` with a comment that Ng-1999
  invariance needs Φ=0 on absorbing states (capture Φ=0 automatically; timeout + swap-capture
  are the real edge cases).
- **#4b sensitivity param:** add a comment that `reward.distance_weight` is policy-invariant
  (Ng-1999) → NOT a valid §9 sweep param; the swept param is `env.view_radius_by_grid`.
- **#7 self-play:** (i) repurpose `window_k` to mean ONLY the frozen-opponent window length;
  delete the "update_ratio:" wording from its comment (197-198). (ii) add `episodes_per_round`
  and state the identity `rounds(50) × episodes_per_round(100) = episodes_per_stage(5000)`.
  (iii) pick **alternating best-response** as THE regime; scope `update_ratio` to within-round
  learner cadence (or drop). (iv) add a per-round ε note + a one-line pool sampling/snapshot rule.
- **#9 DROP `cloud.rate_limits`:** delete config.yaml lines 285-289; `config/rate_limits.json`
  is the single source (superset: also has prefect_deploy/overflow/max_queue). Redirect any
  PLAN/PRD pointer to rate_limits.json.
- **#10 REPORT_RECIPIENT_OVERRIDE:** remove it from `.env-example:34` + the config.yaml:311
  comment; recipient is unconditionally `gmail.to`. (Override, if any, is test-injection only.)
- **#11b session_id:** add a comment that `session_id` is a required request field (GRU keying +
  (session_id,tick) idempotency).
- **#12 single-worker:** add `cloud.workers: 1` with a comment (server-side GRU z_t requires one
  worker; free single-instance tier already provides this; client-carried-hidden-state is the
  fallback).
- **#14 pyjwt[crypto]:** pyproject.toml:26 `"pyjwt>=2.9"` → `"pyjwt[crypto]>=2.9"` (RS256 needs it).
- **#8 / overflow:** rate_limits.json — add a note that on `max_queue` overflow the gatekeeper
  rejects-with-error + logs (no crash).

## PRD (docs/PRD.md)
- **#1:** FR-ENV-1 AC (PRD:126) replace literal `|N|=2` with `|N|==env.num_cops` (=1 graded, 2 on
  4×4) + note the full POSG game has 2 role-agents {cop,thief}. A-row (PRD:78) redefine **A = the
  cop-team joint action** (|A_cop|=5,|A_thief|=4 belong to T/POSG, not Dec-POMDP A). S-row (PRD:77)
  `cop_pos: list[(r,c)]`; R shared over the cop team; Φ = −min_i Manhattan(cop_i,thief)/d_max (PRD:26).
- **#2:** FR-ALG-4 (PRD:164) stop calling the mixer the ONLY variation point — IQL also drops
  mixer+global-state and uses an independent per-agent target (learner branches). PRD:94 relabel
  the IQL target: the target-net/Double-DQN form is an implementation refinement OF eq2/eq4's
  independent target, not byte-identical (PDFs show plain `y_i=r_i+γ max Q_i`).
- **#3b global-state pad:** add an FR + AC that the centralized state is encoded to a
  **stage-invariant 77-float** footprint (out-of-board masked) for ALL curriculum stages, mirroring
  the obs pad-to-5×5. **#3d:** FR-ALG-1 AC (PRD:156) IGM/monotonicity test runs on an explicit
  **N=2 mixer fixture** (+ optional non-monotone negative control that must fail).
- **#4a:** add Φ(terminal)=0 to FR-ENV-6 (+ a unit-test AC on the terminal transition).
  **#4b:** FR-RES-1 (PRD:355) remove `reward.distance_weight`; pin `env.view_radius_by_grid`.
- **#6a/#6c:** the exec-time action selection is **additive −inf masking** (`masked_fill`), NOT
  multiplicative `mask·Q` (wrong for negative Q); store the **next-state legal mask** in the replay
  tuple (PRD:464) for the Double-DQN target. (#6b heads-already-per-role is a NON-issue — at most a
  one-line clarification that "shared net" = shared architecture, per-role heads cop 5/thief 4.)
- **#11:** canonical MCP tool set = `request_move, reveal_location, query_opponent, new_sub_game,
  send_final_report` (PRD:218 already correct — TODO is the inconsistent one). Add `session_id` to
  the request schema inventory; give `query_opponent` a real schema OR fold it into reveal_location.
- **#19:** FR-NFR-11 / P11 gate (PRD:455) enumerate the four `docs/prd/PRD-{ENV,CTDE,MCP,REPORT}.md`
  in `test_required_docs_present`.
- **#20 §16:** add a short requirement: components document Input/Output/Setup + `_validate_input()
  →TypeError` for the MCP tool handlers + report builder (DI + `_validate_config` already specified).
- **#21 §13:** optional — extend FR-QUAL-1's inline ISO example list from 3 to all 8 characteristics.
- **#24a:** FR/README usage must point at real `scripts/` entry points (no `src.cli`).
- **#26:** carve out F3/F4 in the figures claim (PRD:335 FR-ANL-7, PRD:112 K5): F1/F2/F5/F6 plot
  from runs/*.jsonl; F3 (GUI) + F4 (MCP-comms) are deterministically CAPTURED, not plotted.
- **#8 overflow:** FR-API-1 AC — gatekeeper rejects-with-error + logs on `max_queue` overflow.

## PLAN (docs/PLAN.md)
- **#2:** ADR-0008 (PLAN:504) + the PLAN:136 seam bullet reframed (mixer not the ONLY variation
  point). DROP the fictional `iql_mixer.py` "identity module" (PLAN:195) — IQL branches in the
  learner. State `IqlLearner` must NOT keep the QMIX 2-param-group (lr_mixer) optimizer.
- **#3a/#3b/#3c:** PLAN:197 softplus; add the global-state pad-to-77 to the encoder/critic section;
  remove include_opponent_utility from ADR-0007.
- **#5 conv→MLP:** reconcile PLAN:187 (encoder.py "MLP trunk") with PLAN:192/TODO:104 ("conv+vec
  encoder") to ONE definition: flat obs → Linear trunk (encoder_hidden, OLoRA-wrapped) → GRUCell →
  role-conditioned Q-head, per L10 eq8. Remove conv from PLAN:358 config enumeration.
- **#11:** sweep the start_sub_game/end_sub_game (4+4 occurrences) to the canonical new_sub_game;
  add session_id to schemas.py (PLAN:221); resolve query_opponent (schema or fold).
- **#12:** state single-worker as the chosen resolution in the §7.2 cloud-deploy section + the
  client-carried-hidden-state fallback; add the rationale to the ADR-0011 row.
- **#13 cloud model delivery:** commit ONE small actor-only inference checkpoint (GRU+MLP
  state_dict + shape sidecar, sub-MB) under a non-ignored path; set `MODEL_PATH` to it; state the
  mechanism in §7.2 + the push-to-deploy diagram (commit only actor weights — no mixer/critic/replay).
- **#14:** note the jti deny-list is a CUSTOM verifier wrapper around FastMCP JWTVerifier
  (which validates sig/exp/aud/scopes only) — one clarity line (not a claimed-builtin correction).
- **#24b/#24c:** collapse the duplicate action/mask — ONE source `src/marl/env/actions.py`;
  `src/mcp/actions.py` imports/re-exports it (or is deleted). Reconcile `referee.py` placement with
  the `test_mcp_servers_have_no_logic` gate: either move it out of `src/mcp/` (it is "the
  environment" → `env/` or `services/`) or explicitly carve it out of the gate's scope.
- **#25 (scoped):** surface the 100k-episode envelope (5000 × 4 stages × 5 seeds, already in PRD
  FR-COST-1) ONE sentence earlier in PLAN §1; add ONE consolidated defer-order line ("if behind,
  drop in order: P8 cloud → OLoRA ablation arm → P-bonus") referencing existing R2/R12/R16/P-bonus.
  Reframe the R16 "role split A/B" as **solo work-streams** (this is a solo submission). DO NOT add
  a day-level Gantt or formal CPM critical path (the "50-day window" is fabricated; rubric doesn't
  require it).
- **#26:** fix the stale parenthetical PLAN:292 "runs/*.jsonl (git-ignored)" → "(tracked: small
  per-episode telemetry needed for figure regen)"; reword the blanket figures claim at PLAN:44.

## TODO (docs/TODO.md)
- **#7:** self-play tasks (T4.6) match the single alternating-best-response regime + the budget
  identity; #3a softplus at TODO:162.
- **#11:** TODO:220/304/432 sweep start_sub_game/end_sub_game → new_sub_game; add session_id +
  query_opponent to the schema task (TODO:208).
- **#16 PROMPTS.md:** add a task to create + maintain `docs/shared/PROMPTS.md` (the prompt log,
  §1.4 evidence) AND add it to the `test_required_docs_present` enumeration.
- **#17:** cosmetic — append "(to git-ignored cover sheet)" at TODO:403/418/419.
- **#18 docstring gate:** add Ruff `D` to the lint select (pyproject) + `D` per-file-ignore for
  tests; add a hard-gate-checklist row (TODO:461) + §9-map row (PLAN:606) naming the enforcement;
  give every package `__init__.py` a one-line module docstring.
- **#19:** tighten T0.5 DoD (TODO:67) to enumerate the four PRD-ENV/CTDE/MCP/REPORT in
  test_required_docs_present (do NOT split into 4 tasks).
- **#23 check_pii:** add ONE line to T0.6 (TODO:70) + PLAN:256: check_pii.py MUST NOT enumerate
  literal deny-list tokens — load patterns from the git-ignored players.local.yaml/secrets/ or
  derive generically (regex shapes). (A4 lesson.)
- **#27 build-order:** move T1.4 (Mixer ABC + IQL/VDN mixers) and T1.5 (RecurrentQNet) from P1 to
  P4 so TODO P1 = theory/env/reward/scorer only (matches PLAN P1/P4, BRIEF §6 step 4). Keep
  reward.py (uses gamma const) in P1. Do NOT touch PLAN P1/P4 (already correct). Skip the BC/GUI
  re-phasing sub-clause (already correct: BC train = T4.4 in P4, GUI build = P7).
- **#20 §16 + #21 §13 + #26 figures:** mirror the PRD edits in the matching TODO tasks/DoD.

## NEW FILES
- **#15 CLAUDE.md:** create `CLAUDE.md` at the A6 root — adapt A1/A5: §1.4 human-architect/
  AI-implementer contract table + canonical values (grid 5×5/25 moves/6 games/scoring 20-10-5-5,
  QMIX-primary + VDN + IQL baseline, seeds, group adrl-001, version 1.0.0) + hard constraints
  (≤150 LOC, TDD ≥85%, OOP, no-hardcode with the §4 config-vs-local-UI rule, ruff 0, uv-only) +
  the PII deny-list rule. Resolves ~21 dangling "CLAUDE.md §1.4/§4" citations.
- **#16 docs/shared/PROMPTS.md:** create the prompt-log stub (matches the PLAN:289 tree mention).
- **#22 LICENSE:** replace line 1 with `Copyright (c) 2026 adrl-001 — MARLCopsRobbers` (drop the
  personal owner slug + its github URL — the owner slug is a self-declared deny-list
  token kept only in git-ignored players.local.yaml).
- **#13 inference checkpoint path:** add the `.gitignore` allow-list exception for the ONE committed
  actor-only inference checkpoint (place after the `results/**/*.pt` rule, or a `deploy/model/` path).
