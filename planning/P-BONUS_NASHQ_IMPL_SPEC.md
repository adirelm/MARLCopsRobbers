# P-BONUS — Minimax-Q / Nash-Q tabular baseline (L11 §5 self-challenge)

> **Status: PLANNED — DEFERRED.** Do this when Claude usage limits reset (next working session).
> v1.0.0 already shipped + converged (16-round audit → 0); this is a **post-1.0, §7.2 enrichment**,
> NOT required by ex06 (A6 is L10-based). It realizes Lecture 11's §5 self-challenge and turns our
> *prose* justification (THEORY §3: "self-play over Minimax-Q/Nash-Q/MADDPG") into an **empirical arm**.

## Goal
A **tabular Minimax-Q** baseline (zero-sum; optional Nash-Q general-sum extension) for the cop↔thief
game on a small grid, as an empirical §7.2 comparison to our deep **self-play best-response**. Question:
does the equilibrium learner converge to the game value, and how does our deep self-play compare?

## Hard gates (unchanged — this is graded code)
≤150 LOC/file (tests too); TDD RED→GREEN→REFACTOR, ≥85% cov; OOP/no-dup; **config-driven (no hardcode)**;
ruff(+D) 0; `ruff format` clean; **uv-only**; single `MarlSDK` entry (runner imports only `src.sdk`);
honest reporting. After the work, run a `/goal` convergence pass (the doc→test gate
`test_prd_cited_test_files_resolve` must stay green; regen `experiment_manifest.json` → keep zero-drift).

## New dependency
`uv add scipy` — the maximin LP uses `scipy.optimize.linprog` (L11 p.31-32). Commit the updated `uv.lock`
+ `pyproject.toml`. (Alternative if scipy is unwanted: a tiny pure-NumPy 2-action LP — but scipy matches
L11 and generalizes; prefer scipy.)

## Tasks (bite-sized; RED test first, commit each)
- **T1 — config.** Add a `minimax_q` block to `config/config.yaml` (`grid: 3`, `alpha`, `gamma`,
  `episodes`, reuse `training.seeds`). If it's a new top-level section, add it to `config_loader`
  `REQUIRED_SECTIONS` + `test_config_loader` `EXPECTED_SECTIONS`.
- **T2 — the minimax LP** `src/marl/baselines/minimax_lp.py` (≤150 LOC): `solve_minimax_value(q_matrix)
  -> (mixed_pi, game_value_V)` via `linprog` (maximin: max V s.t. `Σ_i π_i Q(i,j) ≥ V ∀j`, `Σπ=1`, `π≥0`).
  **RED first** `tests/unit/test_minimax_lp.py`: the L11 worked example `Q=[[-2,4],[3,-1]]` → `p≈0.4`,
  `V≈1.0` (eq 2.3); + a dominated-action case; + a pure saddle-point case.
- **T3 — tabular Minimax-Q** `src/marl/baselines/minimax_q.py` (≤150 LOC): `MinimaxQ` (Q-table
  `{state: (n_a1,n_a2) array}`); update eq 2.1 `Q(s,a1,a2) ← (1-α)Q + α[r + γ·minimax_value(Q[s'])]`
  using T2. RED tests: 1-step update; convergence on a tiny 2-state zero-sum game to its analytic value.
- **T4 — tabular env adapter** (3×3): reuse `CopsRobbersEnv` at 3×3 with a flattened (cop,thief)-position
  state key; `src/marl/baselines/tabular_env.py` (≤150 LOC) or a helper. Reuse, don't duplicate.
- **T5 — SDK entry.** `MarlSDK.run_minimax_q_baseline(...)` (single-entry); `scripts/train_minimax_q.py`
  thin wrapper (mirror `scripts/train_iql_baseline.py`).
- **T6 — figure (F7).** Minimax-Q game-value-vs-episode curve + convergence-to-Nash-value line →
  `results/figures/` via a `make_figures`-style generator; add to the figure manifest; regen → zero-drift.
- **T7 — §7.2 empirical write-up.** Extend `docs/ANALYSIS.md` + README §7.2 with the DATA (tabular
  Minimax-Q converges to value X on 3×3; deep self-play reaches Y; discussion), upgrading the THEORY §3
  prose to an evidenced comparison. Honest.
- **T8 — docs sync.** README §7.3 manifest (F7); PRD (a bonus FR-ANL + its `Evidence:` test pointer that
  RESOLVES); link THEORY §3 → the empirical result. Keep PRD-pointer + ADR gates green.
- **T9 — gates + CI + /goal pass.** ruff/format/sizes/pytest ≥85%; new files ≤150 LOC; commit per task.

## Scope decision (next session)
- **Minimum (sufficient for the L11 §5 challenge):** Minimax-Q (zero-sum) + LP + 3×3 comparison + §7.2
  data. Cop↔thief is ~zero-sum, so Minimax-Q is the principled fit and is provably LP-solvable.
- **Optional:** Nash-Q (general-sum, eq 2.4) as an extension — heavier (stage-game Nash solver, PPAD-hard
  in general); only if time. Honest framing: zero-sum Minimax-Q suffices; Nash-Q is the generalization.

## Acceptance
L11 §5 realized (working tabular Minimax-Q on a small grid, learning curve + convergence-to-game-value,
honest write-up); §7.2 has an empirical equilibrium-vs-self-play comparison; all hard gates green;
`/goal` pass clean.

---

## Other deferred / next items (USER-GATED — never autonomous)
1. **§3.5 report email — HARD GATE.** Never sent without the user's explicit "send" + `.env` Gmail creds
   (memory: `do-not-send-report-until-explicit-go`). The report is assembled/validated/dry-run-tested only.
2. **Cloud live-deploy.** Built + tested, NOT live-run (account/cred-gated: Prefect Horizon / FastMCP
   Cloud). Run the P8 runbook (`deploy/runbook.md`) when the user provides accounts.
3. **Add `rmisegal` as read-only collaborator** on the GitHub repo (memory: `lecturer_github_handle`) —
   a GitHub-UI action for the user.
4. **Methodology fold-in (optional).** Add the convergence meta-lessons (close finding-classes via
   exhaustive sweep + permanent gate; sweep adjacent honesty siblings; recompute printed numbers; audit
   validated-fixes) into `instructions/review_methodology/` (memory: `audit-loop-close-classes`).
