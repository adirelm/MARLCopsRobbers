# ANALYSIS — empirical notes backing the README §7 paper

This file collects the *honest* quantitative caveats the README §7 academic
write-up cites. Each section is reproducible from code/config; numbers here must
never drift from `config/config.yaml` or the seeded generators.

---

## §0 BC val-acc: random floors and the per-grid Bayes ceiling

The behavior-cloning gate (`bc.val_acc_gate_by_grid`) is **per-grid and modest by
design**, not a bug. This section derives why a "≥ 0.9 everywhere" gate (the
original stale target) is *unachievable in principle* on this task, and why the
user-signed-off `{2: 0.50, 3: 0.78}` gate is the realistic, honest choice
(ADR-D4; the §7.2 "imitation is information-limited" point).

### Random floors (chance accuracy)

A uniform-random classifier over the role's action head scores `1 / A`:

| Role  | Action head (`env.actions`) | Floor `1/A` |
|-------|-----------------------------|-------------|
| Cop   | `a_cop = 5` (UP, DOWN, LEFT, RIGHT, PLACE_BARRIER) | **0.20** |
| Thief | `a_thief = 4` (UP, DOWN, LEFT, RIGHT)              | **0.25** |

These are the floors the CI overfit smoke (`tests/unit/test_bc_train.py`) beats
*by a margin* — it does **not** assert the full gate, because a tiny vendored set
(`tests/fixtures/bc_mini_*.npz`) under-trains by construction.

### Why the ceiling is well below 1.0 — the privileged-expert vs local-obs gap

The BC **label** is produced by a *privileged* Manhattan-heuristic expert that
reads the full `GlobalState` (it always knows both agents' exact cells —
`cop_expert` runs a barrier-aware BFS to the thief; `thief_expert` maximizes
distance to the nearest cop). The BC **input** is the role's *local*
`Observation`: an egocentric window of radius `env.view_radius_by_grid` plus the
aliasing-memory scalars. When the opponent is **outside the view radius**, two
different global states that demand *different* privileged actions can map to the
*same* local observation. No classifier — however large — can separate them, so
the Bayes-optimal accuracy `P*` is strictly `< 1.0`. This is the documented BC
realizability gap in the P3 `src/marl/data/bc_dataset.py` module docstring
("Privileged-expert vs local-obs imitation gap (B3)").

`P*` is exactly the expected majority-label fraction over the partition of the
on-policy state distribution by *distinct local observation*:

```
P*  =  E_obs[ max_a  P(a* = a | obs) ]
```

The radius schedule makes this gap **grid-dependent**:

| Grid | `view_radius_by_grid` | Cop Bayes ceiling `P*` | Thief Bayes ceiling `P*` |
|------|-----------------------|------------------------|--------------------------|
| 2×2  | **0** (near-blind by design) | **~0.635** | ~0.66 |
| 3×3  | **1**                        | **~0.823** | ~0.80 |

The 2×2 stage is *near-blind on purpose* (radius 0 — a pipeline-sanity stage, not
a learning benchmark), so its ceiling is much lower; 3×3 opens the window to
radius 1 and the ceiling rises sharply. (An independent seeded estimate of `P*`
via majority-label-per-distinct-obs over 30 000 pairs lands at **0.66 / 0.80**
for the cop, corroborating the pinned `~0.635 / ~0.823` figures; the small
spread is the on-policy-distribution difference between the ε=0 estimator and the
ε-diversified collection.)

### Why the gate is `{2: 0.50, 3: 0.78}`

The gate sits **between** the random floor and the Bayes ceiling, leaving
headroom so a *passing* BC model is genuinely informative yet the gate is not
mathematically impossible:

| Grid | Random floor (cop) | Gate | Bayes ceiling (cop) |
|------|--------------------|------|---------------------|
| 2×2  | 0.20 | **0.50** | ~0.635 |
| 3×3  | 0.20 | **0.78** | ~0.823 |

`0.78 @ 3×3` keeps a deliberate ~0.04 margin under the ~0.823 ceiling — tight
enough to demand a near-Bayes model, loose enough to remain attainable. A single
`0.9` gate would exceed *both* ceilings and could never pass: that is the honest
§7.2 correction this analysis records. The gate is read at attach time by
`src.marl.nets.bc_train.gate_for(cfg, grid)` (keyed on `min(h, w)`), required for
**both** roles before OLoRA is attached.

> Reproduce: `gate_for` + the floors are pinned in
> `tests/unit/test_bc_train.py`; `P*` is estimable from
> `build_bc_dataset(cfg, grid, n, seed, role)` by grouping records on the exact
> `(image, scalars)` observation and averaging the per-group majority fraction.

## §9 Sensitivity analysis (V3-§9 — ex06's §9 is the inter-group bonus) — `env.view_radius_by_grid[4]` ∈ {1, 2}

A controlled SINGLE-parameter sweep: vary ONLY the 4×4 execution view radius (1 → 2)
with everything else pinned (algorithm = QMIX, nets / replay / γ / target cadence / the
256-episode warmup / 3 seeds identical). This is the §9 sensitivity analysis — distinct
from the IQL/VDN/QMIX ablation (which swaps the *algorithm*). `scripts/sensitivity_sweep.py`
→ `results/runs/sensitivity_view_radius.jsonl` → `results/figures/sensitivity_view_radius.png`.

| 4×4 view radius | final capture (mean ± SE, 3 seeds) |
|---|---|
| 1 (3×3 window — partial) | **0.713 ± 0.073** |
| 2 (5×5 window — covers the 4×4 board) | 0.670 ± 0.315 |

**Finding (honest, possibly counterintuitive).** Capture is genuinely *sensitive* to the
view radius — but **more observability did NOT clearly help** at the 50-round budget: the
means are close (0.71 vs 0.67) yet radius 2's SE is **~4× larger** (±0.32 vs ±0.07).
The wider window is mostly out-of-bounds padding on a 4×4 board and enlarges the encoder's
effective input, so it is harder to learn *stably* within budget (the high SE = some seeds
converge, some stall) — the same monotonic-mixer instability story as F5, now driven by
observation size rather than mixer richness. With only 3 seeds this is a directional
result, not a tight estimate; the takeaway is that the model is view-radius-sensitive and
that "see more" is **not free** — it trades a marginal mean for markedly worse stability.

> Reproduce: `uv run python scripts/sensitivity_sweep.py` (4×4 used because 5×5 training
> is too slow to sweep); the one-key-only guarantee is asserted by
> `tests/unit/test_sensitivity.py::test_make_variant_changes_only_the_one_key`.

## §10 Minimax-Q equilibrium baseline — the L11 §5 self-challenge (the SHIPPED bonus; ex06-§9's inter-group bonus is separate + deferred)

The deep arms (IQL/VDN/QMIX cops + a self-play Double-DQN thief) play *well* but certify no
equilibrium (THEORY §3). To close that gap — and the **L11 §5 self-challenge** — `src/marl/baselines/`
implements the tabular **Minimax-Q** learner L11 §2.2 prescribes for the *competitive* regime
(Littman 1994): one Q-table `Q(s, a_cop, a_thief)` whose per-state value is the **maximin LP**
`max_π min_j (πᵀQ)_j`, solved by `scipy.optimize.linprog` and validated against the L11 §2.2.1 worked
example (payoff `[[-2,4],[3,-1]]` → cop mix p=0.4, game value V=1.0). It runs on the reduced
**1-cop-vs-1-thief 3×3** zero-sum pursuit (capture → cop +1, escape/timeout → −1, γ=0.95, horizon
H=25), **reusing the production `CopsRobbersEnv`** (`TabularPursuit` adapter) so the baseline shares the
real transition/capture rules — only the deep-net observation/shaping path is bypassed.

**Result — the certified value converges to a closed-form floor** (seed 7, 5000 episodes; figure F7):

| metric (final window, ep 5000) | value | theory |
|---|---|---|
| game value at the reference start state | **−0.2915** | **−γ^(H−1) = −0.95²⁴ = −0.2920** (matches to 5e-4) |
| cop capture rate (rolling, ε-annealed) | **0.036** | ≈ 0 — the equilibrium thief escapes |

The certified game value descends (with small transient upticks) onto **−γ^(H−1)**, the discounted payoff of a *guaranteed
escape at the horizon*: a sure escape pays the cop −1 on the H-th transition, discounted by γ^(H−1) back
to the step-0 reference state. This is the genuine minimax equilibrium — on an **open** 3×3 grid with
**equal-speed simultaneous** moves a single pursuer provably cannot corner the evader, so the thief
survives to timeout and the cop's value is exactly the discounted escape penalty. The value **cannot fall
below** this floor (it is the worst single-episode return), so the curve *asymptotes onto it* rather than
diverging — the hallmark of correct convergence, and a closed-form check that the LP + learner are right.
In lock-step the capture rate falls from ~0.50 to **~0.04** as GLIE exploration anneals: under near-greedy
minimax play the lone cop catches the evader only ~4 % of the time.

**Convergence needed decaying α + GLIE (honest methodology).** *Constant* α=0.1 does **not** converge — the
value drifts *past* the floor to ≈−0.38 (an overshoot artifact of a non-vanishing step size on a moving
target). Adding Robbins-Monro **α-decay** (0.5→0.01) bounds it but leaves ε-noise; adding **GLIE ε-decay**
(0.3→0.01) lands it exactly on −γ^(H−1). This is the same "α must decay" lesson the course's tabular
learners encode, now visible on a 2-player game.

**Why this is the right contrast (README §7.2).** Minimax-Q buys an equilibrium *certificate* the deep
self-play cannot — but only by being tabular (a per-state Q + a per-state LP), which does not scale to the
padded recurrent multi-cop observation of the main task. And it certifies that **a lone minimax cop
loses**: capture on the open grid requires *cooperation*. That is exactly why the main project factors a
**team** of cops with VDN/QMIX value decomposition (which corners the thief a single minimax pursuer
cannot) and keeps the adversary in `T` (ADR-0006). The baseline is the theoretical *floor* the cooperative
deep arms rise above — not a competitor to them.

> Reproduce: `uv run python scripts/plot_minimax_q.py` (seed = `training.seeds[0]`; slow — per-step maximin
> LP, like the IQL/sensitivity baselines) → `results/figures/minimax_q.png`. The LP is checked against the
> L11 worked example by `tests/unit/test_minimax_lp.py::test_l11_worked_example_mixed_strategy`, and
> convergence-to-game-value by `tests/unit/test_minimax_q.py::test_q_table_converges_to_game_value`.

## §11 The seed population behind F5 — what the box plot (F8) and the heatmap (F9) add

F5 reports one mean±SE per arm, which is the right *summary* and the wrong *diagnosis*: two arms
with the same mean can be "uniformly mediocre" or "mostly fine with one catastrophic seed", and the
fix for those is not the same. The V3-§9.3 chart-variety pair exists to answer that — F8
(`results/figures/final_distribution.png`) keeps the seeds APART, F9
(`results/figures/capture_heatmap.png`) keeps the *stages* apart. Both read the same
`results/runs/history.jsonl` as F1/F2/F5/F6; "final" is each seed's mean over its **last 5 rounds**
(`aggregate.final_values_by_seed`, `last_k = 5`). The SEED is the independent unit everywhere:
`aggregate.final_by_algorithm` takes its mean±SE ACROSS the 5 per-seed means (n = 5), never across
the pooled 25 seed×round observations — rounds within a seed are autocorrelated, and pooling them
would have narrowed QMIX's SE from the honest ±0.102 to a spurious ±0.047.

### F8 — per-seed final capture rate at the 4×4 two-cop focus stage

| Algorithm | seed 7 | seed 17 | seed 37 | seed 71 | seed 107 | median | mean | F5 mean ± SE |
|---|---|---|---|---|---|---|---|---|
| IQL  | 0.788 | 0.812 | 0.812 | 0.852 | 0.814 | 0.812 | 0.816 | 0.816 ± 0.010 |
| VDN  | 0.870 | 0.860 | 0.794 | 0.876 | 0.824 | 0.860 | 0.845 | 0.845 ± 0.016 |
| QMIX | 0.816 | 0.736 | 0.694 | **0.232** | 0.660 | 0.694 | 0.628 | 0.628 ± 0.102 |

**Finding (honest, and unflattering to the arm we would rather defend).** QMIX's headline `0.63` is
**not** a uniformly weak arm — it is **four seeds in 0.66–0.82 plus one collapsed seed 71 at 0.232**.
On the box plot's own 1.5×IQR rule that point is a genuine flier (Q1 = 0.660, Q3 = 0.736, IQR =
0.076, lower fence = **0.546**), which is why F8 draws it detached below the whisker. Excluding it,
QMIX averages **0.727** — so a single failed run costs the reported mean ≈0.10. Three consequences,
stated plainly:

1. **The §7.2 ranking survives the outlier.** Even at 0.727, QMIX stays below IQL's *worst* seed
   (0.788) and VDN's worst (0.794). `VDN ≥ IQL > QMIX` at this 50-round budget is a real effect,
   not an artifact of one bad seed — we checked before claiming it.
2. **But the defect is one collapsed run (an outlier), not a lower plateau.** The right description
   of QMIX here is "four of five seeds learn, one stalls outright", which is exactly the R1
   monotonic-mixer instability and is invisible in a mean±SE. (Five observations are too few to
   claim any distribution shape beyond "an outlier exists".) Reporting only F5 would have
   understated QMIX's typical run *and* hidden its actual failure mode — both directions of error
   at once.
3. **IQL and VDN both learn on every seed** (ranges 0.788–0.852 and 0.794–0.876, seed-level SE
   ≈0.010/0.016). Between them the honest split is: **VDN has the highest mean (0.845), IQL the
   tightest spread** (per-seed SD ≈0.023 vs VDN's ≈0.035) — VDN is *not* the most consistent arm
   on dispersion, it is the highest-scoring one; IQL is the most consistent.

With 5 seeds a single outlier is 20 % of the sample, so this is a directional diagnosis, not a
tight variance estimate; the actionable read is "re-run QMIX with more seeds / a longer budget
before trusting its mean", not "QMIX is bad".

### F9 — mean final capture rate, algorithm × curriculum stage

| Algorithm | stage 0 (2×2, 1 cop) | stage 1 (3×3, 1 cop) | stage 2 (4×4, **2 cops**) | stage 3 (5×5, 1 cop) |
|---|---|---|---|---|
| IQL  | 0.998 | 0.947 | 0.816 | 0.611 |
| VDN  | 0.998 | 0.947 | 0.845 | 0.611 |
| QMIX | 0.999 | 0.920 | 0.628 | **0.716** |

**Finding — the arms only separate where the task is genuinely multi-agent.** The IQL and VDN rows
are **identical** at every **1-cop** stage (0, 1, and the 5×5 stage 3: 0.998 / 0.9468 / 0.611), and
that is a correctness signal rather than a coincidence: `env.curriculum.num_cops_by_stage = [1, 1, 2, 1]`,
so those rungs train a **single** cop, and VDN's sum-decomposition `Q_tot = Σ_i Q_i` over one agent
reduces exactly to IQL. Only stage 2 — the 2-cop team — creates a credit-assignment problem for a mixer
to solve or botch, and only there do the three arms spread (0.816 / 0.845 / 0.628). This is the empirical
justification for PRD K1's choice of the 4×4 two-cop stage as the comparison focus instead of the
degenerate 1-cop rungs (§7.2 L4). **The 5×5 column (brief §5.1 final test) is the control that clinches
it:** QMIX is *worst* at 4×4 (0.628, the 2-cop credit-assignment collapse) yet *best* at 5×5 (**0.716**,
where there is no team to coordinate) — so QMIX's instability is a multi-agent effect, not a property of
the harder board. Read left→right, every row also decays with board size + tighter view radius (the same
partial-observability cost F6 plots) — except QMIX's 4×4→5×5 rise, which is exactly the 2-cop→1-cop drop
in coordination burden.

> Reproduce: `uv run python -m src.results.make_figures` (builders:
> `src/results/plots_extra.py::plot_final_distribution` / `plot_capture_heatmap`); the per-seed
> numbers above are `aggregate.final_values_by_seed(load_runs("results/runs/history.jsonl"),
> "capture_rate", algorithm, stage=2)` and the matrix rows are their per-stage means.

## §12 Exploitability probe — why the greedy §3.5 match is 0–6 while the cop is a ~99% pursuer

The shipped match (`scripts/run_match.py`, seeds 7–12 on the 5×5 one-cop board — board size
fixed by `run_local_match`'s `stage=(5,5,1)` default plus `config game.grid_size: 5`, not by the
§3.5 report JSON, which carries no grid field) ends **Cop 30 – Thief 60**: the thief evades for
all 25 moves in all six sub-games. Read alone, that looks like a failed cop. The head-to-head
probe below shows the true picture is more interesting. The **committed evidence** for the table
is [`results/matchup/block_1000.json`](../results/matchup/block_1000.json) +
[`results/matchup/block_5000.json`](../results/matchup/block_5000.json), written by
`scripts/eval_matchup.py --json-out` (second block via `--base-seed 5000`). During development
the probe was additionally cross-checked adversarially — independent verification passes plus a
cross-examining critic, including a move-for-move fidelity check of the harness against the real
MCP referee path on the match seeds (300/300 ticks) — a process-level audit of that session,
attributed here as such rather than claimed as an in-repo artifact.

| Arm (60 sub-games/block, cop always greedy) | seeds 1000–1059 (`block_1000.json`) | seeds 5000–5059 (`block_5000.json`) |
|---|---|---|
| cop vs scripted flee baseline | **59/60** | 59/60 |
| cop vs uniform-random thief | **59/60** | 60/60 |
| cop vs OUR self-play thief (greedy, ε=0) | **8/60** | 8/60 |
| cop vs OUR thief, ε=0.10 exploration noise | **26/60** | 25/60 |
| cop vs OUR thief, ε=0.25 | **47/60** | 45/60 |

The flee baseline, described accurately: uniform-random over legal moves **before** first
opponent sighting, deterministic greedy distance-maximising **after**; it ignores its epsilon
argument. Pooled over the two committed blocks the cop catches a uniform-random thief
**119/120 (99.2%)**.

**Finding 1 — the cop is a competent pursuer, barriers included.** 59/60 against the scripted
flee policy (34 §3.3 barrier placements across the arm's 472 cop moves — both counts are in
`block_1000.json`) and 99.2% pooled against a random walker. The 0–6 match is not general
incompetence.

**Finding 2 — against our own self-play thief at greedy evaluation the cop captures 8/60
(~13%; 95% CI 8–21%, pooled over the two committed 60-seed blocks, which both landed on exactly
8/60).** The thief our self-play produced beats the cop our self-play produced, decisively,
under the serving convention.

**Finding 3 — but that advantage is a brittle DETERMINISTIC LOCK, not across-the-board thief
superiority.** Injecting small exploration noise into the thief alone recovers captures
monotonically in both committed blocks (8 → 26/25 → 47/45). At ε=0 both policies are bit-identically
reproducible across runs *and* across RNG seeds (the RNG is consumed every tick but provably
inert on actions), so each start position replays one fixed greedy escape trajectory — **distinct
per seed** (the 6 match seeds give 5 distinct sequences — seeds 7 and 8 happen to coincide — and one
of them is even captured), sharing a
left/right-oscillation motif — that exploits *this specific cop's* deterministic greedy
responses. One tick-level illustration from the development probe (a session observation, not a
committed artifact): on actual match seed 7, a single ε=0.10 deviation at tick 14 flipped a
25-move evasion into a capture at move 15. No claim is made about any other cop: only this one
was evaluated.

**Interpretation (§7.2 non-stationarity, honestly reported).** This is the classic self-play
pathology: alternating best-response converges to a co-adapted pair, and the frozen evaluation
pits a policy against the very opponent it was last shaped by. The correct summary of the match
is therefore *"the co-adapted thief wins the deterministic replay 6–0"*, not *"the thief is the
stronger agent"* — the ε-sweep refutes the stronger reading, and we report it anyway because the
probe was designed to be able to refute our own preferred narrative. Scope note: greedy ε=0 is
the **serving/match** convention (`agent_runtime` plays `act(..., 0.0, ...)`); the training-curve
capture rates elsewhere in this document are ε-annealed (1.0 → 0.05) rollout metrics — a
different quantity, deliberately not compared against this table.

> Reproduce: `uv run python scripts/eval_matchup.py --json-out results/matchup/block_1000.json`
> and `… --base-seed 5000 --json-out results/matchup/block_5000.json` (~minutes each, CPU; block
> shape from the `matchup_eval` config; service: `src/services/matchup_eval.py` via
> `MarlSDK.run_matchup_eval`). The committed JSONs above were produced by exactly those two
> commands; each column of the table reproduces bit-for-bit.

## §13 Foreign-cop stress test — §9 bonus readiness (adversarially verified)

Before the inter-group bonus we asked: *does the shipped thief lineup
(`AdaptiveThiefPolicy`: flee-primary, switch-to-net on first barrier sighting) survive cops we
did not train it against?* We built a battery — a perfect-information BFS oracle, a
partial-obs pursuit cop, a barrier-placing pursuit cop (`src/services/foreign_cops.py`, 15
TDD tests), and two best-response **exploiter** cops trained locally against our frozen
policies — and ran every cop against three thief candidates over 60-seed blocks, with a
disjoint second block, cop-side ε=0.05 noise variants, and 10×6-game **sticky-switch** matches
matching the real §9 semantics. During development a critic pass then independently re-drove
**17/17 decisive cells — all reproduced exactly** — and re-derived the §9 scoring from the
brief; that audit's verdict is recorded in the committed report's `critic_verdict` block (the
re-runs themselves were session work, not a separate artifact).

| Cop \ Thief (captures/60, greedy) | shipped Adaptive | raw net | pure flee |
|---|---|---|---|
| PursuitCop (no barriers) | **0** | 21 | 0 |
| OracleBfsCop (perfect info, no barriers) | **0** | 20 | 0 |
| BarrierPursuitCop | 25 (29 @block-2) | 35 (23 @block-2) — **statistical tie** | 1 |
| our own serving cop | 28 (sticky: 14) | 8 | 59 |
| exploiter (BR vs shipped) | 59 | 44 | 38 |
| exploiter (BR vs raw net) | 39 | **60** | 5 |

**Verdict — KEEP the shipped lineup (critic: UPHOLD).** Against barrier-less cops, staying in
flee mode is not a loophole but the optimal branch: even the perfect-information BFS oracle
scores **0/120 pooled** against flee (an equal-speed move-only chaser can never convert under
simultaneous resolution — the Minimax-Q escape floor, empirically), while the raw net concedes
~20/60 to the same cops. Against barrier cops the switch fires reliably (47–49/60; 10/10
sticky matches) and the lineups tie. Against dedicated best responses **both** lineups lose
(59–60/60) — exploiters exist for anything deterministic.

**Win model** (Table 1 ⇒ margin = 20·(A−B), win iff our cop's captures exceed theirs;
critic-audited): vs weak/median opponents P(win) ≈ 1.0; vs a strong barrier cop ≈ 0.87–0.90;
vs an opponent fielding a **net-quality thief** we likely lose (P ≈ 0.08) — the binding
constraint is **our cop** (proxy p=16/120 vs a net evader, likely pessimistic since the proxy
is our own co-adapted thief). Honest risks, stated plainly: a targeted exploiter takes only
~2 min to train **given the weights**, so weight visibility is the dominant risk lever — this
weight visibility is the dominant risk lever. It was managed by TIMING rather than by
secrecy: the repo stayed **private for the whole §9 negotiation and match** — read access to
the lecturer only — so neither group could train a best-response against the other's
published weights, and both sides played blind. That is the case the table above measures.
The repo was made **public after the match was played and both reports were agreed**, when
exposure can no longer affect a result.
§9 pays win +10 / **loss +7 / tie +5** (a tie is worth *less* than a
loss); and the bonus requires both groups' valid emails with `mutual_agreement=true`, else
both score 0. Full matrix, scoring model, and 12 caveats: `results/bonus/stress_report.json`.

> Reproduce: policies in `src/services/foreign_cops.py` (`foreign_cop_factories`), driven via
> `src/services/matchup_eval.py` (the cop now receives `state=` — net policies ignore it, the
> thief never gets it, referee bit-identity unaffected).
