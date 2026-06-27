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

## §9 Sensitivity analysis — `env.view_radius_by_grid[4]` ∈ {1, 2}

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
means are close (0.71 vs 0.67) yet radius 2's variance is **~4× larger** (±0.32 vs ±0.07).
The wider window is mostly out-of-bounds padding on a 4×4 board and enlarges the encoder's
effective input, so it is harder to learn *stably* within budget (the high SE = some seeds
converge, some stall) — the same monotonic-mixer instability story as F5, now driven by
observation size rather than mixer richness. With only 3 seeds this is a directional
result, not a tight estimate; the takeaway is that the model is view-radius-sensitive and
that "see more" is **not free** — it trades a marginal mean for markedly worse stability.

> Reproduce: `uv run python scripts/sensitivity_sweep.py` (4×4 used because 5×5 training
> is too slow to sweep); the one-key-only guarantee is asserted by
> `tests/unit/test_sensitivity.py::test_make_variant_changes_only_the_one_key`.

## §10 Minimax-Q equilibrium baseline — the L11 §5 self-challenge (P-bonus)

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

The certified game value descends monotonically to **−γ^(H−1)**, the discounted payoff of a *guaranteed
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
