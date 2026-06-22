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
