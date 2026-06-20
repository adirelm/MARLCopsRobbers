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
