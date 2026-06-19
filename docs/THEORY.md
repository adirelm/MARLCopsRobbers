# THEORY — Dec-POMDP / POSG formalism (Assignment 6, T1.1)

> **Group:** `adrl-001` · **Version:** `1.0.0` · **Source of truth:** `planning/BRIEF.md`
> (ex06.pdf §2.1, eq 1 / eq 3) + `L10-MARL.pdf`. Bibliography `[1]`–`[11]` per BRIEF §10.
> This is the **theory-first** (BRIEF §6.1) companion to README §7.1. Every value
> below is read from `config/config.yaml` — no number is hardcoded in prose or code.
>
> **Citation-numbering hazard (R13):** ex06/BRIEF and L10 number references differently
> (BRIEF `[2]` = VDN vs L10 `[7]` = VDN; ex06 §7.2 "eq 2" ≡ L10 "eq 4"). This doc and the
> README use **ex06/BRIEF numbering as primary** and footnote the L10 equivalence.

---

## 1. The cooperative Dec-POMDP `M` (eq 1, cite [1])

The cooperative pursuer team is modeled as a **Dec-POMDP** with the canonical tuple
(ex06 **eq 1**; primary source Bernstein et al. 2002 `[1]`; Oliehoek & Amato `[4]`):

> **(eq 1)**  `M = ⟨ N, S, {A_i}, T, R, {Ω_i}, O, γ ⟩`

| Symbol | Meaning | Concrete instantiation (config key) |
|---|---|---|
| **N** | agent set = the **cooperative COP TEAM** | `N` is the cooperative pursuer team that value decomposition spans — **NOT** `{cop, thief}`. `\|N\| = env.num_cops`: **1** in the graded 5×5 match (`env.num_cops`), **2** on the 4×4 training stage (`env.curriculum.num_cops_by_stage`) so the QMIX mixer is genuinely multi-agent. **The Thief is folded into the transition `T`** (a separate adversarial Double-DQN) and is NOT a member of `N`. Value decomposition NEVER crosses the cop/thief boundary. |
| **S** | global state (**train-only**) | `GlobalState(cop_pos, thief_pos, barriers, barriers_used, step, h, w, terminal)` — reachable ONLY via `env.state()`, never via `step()` (CTDE split). `h, w` are carried on the state (architect decision #1) so the live grid is self-describing. |
| **{A_i}** | per-agent action sets | The Dec-POMDP joint action is `A = ×_{i∈N} A_cop`. **`\|A_cop\| = 5`** `{UP, DOWN, LEFT, RIGHT, PLACE_BARRIER}` (`env.actions.a_cop`); **`\|A_thief\| = 4`** `{UP, DOWN, LEFT, RIGHT}` (`env.actions.a_thief`, `PLACE_BARRIER` masked). `STAY`/`NOOP` exists in the enum but is **config-gated OFF** (`env.actions.enable_stay: false`) — the BRIEF rule is "both agents move one cell each turn", so there is no STAY/NOOP action head. `PLACE_BARRIER` is cop-only and masked when `barriers_used == game.max_barriers`. The thief's `{A_thief}` belongs to the full POSG `T` (§3), not to `M`'s joint action. |
| **T** | transition `T(s'\|s, ā)` | Deterministic core: **simultaneous** joint-move resolution (`env.move_resolution`); barriers and board edges impassable to both; barrier placement consumes the move; optional `env.transition_noise` (default 0.0). The Thief is folded into `T` from any single cop learner's view ⇒ **non-stationarity**. |
| **R** | reward | RL signal = config-driven shaped reward, **shared over the cop team** in `dec_pomdp` mode (`env.reward_mode`), per-agent in `posg`. Potential-based shaping with team potential `Φ = −min_{i∈N} manhattan(cop_i, thief) / d_max` (the *closest* cop sets the potential). The §3.4 scoreboard (20/10/5/5) is a **separate** `Scorer`, never the training signal. See §4. |
| **{Ω_i}** | observation spaces | Per-agent egocentric Manhattan-radius window, padded to a fixed `2·env.view_radius_max + 1` footprint so one encoder spans 2×2→5×5. Exposed as the local `Observation(image, scalars)` TypedDict — it carries NO global field (architect decision #3). |
| **O** | observation function `O(ō\|s', ā)` | Deterministic crop of `S` to radius-`r` cells around agent `i` (`env.view_radius_by_grid`); `O = 1` for the true window; optional `env.obs_noise` flips the opponent-visible channel. |
| **γ** | discount | `algo.gamma` (default `0.99`, suited to the ≤25-step horizon, `game.max_moves`). |

We **deliberately collapse** the pursuer view to a cooperative Dec-POMDP with shared `R`
to license CTDE value decomposition (VDN `[2]` / QMIX `[3]`) on the **IGM** principle. The
README §7.1 names this proxy explicitly as a modeling limitation — see §3.

---

## 2. Value functions, targets & non-stationarity (README §7.2)

```
(eq 8)  Recurrent state:        z_t = f_φ(z_{t-1}, o_t)             # GRU hidden, defeats aliasing
(eq 5)  IGM:                    argmax_ā Q_tot = (argmax_{a_1}Q_1, …, argmax_{a_N}Q_N)
(eq 6)  VDN (ablation):         Q_tot = Σ_{i∈N} Q_i(h_i, a_i)
(eq 7)  QMIX (primary):         Q_tot = f_mix(Q_1..Q_N ; s),   ∂Q_tot/∂Q_i ≥ 0
(eq 2 ≡ eq 4) IQL target:       y_i = r_i + γ max_{a'_i} Q_i(o'_i, a'_i)      # plain ex06/L10 form
        Non-stationarity (why the IQL target dances):
            P_i(s'\|s,a_i) = Σ_{a_-i} π_-i(a_-i\|o_-i) · T(s'\|s,a_i,a_-i)    # π_-i drifts → target moves
```

> **Footnote (R13):** ex06/BRIEF "eq 2" ≡ L10 "eq 4" (IQL target); BRIEF `[2]` (VDN) ≡ L10 `[7]`.

---

## 3. The general-sum POSG caveat (eq 3 — required honesty, README §7.1)

The *faithful* full game is a general-sum **Partially-Observable Stochastic Game (POSG)**:

> **(eq 3)**  `G = ⟨ I, S, {A_i}, {O_i}, P, Ω, {R_i}, γ ⟩`,  with `R_cop ≠ R_thief`

Here `I = {cop, thief}` (a **2-role** game; distinct from the Dec-POMDP `N` = cop team only),
the rewards are role-specific (the 20/10/5/5 scores are opposed), and the worst-case planning
complexity class is **NEXP^NP**. A cooperative mixer over `{cop, thief}` is **invalid by
construction**: opposed rewards violate the monotonicity constraint `∂Q_tot/∂Q_i ≥ 0` (eq 7).
We therefore confine value decomposition to the cooperative cop team `N` and fold the Thief
into the transition `T(s'│s, ā)` as a separate adversarial Double-DQN trained by self-play.
README §7.1 names the Dec-POMDP-with-shared-R as the deliberate proxy; §7.2 analyzes the
limits (IGM / monotonicity → QPLEX `[10]` / Weighted-QMIX `[9]`).

---

## References

`[1]` Bernstein et al. 2002 — Dec-POMDP complexity. `[2]` Sunehag et al. 2018 — **VDN**
(1706.05296). `[3]` Rashid et al. 2018 — **QMIX** (1803.11485). `[4]` Amato 2024 —
Cooperative MARL intro (2405.06161). `[9]` Rashid et al. 2020 — **Weighted QMIX**
(2006.10800). `[10]` Wang et al. 2021 — **QPLEX** (2008.01062). Full list in `docs/PRD.md` §13.
