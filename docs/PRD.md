# PRD — Assignment 6: MARL Cops & Robbers (Cloud-MCP, Vibe Coding)

> **Group:** `adrl-001` · **Version:** `1.0.0` · **Recipient of record:** `rmisegal+marl@gmail.com`
> **Source of truth:** `planning/BRIEF.md` (distilled from `ex06.pdf`, 23 pp, Dr. Yoram Segal, 2026) + `L10-MARL.pdf`.
> **Grading rubric:** V3 software-excellence guidelines (`software_submission_guidelines V3.00`).
> **Status:** Foundational requirements document. Human-decided columns (CLAUDE.md §1.4) are LOCKED by the architect; AI executes against this spec.

This PRD is the contract for *what* the system must do and *how its success is judged*. Architecture and *how* live in `docs/PLAN.md` (C4/UML/ADRs); the ordered task list lives in `docs/TODO.md`. Every functional requirement carries an **ID**, **acceptance criteria (AC)**, and an **evidence pointer**. Brief sections are cited as `§N`; bibliography as `[1]`–`[11]` (see §13).

---

## 1. Project Overview

DroneRL's successor. We build a **Multi-Agent Reinforcement Learning (MARL)** "Cops & Robbers" pursuit game between **two fully autonomous AI agents** — one **Cop**, one **Thief** — on a dynamic urban grid, developed under the **Vibe Coding** methodology (§0, §1.1). Each agent perceives only a **bounded Manhattan-radius local view** with **no access to global state at execution time**; the problem is therefore modeled as a **Dec-POMDP** (cooperative framing) with an explicit **POSG** (general-sum) caveat (§2.1, eq 1, eq 3, cite [1]).

Training uses **Centralized Training, Decentralized Execution (CTDE)** with **value decomposition** (VDN [2] / QMIX [3]) on the **Individual-Global-Max (IGM)** principle (§2.1, §4, L10 §2–§4). At execution the two agents run as **separate OS processes, each exposing its own FastMCP HTTP server**, mediated by a neutral **referee** that holds ground-truth state and feeds each agent only its masked local observation. Servers run **localhost first (Stage 1)**, then **cloud (Stage 2)** with revocable bearer-token auth (§5.3, §8, [11]). A mandatory **Pygame GUI** renders the match as a god-view spectator (§5.4). At the end of the whole match the **Cop sends exactly one automatic Gmail report** (JSON of all 6 sub-games) to `rmisegal+marl@gmail.com` (§3.5, §5.5).

### 1.1 Locked architect decisions (CLAUDE.md §1.4 — human-decided, non-delegable)

The architect chose **"maximize the grade; complexity/time acceptable."** The locked scope is:

| # | Decision | Rationale / Brief anchor |
|---|---|---|
| **L1** | **Full ambition scope:** MARL + CTDE + two FastMCP servers (localhost **and** cloud) + Pygame GUI + Gmail report + full OLoRA ablation. Build the §9 bonus-capable report schema and an inter-group MCP interface spec (the actual cross-group match needs a partner `biu-rlNN` group — a **coordination dependency**, not buildable solo). | §0, §5, §9 |
| **L2** | **Algorithm:** **QMIX PRIMARY** (monotonic mixer, eq 7) + **VDN ablation** arm (eq 6) + **IQL baseline** (the required §7.2 contrast, eq 2/eq 4). QPLEX [10] + Weighted-QMIX [9] discussed in §7.2 (not necessarily built). MAPPO is an **optional** stretch contrastive arm (note: not in the brief bibliography). Value decomposition applies **only within the cooperative Cop team**; the Thief is a separate **adversarial Double-DQN** folded into the env transition `T(s'│s,ā)` and trained by **self-play** (alternating best-response). One **shared recurrent GRU Q-net** + a **swappable Mixer ABC** seam (DRY). | §2.1, §5.2, §7.2 |
| **L3** | **Reward:** **potential-based shaping** (provably policy-invariant, Ng 1999: `F = γΦ(s') − Φ(s)`, team potential `Φ = −min_{i∈N} Manhattan(cop_i, thief) / d_max`, with `Φ(terminal)=0` on absorbing states — see FR-ENV-6; `−1/step` + distance-to-thief potential) during **training only**, **toggled off at eval**. The official §3.4 scoreboard (cop_win 20 / thief_win 10 / cop_loss 5 / thief_loss 5) is computed **separately** for reports. | §3.4, §7 |
| **L4** | **Cop-team:** the **graded 5×5 match is a faithful 1-cop-vs-1-thief** pursuit (§3); a **2-cop scenario runs ONLY on the 4×4 training/sanity grid** (`config num_cops`) so the QMIX mixer is genuinely multi-agent and §7.2 learning curves show real credit assignment. This does **NOT** change the graded game rules. | §3, §5.1, §7.2 |

### 1.2 Standing defaults (apply unless overridden in config)

Simultaneous move resolution (`env.move_resolution:simultaneous`) + capture on same-cell **OR** swap (`env.capture_on_swap:true`) · per-grid view radius from `env.view_radius_by_grid` (2×2:0, 3×3:1, **4×4:1** so partial-observability bites, 5×5:2; `env.view_radius_max:2`) · `PLACE_BARRIER` counts as the move (`env.actions.enable_barrier:true`) · barriers cop-only, ≤`game.max_barriers`=5/sub-game · curriculum `env.curriculum.stages` 2×2→3×3→4×4→5×5 · seeds `training.seeds` `[7, 17, 37, 71, 107]` · Pygame GUI = god-view spectator reading the referee (NOT agent obs) · a neutral **referee = "the environment"** drives the 6 sub-games and calls each agent MCP tool per turn · cloud = `cloud.platform:prefect_horizon` (FastMCP) · bearer JWT auth (`mcp.auth`, RS256) + `jti` deny-list revoke · Gmail via `smtplib` + App-Password (`gmail.mechanism:smtplib_app_password`) with an `EmailSender` Protocol fallback to Google-API OAuth · version `1.0.0`.

### 1.3 Submission model — SOLO

BRIEF §5 defaults to a **2-student group (roles A, B; optional C)**, but **this is a
SOLO submission** (single student, **role A**) under the course's general
solo-work approval and the A1–A5 precedent. Consequences: the §3.5 report
`students[]` and the cover sheet carry **exactly one student**; the "two AI agents"
(Cop, Thief) are **two processes / two MCP servers**, not two people, so the
architecture is unchanged. The **`_A_` / `_B_` owner tags in `PLAN.md`/`TODO.md`
are work-stream labels** (A ≈ RL/env/training track, B ≈ infra/MCP/GUI/report
track) for sequencing by a **single developer**, not two assignees. The **§9 bonus
is inter-*group*** regardless of group size — it still requires a partner
`biu-rlNN` group, so it stays a coordination dependency (interface built, match
deferred until a partner exists).

---

## 2. Goals & Non-Goals

### 2.1 Goals
- **G1.** Satisfy the four §1.1 learning outcomes: (1) model partial observability as a Dec-POMDP; (2) design + implement CTDE + value decomposition across distributed agents; (3) practical MCP + Vibe-Coding; (4) critically analyze algorithmic limits and real-world implications.
- **G2.** Ship a reproducible, locally-trained CTDE pipeline whose **execution path provably uses local observation only** (the single most-graded modeling requirement, §4, §6.1).
- **G3.** Produce all six mandatory §7.3 figures (learning curves, loss curves, GUI screenshots, MCP-comms proof, IQL-vs-CTDE baseline, scale effect) **reproducible from one command**.
- **G4.** Pass every V3 hard gate (see §6) with machine-checkable evidence.
- **G5.** Deliver a polished README that *is* the §7 academic paper, mapping 1:1 onto §7.1/§7.2/§7.3.

### 2.2 Non-Goals (explicitly out of A6 scope)
- Running the actual §9 **inter-group bonus match** (requires a partner `biu-rlNN` group; counts toward the **final project**, not A6). We ship the *schema* and *interface spec* only.
- Training in the cloud (cloud is **inference-only**, decentralized execution; training MUST be local, §5.2).
- A human-in-the-loop game (the match is fully autonomous, §3).
- Importing any foundation/HF LLM (violates "training MUST be local"; the "pre-trained model" is a locally behavior-cloned encoder, §5.2, see PRD-D4-1).

---

## 3. The Dec-POMDP Formalism (eq 1, cite [1])

The cooperative pursuer team is modeled as a **Dec-POMDP** with the canonical tuple (ex06 eq 1, primary source Bernstein et al. 2002 [1]; Oliehoek & Amato):

> **(eq 1)**  `M = ⟨ N, S, A, T, R, Ω, O, γ ⟩`

Instantiated for Cops & Robbers (cooperative pursuer framing, L10 §2.2):

| Symbol | Meaning | Concrete instantiation |
|---|---|---|
| **N** | agent set = the **cooperative Cop team** | `N` is the cooperative pursuer team that the Dec-POMDP/value-decomposition spans, **NOT** `{cop, thief}`. `\|N\| = env.num_cops`: **1** in the graded 5×5 match (`env.num_cops:1`), **2** in the 4×4 training scenario (`env.curriculum.num_cops_by_stage:[1,1,2,1]`) per L4 so the QMIX mixer is genuinely multi-agent. The **Thief is folded into the transition `T`** (a separate adversarial Double-DQN, ADR-D1-B/ADR-D3-A) and is NOT a member of `N`; the full game is the general-sum **POSG** (§3.1, eq 3). Value decomposition never crosses the cop/thief boundary. |
| **S** | global state (**train-only**) | `GlobalState(cop_pos:tuple[(r,c),…], thief_pos:(r,c), barriers:frozenset[(r,c)], barriers_used:int, step:int, h:int, w:int, terminal:bool)` — `cop_pos` is a tuple (1 entry graded 5×5, 2 entries on the 4×4 2-cop stage). **`h, w` are carried on the state (architect decision #1)** so the live grid is self-describing (drives the derived `d_max=(H-1)+(W-1)` of FR-ENV-6 and the per-stage obs/state pad, with no per-stage size constant). Encoded for the centralized critic as a **stage-invariant 77-float** footprint (`3·G·G + 2`, G=5; out-of-board cells masked across all curriculum stages — see FR-ENV-7b). Reachable ONLY via `env.state()`, **never** via `step()`. |
| **A** | Dec-POMDP joint action = the **cop-team joint action** | In the Dec-POMDP `M`, **`A` is the joint action over the cooperative Cop team `N`** (`A = ×_{i∈N} A_cop`). The per-role enums `\|A_cop\| = 5 {UP,DOWN,LEFT,RIGHT,PLACE_BARRIER}` and `\|A_thief\| = 4 {UP,DOWN,LEFT,RIGHT}` belong to the full **POSG/transition** `T` (§3.1), **not** to this Dec-POMDP `A` (the Thief is folded into `T`, not into `N`'s joint action). `STAY`/`NOOP` exists in the enum but is config-gated **off** by default (brief rule: both move one cell). `PLACE_BARRIER` is cop-only, masked when `barriers_left == 0`. |
| **T** | transition `T(s'│s,ā)` | **Deterministic, RNG-free** core (reproducible CI): **simultaneous** joint-move resolution; barriers & edges impassable to both; barrier placement consumes the move. No stochastic-transition option is implemented — the kernel is intentionally pure. The Thief is folded into `T` from any single cop learner's view → **non-stationarity** (§2.1). |
| **R** | reward | RL signal = config-driven shaped reward (L3), **shared over the cooperative Cop team** in `dec_pomdp` mode (one global scalar all cop agents receive), per-agent in `posg` mode. Potential-based shaping uses a team potential `Φ = −min_{i∈N} Manhattan(cop_i, thief) / d_max` (grid-normalized; the *closest* cop sets the potential so the team is rewarded for any cop closing in). The official §3.4 scoreboard is a **separate** `Scorer`, never the training signal. |
| **Ω** | observation space | Per-agent egocentric Manhattan-radius window. |
| **O** | observation function `O(ō│s',ā)` | Deterministic crop of S to radius-`r` cells around agent `i`; `O = 1` for the true window. Partial observability comes from the **bounded view radius**, not observation noise — no `obs_noise` option is implemented. |
| **γ** | discount | config `gamma`, default `0.99` (≤25-step horizon). |

**Exec-time `Observation` (the only thing `step()` / MCP ever emits — architect decision #3).** `Observation` is a `TypedDict` carrying ONLY local fields; it MUST NOT contain `barriers`, absolute `cop_pos`/`thief_pos`, `h`/`w`, or the global `step` int (those live on `GlobalState` above and are train-only). The P1 type-level split test (`tests/architecture/test_train_exec_split.py`) and the P2 runtime leak test (T2.3) both enforce this boundary.

| Field | Shape / type | Meaning |
|---|---|---|
| `image` | `(env.obs_channels, W_v, W_v)` float32, `W_v = 2·env.view_radius_max + 1 = 5` | Egocentric window; channels: self, other_visible, barrier, out_of_bounds, time_norm. |
| `scalars` | `(env.obs_scalars,)` float32 | own_r_norm, own_c_norm, step_norm, barriers_left_norm, other_seen_now, steps_since_seen_norm (aliasing-memory hooks). |

### 3.1 POSG caveat (eq 3 — required honesty, §7.1)
The *faithful* game is a general-sum **POSG** (eq 3) `G = ⟨I, S, {A_i}, {O_i}, P, Ω, {R_i}, γ⟩` with `R_cop ≠ R_thief` (the 20/10/5/5 scores are role-specific), complexity class **NEXP^NP**. We **deliberately collapse** the pursuer view to a cooperative Dec-POMDP with shared `R` to license CTDE value-decomposition; §7.1 names this proxy explicitly as a modeling limitation. A cooperative mixer over `{Cop, Thief}` is invalid by construction (opposed rewards violate `∂Q_tot/∂Q_i ≥ 0`); **value decomposition lives only inside the cooperative Cop team** (L2, ADR-D1-B / ADR-D3-A).

### 3.2 Value function & targets (§7.1/§7.2)
```
(eq 8)  Recurrent state:        z_t = f_φ(z_{t-1}, o_t)             # GRU hidden, defeats aliasing
(eq 5)  IGM:                    argmax_ā Q_tot = (argmax_{a_1}Q_1, …, argmax_{a_N}Q_N)
(eq 6)  VDN (ablation):         Q_tot = Σ_{i∈cop} Q_i(h_i, a_i)
(eq 7)  QMIX (primary):         Q_tot = f_mix(Q_1..Q_N ; s),   ∂Q_tot/∂Q_i ≥ 0
(eq 2≡4) IQL target (baseline): y_i = r_i + γ max_{a'_i} Q_i(o'_i, a'_i)          # plain ex06/L10 form
        ↳ our IQL learner uses the Double-DQN target-net refinement OF this independent target
          (y_i = r_i + γ(1-d) Q_i(o'_i, argmax_{a'_i}Q_i(o'_i,·;θ_i); θ_i⁻)), NOT a byte-identical
          copy of eq 2/eq 4 — it stays an *independent per-agent* target (no mixer, no global s).
        Centralized CTDE target: y_tot = r_team + γ(1-d) Q_tot⁻(s', greedy(ā'))
        Non-stationarity (why eq 4 dances):
            P_i(s'│s,a_i) = Σ_{a_-i} π_-i(a_-i│o_-i) · T(s'│s,a_i,a_-i)   # π_-i drifts → target moves
(eq 10-11) OLoRA:               W0 = QR(thin);  W_adapted = W0 + s·(B@A)
```
> **Citation-numbering hazard (carry into README, R13):** ex06/BRIEF and L10 number references differently (BRIEF [2] = VDN vs L10 [7] = VDN; ex06 §7.2 "eq 2" ≡ L10 "eq 4"). README MUST use ex06/BRIEF numbering as **primary** and footnote the L10 equivalence.

---

## 4. KPIs & Success Criteria

| KPI | Target | Measurement | Evidence |
|---|---|---|---|
| **K1 Credit-assignment / decomposition (4×4 2-cop)** | Credit-assignment studied on the **4×4 2-cop scenario** (`env.curriculum.num_cops_by_stage[2]=2`) — the genuinely multi-agent mixer, NOT the degenerate 1-cop 5×5 (trivial scalar gain, §7.2 L4). **Honest empirical finding (reported faithfully, README §7.2): the `QMIX ≥ VDN ≥ IQL` hypothesis is FALSIFIED at the 50-round budget** — `VDN (0.85) ≥ IQL (0.82) > QMIX (0.63±0.05)`; QMIX's monotonic hypernetwork is more expressive but harder to train (R1 instability), so the simpler decompositions win at this bounded budget | seeded sweep on the 4×4 2-cop stage, identical nets/replay/seeds/curriculum across arms | F5 `results/figures/baseline_comparison.png` (4×4 2-cop panel) |
| **K2 Cop policy quality (4×4 focus stage)** | Honest result at the **4×4 focus stage** (held-out seeds, shaping off): the best CTDE arm (VDN, **0.85**) clears the IQL baseline (**0.82**); the graded-primary QMIX **lags** IQL at this 50-round budget (**0.63**, the R1 instability — see K1). The graded 1-cop **5×5 was NOT swept** (excluded from the figure matrix for compute — README §7.3), so K2-as-originally-specified (strictly-above-IQL on 5×5) was not evaluated | held-out 4×4 capture-rate, shaping off | F5 (4×4 focus panel) |
| **K3 Decentralized-execution invariant** | 100% — no global state reaches any execution/MCP path | architecture tests (import-boundary + MCP-schema leak tests) | `tests/architecture/` green |
| **K4 Match completeness** | Exactly **6 valid** sub-games per match; technical losses replayed (§3.7) | integration test with fault injection | `tests/integration/test_match.py` |
| **K5 Reproducibility** | The four **plotted** figures **F1/F2/F5/F6** regenerate from `uv run python -m src.results.make_figures` reading `results/runs/*.jsonl`, with zero README/code drift; **F3 (GUI) + F4 (MCP-comms) are deterministically CAPTURED** (headless screenshot / redacted comms log), **not plotted from JSONL** — they reproduce via their seeded capture scripts | `results/runs/*.jsonl` + `experiment_manifest.json` (runs, arms, stages, seeds, config hash) for F1/F2/F5/F6; seeded capture scripts for F3/F4 | F1–F6 + manifest |
| **K6 Cloud comms** | One shared `trace_id` appears in BOTH servers' logs for cop↔thief calls; 401 without token; revoke demo | cloud smoke run + redacted captures | F4 `_cloud.png`, `cloud_auth_401.png`, `cloud_revoke.png` |
| **K7 V3 hard gates** | All green (≤150 LOC/file, ≥85% cov, ruff 0, no-hardcode, no-secrets, uv-only, single-SDK-entry, version 1.0.0) | CI pipeline + `software-excellence-v3` audit | `instructions/assignment-6/submission_guidelines_audit.md` |
| **K8 Seeds** | ≥3 seeds `[7,17,37]` floor; 5 `[7,17,37,71,107]` for error-bar strength | run manifest | F1/F5/F6 with mean±SE |

---

## 5. Functional Requirements

Each FR has an **ID · statement · acceptance criteria (AC) · evidence pointer**. FRs are grouped by capability area and aggregate the `prdContributions` from all ten dimension syntheses.

### 5.1 Game Rules & Environment (§3, §5.1) — Dec-POMDP `CopsRobbersEnv`

- **FR-ENV-1 (eq 1 formalism).** A custom **grid-agnostic** `CopsRobbersEnv` implements the formal Dec-POMDP tuple ⟨N,S,A,T,R,Ω,O,γ⟩ (eq 1, cite [1]).
  - **AC:** a unit test asserts the Dec-POMDP agent set is the **cop team** with `\|N\| == env.num_cops` (=1 in the graded 5×5, =2 on the 4×4 2-cop stage — NOT a literal 2), per-role action enums (`\|A_cop\|=5`, `\|A_thief\|=4`), and `γ` read from config; README §7.1 maps each tuple symbol to code and notes the full game is a **2-role POSG** `{cop, thief}` (§3.1) whose Thief is folded into `T`, so the Dec-POMDP `N` is the cop team only.
  - **Evidence:** `tests/unit/test_env.py`; README §7.1.
- **FR-ENV-2 (train/exec split — top-graded).** `step()` returns **LOCAL** per-agent observations only; global state is reachable ONLY via `env.state()`.
  - **AC:** a full-episode test asserts `step()` output contains no full-board positions / no `GlobalState` fields; `state()` and `render_state()` are the only global accessors; an MCP-payload test proves only local obs is transmitted.
  - **Evidence:** `tests/architecture/test_step_no_leak.py`; `tests/architecture/test_egress_via_gatekeeper.py`.
- **FR-ENV-3 (actions).** Per-agent action set with `PLACE_BARRIER` cop-only; invalid actions map to NOOP, are logged, and still consume the step. Action-selection masking is **additive `−inf`** (`Q.masked_fill(~legal, -inf)` before the argmax/softmax), **NOT** multiplicative `mask·Q` — multiplicative masking is wrong for **negative** Q-values (it would make an illegal action with `Q=0` look better than a legal action with `Q<0`).
  - **AC:** action-mask tests for thief-`PLACE_BARRIER`-masked, out-of-bounds, barrier-blocked, and budget-exhausted cases; a test asserts that with a **negative-Q** vector an illegal action is never selected (i.e. additive `−inf` masking, not `mask·Q`).
  - **Evidence:** `tests/unit/test_actions.py`.
- **FR-ENV-4 (transition & barriers, §3.3).** Transition is **simultaneous-resolution** (`env.move_resolution:simultaneous`); barriers and board edges are impassable to **both** agents; a barrier is placed on the cop's current cell and consumes the move; ≤`game.max_barriers` (5) per sub-game.
  - **TransitionResult contract.** `transition.resolve_joint_action(state, joint_a, cfg)` returns a **frozen `TransitionResult(next_state: GlobalState, winner: str|None, capture: bool)`** (the env's only dynamics kernel). It is **PURE** (no RNG, no mutation): every intended next cell is computed from the SAME pre-move `state` (positions + `state.barriers`), so resolution is order-independent and replay-deterministic. `next_state` is a NEW frozen `GlobalState` at `step+1` with updated `cop_pos`/`thief_pos`/`barriers`/`barriers_used` and the `terminal` flag set per FR-ENV-5.
  - **Deterministic over-budget (codex F1).** Barrier placement honors the **TEAM** budget cap (`game.max_barriers`) iterated **by cop index**: a `PLACE_BARRIER` is honored only while `barriers_used < max_barriers`; once the team cap is hit, later cops' `PLACE` **deterministically** degrades to a stay (lowest cop index wins the last barrier). This makes a 2-cop simultaneous over-budget tick resolve identically on every replay.
  - **Double-occupancy allowed.** On the 4×4 2-cop training stage there is **no inter-cop collision** — two cops may share a cell (cooperative team; not a graded rule). Only a cop↔thief co-location (or swap) is a capture.
  - **AC:** tests for edge==wall (stay-in-place), barrier blocks entry for both agents, barrier-budget cap, placement increments `step_count`, **2-cop simultaneous over-budget resolves deterministically by cop index**, **2-cop double-occupancy is permitted**.
  - **Evidence:** `tests/unit/test_transition.py`.
- **FR-ENV-5 (capture & termination, §3.2).** Capture occurs when cop and thief share a cell after the simultaneous update, **including a one-step swap** (`env.capture_on_swap:true`). Termination = capture OR `step ≥ game.max_moves` (25). Treating a simultaneous cop↔thief cell-swap as a capture is a **deliberate, documented architect game-rule call** (BRIEF §3.2 is silent on swap-through under simultaneous resolution; counting it as capture keeps the pursuit well-posed) — a human-decided, non-delegable rule clarification (CLAUDE.md §1.4).
  - **Capture beats timeout (edge-case rule).** Capture is checked **BEFORE** timeout, so a capture **on the final move** yields `winner="cop"` even when `(state.step + 1) >= game.max_moves` (ADR-0004 extended). Timeout (`winner="thief"`) fires only when NOT captured and `(state.step + 1) >= game.max_moves` — the tie `(step+1) == max_moves` is a thief win unless that same tick is also a capture.
  - **Any cop captures.** With ≥2 cops, capture is `any(cop ends on thief's cell) OR any cop↔thief swap` — **any** cop touching the thief ends the sub-game for the team (no per-cop tag distinction).
  - **AC:** tests for shared-cell capture, swap=capture, timeout at exactly `game.max_moves`, **capture-on-final-move beats timeout**, **any-of-N cops captures**, `winner ∈ {cop, thief}`.
  - **Evidence:** `tests/unit/test_transition.py`; ADR-001/ADR-0004 (move resolution + swap-as-capture + capture-beats-timeout clarification).
- **FR-ENV-6 (reward decoupling, L3).** The RL reward is a config-driven shaped signal with **potential-based, grid-normalized** distance shaping (`F = γΦ(s') − Φ(s)`, Ng 1999, team potential `Φ = −min_{i∈N} Manhattan(cop_i, thief) / d_max`); per-agent POSG rewards are also computed. **Normative (architect decision #2): `d_max = (H − 1) + (W − 1)` is derived per live grid** from the state's own `h, w` (the max Manhattan span of an `H×W` board) — it is **NOT a config key**, so the same `Φ` is correct at every curriculum stage 2×2→5×5 without a per-stage constant. **Φ MUST be 0 on every absorbing/terminal state** (`reward.phi_terminal_zero:true`) — Ng-1999 policy-invariance requires `Φ(absorbing)=0`, so capture sets `Φ=0` automatically and the timeout + swap-capture terminals are the explicit edge cases to pin. The Table-1 game points (20/10/5/5) are produced by a **separate** `Scorer` and never reused as RL reward; shaping is toggled off at eval.
  - **AC:** a test asserts the potential-shaping identity; a test asserts `RewardModel` and `Scorer` are distinct modules; a test asserts `dec_pomdp` reward-dict entries are equal; a test on the **terminal transition** asserts `Φ(s_terminal)=0` (so `F = γ·0 − Φ(s) = −Φ(s)`) for capture, timeout, and swap-capture endings.
  - **Evidence:** `tests/unit/test_reward.py`, `tests/unit/test_scorer.py`.
- **FR-ENV-7 (partial observation Ω/O).** A fixed `5×5` egocentric image with `env.obs_channels`=5 channels (self, other_visible, barrier, out_of_bounds, time_norm) + an `env.obs_scalars`=6 scalar vector, **padded across all curriculum stages** to footprint `2·env.view_radius_max+1`=5 so one network spans 2×2→5×5 (enables OLoRA). View radius comes from `env.view_radius_by_grid` (auto `r = max(0, ⌈min(H,W)/2⌉ − 1)`: 2×2:0, 3×3:1, 4×4:1, 5×5:2) with config override; the opponent channel is zeroed beyond Manhattan radius.
  - **Env-owned visibility memory (ownership rule).** `build_observation` is **PURE given `memory`**; the **env owns and updates** the per-role visibility memory (`{role: VisibilityMemory(steps_since_seen)}`). `reset` re-initializes it; `step` resets a role's `steps_since_seen` to 0 when its opponent is in view, else `+1`. The memory feeds the `steps_since_seen_norm` aliasing-memory scalar — it is **train-and-exec state local to the env**, never a `GlobalState` field and never on a step()/reset() payload (see FR-ENV-2 leak test).
  - **AC:** tests assert obs image shape `(5,5,5)` at 2×2/3×3/4×4/5×5 and a length-6 scalar vector; `other_visible = 0` when out of radius; `steps_since_seen` resets on opponent-in-view and increments otherwise.
  - **Evidence:** `tests/unit/test_observation.py`, `tests/unit/test_env.py`.
- **FR-ENV-7b (stage-invariant centralized state pad).** The centralized critic/mixer global state from `env.state()` is encoded to a **stage-invariant 77-float footprint** (`3·G·G + 2`, G=`env.view_radius_max·2+1`=5) for **all** curriculum stages, mirroring the obs pad-to-5×5 (FR-ENV-7): cells outside the actual H×W board are **masked** (the out-of-board mask channel is set) so one centralized critic spans 2×2→5×5 with a fixed input dim. This is the train-only `S` encoding (never on the execution path).
  - **AC:** a test asserts `encode_state()` returns a length-77 vector at 2×2/3×3/4×4/5×5; a test asserts out-of-board cells are masked (not arbitrary) at the smaller stages.
  - **Evidence:** `tests/unit/test_qmix_mixer.py`.
- **FR-ENV-8 (dynamic-grid curriculum, Table 2).** The **graded square ladder** is `env.curriculum.stages` = `[[2,2],[3,3],[4,4],[5,5]]` with per-stage cop counts `env.curriculum.num_cops_by_stage` = `[1,1,2,1]` (the 4×4 stage trains a 2-cop team; L4); success-rate-gated promotion. The env is **generic over H≠W** (non-square supported) but the graded ladder is square.
  - **AC:** scheduler promotes on capture-rate ≥ `env.curriculum.promotion_threshold` (0.8) over `env.curriculum.promotion_window` (100); a full ≤25-move episode runs at each graded stage; the generic env also accepts a non-square grid (e.g. 3×2).
  - **Evidence:** `tests/unit/test_curriculum.py`; F6 scaling figure.
- **FR-ENV-9 (no-hardcode gate).** ALL env numerics live in `config/config.yaml`.
  - **AC:** every env numeric resolves from `config/config.yaml` via the config loader (config single-source — verified by the exhaustive dead-config sweep: every algo-relevant key is read or removed); `tests/unit/test_config_loader.py` validates the config contract; `ruff` + code review enforce no inline magic.
  - **Evidence:** `config/config.yaml` + `src/utils/config_loader.py`; `tests/unit/test_config_loader.py`.

### 5.2 MARL Algorithm & CTDE Training (§2.1, §5.2, §7.2)

- **FR-ALG-1 (value decomposition, L2).** Train a value-based CTDE MARL agent for the Cop side using value decomposition. **QMIX is the graded PRIMARY** (`algo.name:qmix`, monotonic mixer eq 7; L2/ADR-D10-A); **VDN is the ablation arm** (eq 6, selected via `algo.mixer.type:vdn`); **IQL is the §7.2 baseline** (`algo.baseline:iql`). The arm is selected via `algo.name` ∈ {qmix, vdn, iql}.
  - **AC:** with `algo.mixer.type=vdn` a unit test asserts `Q_tot == Σ_{i∈cop} Q_i` over the cop team; with `algo.name=qmix` a numeric test on an explicit **N=2 mixer fixture** (two cop Q-streams, a sampled global state) asserts `∂Q_tot/∂Q_i ≥ 0` for **both** inputs and IGM-consistency (eq 5: decentralized greedy joint action == argmax of mixer output). **Optional negative control:** a deliberately non-monotone mixer (negative hypernet weight) must **fail** the `∂Q_tot/∂Q_i ≥ 0` assertion, proving the test has teeth.
  - **Evidence:** `tests/unit/test_mixers.py` (N=2 fixture + optional negative control).
- **FR-ALG-2 (IQL baseline, §7.2).** Implement an IQL baseline (independent recurrent Double-DQN per agent, eq 2/4, no mixer, no global state) for the mandated comparison.
  - **AC:** an `algo=iql` run produces learning curves on the same seeds/obs space as VDN/QMIX; README §7.2 shows the IQL-vs-CTDE non-stationarity contrast (IQL oscillates/diverges where CTDE converges).
  - **Evidence:** F5; `src/marl/learner/learners.py::IqlLearner` (IQL is a no-mixer LEARNER branch — it overrides `_compute_target` for the per-agent eq4 target and drops the mixer + global state; there is NO `iql_mixer.py`).
- **FR-ALG-3 (adversarial boundary, L2).** Cop and Thief NEVER share a cooperative mixer; the Thief is a separate adversarial learner folded into `T(s'│s,ā)`.
  - **AC:** code review + README §7.1/§7.2 document the POSG-vs-Dec-POMDP boundary (eq 3, NEXP^NP); a test asserts no mixer instance receives thief Q-values; two distinct learners + two replay buffers exist.
  - **Evidence:** ADR-D1-B / ADR-D3-A; `tests/unit/test_learners.py`.
- **FR-ALG-4 (shared net + swappable learner/mixer seam, DRY).** "Shared net" means a **shared architecture** (one `RecurrentQNet` class + GRU trunk), not a single shared output head: each role keeps its own action head (`cop` head = 5 logits, `thief` head = 4 logits). All three algorithms share ONE `RecurrentQNet` (GRU, eq 8) and ONE centralized episode replay buffer. VDN↔QMIX differ **only** in the `Mixer` ABC, but **IQL is NOT a mixer swap**: it drops the mixer AND the centralized global state and uses an **independent per-agent target** (a distinct learner branch). So the variation point is the (learner × mixer) seam, not the mixer alone.
  - **AC:** a single `agent_net.py` + buffer is imported by the iql/vdn/qmix paths (no duplication); a test asserts the IQL learner branch carries **no** mixer instance and **no** global-state tensor (it is a learner-level branch, not a `Mixer` subclass).
  - **Evidence:** ADR-D1-C / ADR-D3-C; import graph.
- **FR-ALG-5 (execution = local only).** Execution uses local observation only — no global state, no mixer, no opponent location injected into the policy.
  - **AC:** a test asserts `policy.act()` and the single MCP `request_move` contract (the SAME tool name + path on localhost Stage 1 and cloud Stage 2 — see FR-MCP-3) receive only local obs + legal actions; global state `s` is unreachable on the execution path; import-boundary test forbids runtime imports of `mixer`/`episode_buffer`/`learner`/`GlobalState`/`services`.
  - **Evidence:** `tests/architecture/test_import_boundary.py`; K3.
- **FR-ALG-6 (centralized episodic replay).** Store whole padded episodes with global state; loss is masked (`filled` mask) so padding past termination contributes zero gradient.
  - **AC:** masked-loss test (padded steps → 0 gradient); overfit-one-episode test (loss → ~0).
  - **Evidence:** `tests/unit/test_episode_buffer.py`, `tests/unit/test_learner.py`.
- **FR-ALG-7 (self-play stabilization).** Self-play uses alternating best-response (`selfplay.rounds`=50 best-response rounds per curriculum stage) with frozen-opponent windows (`selfplay.window_k`=1 update-ratio tick) + an opponent pool (`selfplay.pool_size`=5) seeded with a Manhattan heuristic to mitigate non-stationarity/cycling; `selfplay.rounds`×`selfplay.episodes_per_round` = 5000 self-play episodes per stage.
  - **AC:** the trainer logs opponent-pool sampling; fixed-seed evaluation shows non-degenerate (non-cycling) convergence; self-play knobs are read from `selfplay.*` (zero hardcoded).
  - **Evidence:** `src/services/selfplay.py`; F1 learning curves.
- **FR-ALG-8 (losses & optimization).** Centralized Double-DQN target (`algo.double_q:true`) with masked Huber loss (`algo.huber_delta`=1.0); one AdamW per learner with param groups (`algo.lr_agent`=3e-4, `algo.lr_mixer`=5e-4); hard target sync every `algo.target_update_interval` (200) — hard sync is the only mode implemented.
  - **AC:** a fixed-seed short run reproduces a golden metric within tolerance; all hyperparameters live in config (zero hardcoded).
  - **Evidence:** `tests/unit/test_learner.py`; F2 loss curves.
- **FR-ALG-9 (reproducible local training, §5.2).** Training MUST be local and reproducible (single seeded RNG in the buffer + `torch.manual_seed` per learner from each `training.seeds` entry).
  - **AC:** offline run with no network access; golden-metric reproduction.
  - **Evidence:** `experiment_manifest.json`; K5.

### 5.3 OLoRA, Pre-trained Models & Dataset (§5.2, [7])

- **FR-OLoRA-1 (locally pre-trained encoders).** Produce locally pre-trained role encoders (CopNet, ThiefNet) via **behavior cloning** on Manhattan-heuristic data over small grids (2×2/3×3) — the only honest reading of "pre-trained model" that satisfies "training MUST be local".
  - **AC:** `scripts/finetune_ctde.py` (the BC-pretrain stage) runs offline; held-out heuristic-label validation accuracy ≥ the **per-grid** gate `bc.val_acc_gate_by_grid[min(H,W)]` (`{2: 0.50, 3: 0.78}`; honest privileged-expert-vs-local-obs ceilings — see ANALYSIS.md §0) before OLoRA attach; base checkpoints saved under `checkpoints/base/`.
  - **Evidence:** `docs/ANALYSIS.md` §0 (defensible interpretation + 3 rejected readings); BC val-acc log.
- **FR-OLoRA-2 (paper-exact OLoRA, eq 10-11).** Apply OLoRA exactly per [7]: QR-decompose the **pretrained** encoder weight `W0` (eq 3 of [7]); init `B = Q_r[:, :r]` (orthonormal), `A = R_r[:r, :]`; re-base `W0 ← W0 − s·B@A` (eq 4); forward `W_adapted = W0 + s·B@A` (eq 5). Explicitly **reject** QR-of-a-random-matrix init (that is orthonormal-LoRA, not OLoRA) — the rejection is §7.2 grade evidence.
  - **AC:** unit test asserts `‖W_adapted − W0‖ < 1e-5` at init (function-preserving); `Bᵀ@B ≈ I_r` (orthonormality); QR runs on `W0`, not a random matrix.
  - **Evidence:** `tests/unit/test_olora_linear.py`, `test_olora_linear.py`.
- **FR-OLoRA-3 (freeze discipline).** After attach, ONLY adapters `{A,B}`, the plain Q-head, and the mixer are trainable; every encoder `W0` is frozen.
  - **AC:** a test asserts every encoder `W0.requires_grad is False` and optimizer param-groups contain exactly `{A, B, head, mixer}`; the trainable-param count (full vs OLoRA, ~8× fewer; 64×64 @ r=4: 4096→512) is printed for §7.3 evidence.
  - **Evidence:** param-count log; ANALYSIS.md table.
- **FR-OLoRA-4 (scope & rank guard).** OLoRA wraps ONLY encoder Linears (not the Q-head, not the QMIX hypernetwork). Default `rank=4`, sweep `{2,4,8}`, hard-assert `rank ≤ min(m,n)//2`.
  - **AC:** `wrap_encoder` swaps only encoder `nn.Linear` modules; `qr_init` raises on rank violation; head and mixer remain plain.
  - **Evidence:** `tests/unit/test_olora_scope.py`.
- **FR-OLoRA-5 (one encoder spans the curriculum).** One frozen encoder spans the whole 2×2→5×5 curriculum via pad-to-k=5 local patches.
  - **AC:** feeding 2×2/3×3/4×4/5×5 obs yields identical encoder input shape with out-of-range cells masked.
  - **Evidence:** `tests/unit/test_obs_encoder.py`.
- **FR-OLoRA-6 (four §5.2 data sources).** Implement all four §5.2 data sources (Manhattan-heuristic experts, earlier-policy self-play, random-legal coverage, live CTDE transitions) feeding a single `CentralizedReplayBuffer` storing local obs + global state + joint action.
  - **AC:** a `TransitionDTO` schema test passes; `sample()` returns the documented dict with correct shapes.
  - **Evidence:** `tests/unit/test_types.py`.
- **FR-OLoRA-7 (CTDE leakage guard at the type boundary).** `global_state` appears in `TransitionDTO`/replay only; the MCP inference tool signature **cannot** accept it.
  - **AC:** a unit test asserts no `global_state` parameter exists in the inference path and `used_global_state == False` in every response.
  - **Evidence:** `tests/unit/test_mcp_schemas.py`.
- **FR-OLoRA-8 (ablation — stretch, high §7.2 value).** An OLoRA ablation (OLoRA vs full-finetune vs frozen-base-no-adapter [vs standard-LoRA]) is plotted with learning curves + trainable-param counts and an honest verdict.
  - **AC:** `results/` contains the ablation chart; `docs/ANALYSIS.md` states OLoRA is a **stability aid, not the non-stationarity cure**, citing [7] §III and [4], [8].
  - **Evidence:** ablation chart; ANALYSIS.md.

### 5.4 Dual-MCP Architecture (§5.3) — two servers + neutral referee

- **FR-MCP-1 (two independent servers).** Run two independent FastMCP HTTP servers (Cop on `mcp.cop_port` = **8001**, Thief on `mcp.thief_port` = **8002**, Streamable HTTP at `mcp.path` = `/mcp`) as separate OS processes, each owning a private policy and grid position.
  - **AC:** a health ping shows two distinct server processes on the two configured ports (8001/8002); neither imports the other's policy module.
  - **Evidence:** F4 local-comms log; `tests/integration/test_match.py`.
- **FR-MCP-2 (neutral referee = the environment).** A neutral referee process drives all 6 sub-games and is the **sole** holder of ground-truth global state (positions, barriers, capture, clock, scoring).
  - **AC:** an integration test plays a scripted sub-game; the referee adjudicates capture/timeout; neither server can read global state via any tool.
  - **Evidence:** ADR-D5-01; `src/mcp/referee.py`.
- **FR-MCP-3 (local-obs-only execution, single canonical tool contract).** Each server acts using ONLY its referee-supplied local Manhattan-radius observation; no global state and no opponent location enter `request_move`. ONE tool contract is used for BOTH localhost Stage 1 and cloud Stage 2 — the **canonical five tools** are `request_move`, `reveal_location`, `query_opponent`, `new_sub_game`, `send_final_report` (in `src/mcp/`); there is NO divergent `propose_action`/`commit_turn` fork. Every request schema carries a **`session_id`** field (the GRU-keying + `(session_id, tick)` idempotency key — FR-MCP-3b / FR-CLOUD-4). The two location tools are distinct roles, not duplicates: **`query_opponent`** is the *outbound* call one server makes to ask the other for its location (request: `{session_id, requester_role, tick}`); **`reveal_location`** is the *inbound* tool that answers it, **radius-gated** (response: `{visible:bool, position?:(r,c)}`, `{visible:false}` outside `view_radius`). The `query_opponent` result is audit/evidence only and never re-enters `request_move`.
  - **AC:** a unit test asserts the `request_move` input is exactly the `Observation` schema **plus `session_id`**; a test asserts every request schema includes `session_id`; a test asserts the `query_opponent`/`reveal_location` result is **never** passed into the next `request_move`; a test asserts `query_opponent` and `reveal_location` have distinct, registered schemas (asker vs radius-gated responder); a test asserts the same five tool names are registered on the local and cloud server builds.
  - **Evidence:** ADR-D5-02; `tests/unit/test_schemas.py`.
- **FR-MCP-3b (recurrent server-side session state, eq 8).** The decentralized recurrent GRU carries hidden state `z_t = f_φ(z_{t-1}, o_t)` across the ≤25 `request_move` ticks of a sub-game. Each agent server therefore holds **per-sub-game server-side session hidden state** keyed by a `session_id`, **reset on `new_sub_game`**. The agent servers MUST NOT run with `FASTMCP_STATELESS_HTTP=true` (stateless would break GRU continuity); per-session server-side state is the chosen design.
  - **AC:** a test asserts a fresh `new_sub_game` zeroes the session hidden state; a test asserts the hidden state for a given `session_id` is carried (mutated, non-zero) across consecutive `request_move` calls within one sub-game and is isolated per session; the agent servers do not set `FASTMCP_STATELESS_HTTP=true`.
  - **Evidence:** ADR-D5-05; `tests/unit/test_agent_runtime.py`; `tests/integration/test_match.py`.
- **FR-MCP-4 (state/policy hiding).** No tool returns weights, Q-values, logits, hidden state, or the full board; `reveal_location` is radius-gated; no `get_state`/`get_policy` tool exists.
  - **AC:** schema test asserts the `request_move` response has no value/logit/hidden-state field; `reveal_location` returns `{visible:false}` when the requester is outside `mcp.observation.view_radius` (2); no state/policy tool is registered.
  - **Evidence:** ADR-D5-03; `tests/unit/test_schemas.py`.
- **FR-MCP-5 (genuine server-to-server comms, §5.3).** Real cop↔thief HTTP location queries occur each tick (Cop→Thief and Thief→Cop) to satisfy the §5.3 inter-agent-comms requirement; the result is audit/evidence only and is decoupled from the policy input.
  - **AC:** CLI logs show the Cop server calling the Thief's `reveal_location` over HTTP (200 + bearer) and vice-versa.
  - **Evidence:** F4 `results/figures/mcp_comms_local.png` (§7.3d).
- **FR-MCP-6 (revocable bearer auth).** All endpoints require a revocable bearer token; invalid/revoked → HTTP 401.
  - **AC:** integration test asserts 200 with a valid token and 401 with a bad/revoked token. Local Stage 1 = `StaticTokenVerifier`; cloud Stage 2 = JWT (RS256) with `exp` + a `jti` deny-list; per-tool scopes (`position:read`, `game:write`, `report:send`, `admin:revoke`).
  - **Evidence:** `cloud_auth_401.png`; ADR-D5-04 / ADR-D6-2.
- **FR-MCP-7 (full local run, Stage 1).** The system runs end-to-end on localhost (two ports) producing exactly 6 valid sub-games before any cloud deploy.
  - **AC:** `uv run scripts/run_match.py --stage local` completes 6 valid sub-games + emits a referee trace.
  - **Evidence:** P6 exit gate; referee trace.
- **FR-MCP-8 (cop-only single report).** The Cop (and only the Cop) sends exactly ONE Gmail report after the 6th sub-game via `send_final_report`; the Thief has no report tool/credentials.
  - **AC:** an integration test asserts a single report assembled with 6 sub_games + correct totals; the Thief server has no report tool.
  - **Evidence:** `tests/integration/test_match.py`; ADR-D8-5.
- **FR-MCP-9 (technical-loss replay, §3.7).** Technically-failed sub-games do not count and are replayed to reach exactly 6 valid.
  - **AC:** a fault-injection test kills a server mid-sub-game; the referee marks a technical loss and replays; the final report still has exactly 6 valid sub_games.
  - **Evidence:** K4; `tests/integration/test_match.py`.
- **FR-MCP-10 (zero hardcoded ports/URLs/tokens).** All ports/URLs/tokens live in `config.yaml`/`.env`; cloud vs local selected by env (`*_PUBLIC_URL`).
  - **AC:** grep shows no literal ports/URLs/tokens in `src`; one env-switched URL code path.
  - **Evidence:** the CI "Secrets + recipient guard" step (`.github/workflows/ci.yml`); `tests/unit/test_config_loader.py`.

### 5.5 Cloud Deployment (§5.3 Stage 2, §6 step 8, §8)

- **FR-CLOUD-1 (public deploy).** Both MCP servers are deployable to a public cloud host with HTTPS URLs `https://adrl-001-<role>.fastmcp.app/mcp` (primary: Prefect Horizon / FastMCP Cloud; documented + pre-tested Render fallback).
  - **AC:** both URLs return a successful `health()` over the public internet from a machine outside the dev host; `deploy/render.yaml` exists with ≥1 captured successful Render `health()` round-trip.
  - **Evidence:** ADR-D6-1 / ADR-D6-5; `deploy/runbook.md`.
- **FR-CLOUD-2 (cloud-to-cloud comms).** Cop↔Thief comms work cloud-to-cloud with one shared `trace_id` in both servers' logs, using the SAME canonical tool contract as localhost (no divergent fork).
  - **AC:** a 3-turn cloud smoke run logs `request_move` + `reveal_location`/`query_opponent` success in both directions, one shared `trace_id` in both logs.
  - **Evidence:** K6; F4 `_cloud.png`.
- **FR-CLOUD-3 (inference-only, CTDE faithful).** Cloud `request_move` operates on local observation only; deployed model files contain **actor weights only** (no trainer/mixer/replay buffer/global-state code).
  - **AC:** a unit test asserts `request_move` rejects a request containing a `global_state` field; `export_weights()` saves the GRU agent-net `state_dict` only + a shape sidecar.
  - **Evidence:** ADR-D6-3; `tests/unit/test_mcp_schemas.py`.
- **FR-CLOUD-4 (revocable, idempotent, resilient).** Tokens are revocable (jti deny-list primary; key rotation hard lever) and externally minted with a short TTL verified via the `exp` claim — no token is minted in-repo (deploy is inference-only; token VALUES live in `.env`, see ADR-0012); the client uses `mcp.client.timeout_s`=10, `mcp.client.max_retries`=3 with `backoff_s`=0.5, and a pre-game warm-up ping (`prewarm_ping:true`); `request_move` is idempotent per `(session_id, tick)`.
  - **AC:** after revoking a `jti` (or rotating the key) the old token returns 401 while a fresh token returns 200, both screenshotted; a retried move never double-applies.
  - **Evidence:** `cloud_revoke.png`; ADR-D6-6.
- **FR-CLOUD-5 (cloud match emits JSON, does NOT send).** A full 6-sub-game match runs over the two cloud URLs and emits the §3.5 JSON body; it does NOT send the email (that is §6 step 9).
  - **AC:** `uv run scripts/cloud_match.py` produces a valid JSON artifact and sends no mail.
  - **Evidence:** ADR-D6-8 (§6 ordering 8→9); `full_match.json`.

### 5.6 GUI (§5.4, §7.3c) — Pygame god-view spectator

- **FR-GUI-1 (mandatory real-time render).** A Pygame GUI renders the live 5×5 grid (and 2/3/4 sanity sizes) showing cop token, thief token, barriers, and a move/step counter, updating in real time as the two MCP agents play.
  - **AC:** launching `scripts/play.py` over a local referee shows both tokens moving each tick, barriers appearing on placement, and the move counter incrementing 0..`max_moves`.
  - **Evidence:** F3 screenshots; ADR-GUI-01.
- **FR-GUI-2 (spectator purity / partial-observability invariant, §2.1).** The GUI is a strict god-view spectator with **zero** effect on agent observation; it reads referee-owned ground truth, never calls agent MCP servers, and never writes state.
  - **AC:** `tests/architecture/test_gui_purity.py` and `tests/architecture/test_step_no_leak.py` pass — the GUI never imports referee/agent/MCP internals, `SpectatorFrame` is frozen, `referee.build_local_obs` Manhattan-masks beyond `view_radius`, and no agent payload contains global positions/totals/`SpectatorFrame`.
  - **Evidence:** `tests/architecture/`; ADR-GUI-02.
- **FR-GUI-3 (deterministic multi-size screenshots, §7.3c).** Automated, deterministic GUI screenshots at 2×2/3×3/4×4/5×5, each showing cop + thief + ≥1 barrier on a mid-game board.
  - **AC:** `uv run scripts/capture_screens.py` produces `results/screenshots/grid_{2,3,4,5}x{n}.png` headlessly (`SDL_VIDEODRIVER=dummy`); `tests/integration/test_gui_render.py` asserts existence, dimensions, and non-background cop/thief/barrier/HUD pixels.
  - **Evidence:** F3; screenshot matrix fixture table.
- **FR-GUI-4 (HUD & scoreboard, §3.4).** The HUD shows sub-game i/6, move k/25, per-sub-game scores + cumulative totals (Table 1), last actions, and a winner banner on terminal.
  - **AC:** hud/score_view tests render all variants including winner and no-winner.
  - **Evidence:** `tests/unit/test_draw_plan.py`, `tests/unit/test_scorer.py`.
- **FR-GUI-5 (dynamic grid).** One renderer supports dynamic grid sizes with auto-scaled square cells; a mid-match `grid_size` change re-fits the view without distortion.
  - **AC:** `tests/unit/test_gui_logic.py` covers sizes 2–5; `tests/unit/test_gui_logic.py` confirms `GridView` rebuild on `grid_size` change.
  - **Evidence:** `src/gui/transform.py`.
- **FR-GUI-6 (cloud spectator path, Stage 2).** A read-only HTTP spectator path (SSE + polling fallback, **separate** spectator bearer token) lets the GUI watch a distributed cloud match without ever calling agent MCP servers.
  - **AC:** `tests/unit/test_gui_logic.py` confirms in-proc and HTTP paths both yield identical `SpectatorFrame` DTOs; the spectator token is distinct from agent tokens.
  - **Evidence:** ADR-GUI-03.
- **FR-GUI-7 (local rendering literals, CLAUDE.md §4).** Rendering literals (colors, line widths, cell px, fonts, FPS, animation ms) stay LOCAL in `palette.py`; only algo/board params come from config via `SpectatorFrame`.
  - **AC:** `tests/unit/test_gui_logic.py` asserts `palette` has no config import.
  - **Evidence:** ADR-GUI-04.

### 5.7 Gmail Report (§3.5, §5.5)

- **FR-RPT-1 (single end-of-game send).** After the 6th **valid** sub-game, the Cop process sends exactly ONE email to `gmail.to` (`rmisegal+marl@gmail.com`) via the `send_final_report` MCP tool with a structured subject and a JSON body covering all 6 sub-games.
  - **AC:** an integration test with `FakeEmailSender` proves exactly one send occurs and the body validates against the JSON schema `docs/schema/report.schema.json` (loaded by `src/reporting/schema.py`).
  - **Evidence:** `tests/unit/test_report_send.py`; `tests/integration/test_match.py`; ADR-D8-5.
- **FR-RPT-2 (schema conformance, §3.5).** The JSON body conforms to `{group_name, students[]{role/full_name/id}, github_repo, timezone="Asia/Jerusalem", sub_games[6]{id,start,end,moves,winner,scores{cop,thief}}, totals{cop,thief}}`. Per the SOLO model (§1.3), the report-schema `students` array is **`minItems: 1`** (allows 1–3 entries; this submission carries exactly one, role A) — NOT `minItems: 2`.
  - **AC:** a `jsonschema` (draft-2020-12) test passes; the schema declares `students` `minItems: 1`; runtime `validate()` rejects `len(sub_games)!=6`, `ids!=[1..6]`, bad winner, `end<start`, and `totals != Σ scores`.
  - **Evidence:** `docs/schema/report.schema.json` (loaded by `src/reporting/schema.py`); `tests/unit/test_reporting.py`.
- **FR-RPT-3 (scores derived, not supplied, §3.4).** Scores are derived solely from `winner` + `game.scoring` (cop_win=20, thief_win=10, cop_loss=5, thief_loss=5 — the §3.4 REPORT scoreboard, NEVER the RL `reward.*` signal); callers cannot supply scores.
  - **AC:** a unit test asserts `winner=cop → {cop:20, thief:5}` and `winner=thief → {cop:5, thief:10}`; a record call attempting to pass scores is ignored/rejected.
  - **Evidence:** `tests/integration/test_mcp_servers.py`.
- **FR-RPT-4 (timezone-aware timestamps).** Timestamps are timezone-aware Asia/Jerusalem ISO-8601 captured at real sub-game boundaries.
  - **AC:** a unit test asserts `start`/`end` `isoformat(timespec="seconds")` ends with `+03:00` (or `+02:00` DST) and matches the schema regex; naive datetimes raise.
  - **Evidence:** `tests/unit/test_run_log.py`.
- **FR-RPT-5 (idempotent send).** A crash-restart or §3.7 technical-loss rerun never produces a duplicate email (sha256 of canonical report JSON + a sentinel written only after a successful send).
  - **AC:** calling `send_final_report` twice with the same report writes the sentinel once and the second call performs no network send (`FakeEmailSender` call count == 1).
  - **Evidence:** `src/reporting/send.py` (`report_digest`/`already_sent`/`mark_sent`); ADR-D8-4.
- **FR-RPT-6 (no PII / no secrets in tracked files — hard).** Real `full_name`/`id` exist ONLY in the §3.5 email JSON, injected at send time from a git-ignored repo-root source (`players.local.yaml`; placeholders in the tracked repo-root `players.example.yaml`); the App Password lives only in git-ignored `.env` (placeholders in `.env-example`). Cover sheet `adrl-001-ex06.pdf` is git-ignored, Moodle-only.
  - **AC:** a CI deny-list grep over tracked files finds no real surnames, no student-id patterns, no secret values; a redacted (role-only) copy is written to `results/reports/*.redacted.json` with no `full_name`/`id` keys.
  - **Evidence:** the CI "Secrets + recipient guard" step (`.github/workflows/ci.yml`); `src/reporting/send.py` (role-only redaction); ADR-D8-3.
- **FR-RPT-7 (send mechanism + fallback seam).** Send uses `smtplib` + STARTTLS + Gmail App Password by default behind an `EmailSender` Protocol, with a `GmailApiSender` (Google-API OAuth) drop-in fallback and a `FakeEmailSender` for tests/dry-run.
  - **AC:** a mailer test with an injected `smtp_factory` MagicMock asserts `starttls → login → send_message` order + UTF-8 body, with no network/creds; coverage ≥85% with zero live sends.
  - **Evidence:** `tests/unit/test_mailer.py`; ADR-D8-1.
- **FR-RPT-8 (subject template, no hardcoding).** The subject is built from a config template: `[MARL Cops&Robbers] {group_name} | Final {num_games} sub-games | Cop {cop_total} - Thief {thief_total} | {date}` (no names/ids/emails in the subject).
  - **AC:** CI grep fails if `src/` contains the literal `rmisegal+marl` or the score magnitudes as inline literals.
  - **Evidence:** `tests/integration/test_mcp_servers.py`; the CI "Secrets + recipient guard" step.

### 5.8 Academic Analysis, Figures & §9 Bonus Interface (§7, §9)

- **FR-ANL-1 (README is the §7 paper).** `README.md` at the repo root IS the §7 academic paper with a fixed skeleton mapping 1:1 onto §7.1/§7.2/§7.3.
  - **AC:** README contains the Dec-POMDP tuple (eq 1), the POSG caveat (eq 3), the chosen value function (QMIX eq 7 + VDN eq 6 special case), all four §7.2 sub-parts, and figures F1–F6 with §-anchored captions.
  - **Evidence:** README §7; PRD-D10-1.
- **FR-ANL-2 (dual formalism honesty, §7.1).** §7.1 presents BOTH the cooperative Dec-POMDP (shared R, CTDE proxy) AND the general-sum POSG caveat (role-specific `R_cop ≠ R_thief`), naming the proxy as a modeling limitation.
  - **AC:** both formalisms present; the deliberate collapse is justified with the licence-CTDE rationale.
  - **Evidence:** README §7.1; ADR-D10-D.
- **FR-ANL-3 (non-stationarity contrast, §7.2).** §7.2 contrasts the independent IQL target (eq 2 ≡ eq 4) against the centralized CTDE target side-by-side, including the non-stationarity equation `P_i(s'│s,a_i)=Σ_{a_-i} π_-i·T`.
  - **AC:** both target equations + the drift explanation appear; the `eq2≡eq4` cross-reference is footnoted.
  - **Evidence:** README §7.2; R13.
- **FR-ANL-4 (empirical IQL-vs-CTDE baseline, F5).** The §7.2 baseline is EMPIRICAL (figure F5), using identical nets/replay/seeds/curriculum so the mixer is the only difference. The credit-assignment study (K1) on the **4×4 2-cop** panel (genuinely multi-agent mixer) shows the **honest empirical ordering `VDN ≥ IQL > QMIX`** — the `QMIX ≥ VDN ≥ IQL` hypothesis is FALSIFIED at the 50-round budget (QMIX's monotonic mixer destabilizes, R1; more-expressive-but-harder-to-train), reported faithfully per README §7.2. The graded 1-cop **5×5 stage was NOT swept** (excluded from the figure matrix for compute — README §7.3), so F5 carries only the 4×4 focus panel; the K2 cop-quality result + the N=1 trivial-decomposition discussion are therefore scoped to 4×4 / prose (§7.2), NOT a 5×5 figure.
  - **AC:** F5 shows IQL vs VDN vs QMIX with final capture-rate + cross-seed variance on the **4×4 2-cop focus panel** (the 5×5 1-cop stage is not swept — README §7.3).
  - **Evidence:** `results/figures/baseline_comparison.png` (4×4 focus panel); K1.
- **FR-ANL-5 (IGM lossy-decomposition critique, §7.2).** §7.2 discusses IGM/QMIX monotonicity (eq 7) as a lossy decomposition with the non-monotonic (pincer) example, naming QPLEX [10] + Weighted-QMIX [9] as fixes (exact arXiv IDs) and reproducing L10 Table 3; with the honest N=1 cop-mixer "trivial-decomposition" caveat (L4).
  - **AC:** monotonicity constraint stated, expressiveness limit explained, both fixes cited, Table 3 reproduced.
  - **Evidence:** README §7.2; ADR-D3-C.
- **FR-ANL-6 (all six §7.3 figures).** F1 (both-agent learning curves), F2 (loss curves per stage), F3 (GUI at 2×2/3×3/4×4/5×5), F4 (MCP-comms proof, localhost canonical), F5 (IQL/VDN/QMIX baseline), F6 (scale effect) all exist under `results/figures/` and are referenced in README.
  - **AC:** 6 non-placeholder figures present.
  - **Evidence:** `results/figures/`.
- **FR-ANL-7 (reproducible figures).** The **plotted** figures **F1/F2/F5/F6** regenerate from one command `uv run python -m src.results.make_figures` reading `results/runs/*.jsonl`, with a `results/figures/experiment_manifest.json` recording runs, arms, stages, seeds, and the config hash. **F3 (GUI screenshots) and F4 (MCP-comms proof) are NOT plotted from JSONL** — they are deterministically **captured** by their seeded scripts (`scripts/capture_screens.py` for F3; the redacted cop↔thief comms log for F4) and reproduce via those scripts, not `make_figures`.
  - **AC:** rerunning `make_figures` reproduces the committed F1/F2/F5/F6; rerunning the F3/F4 capture scripts reproduces those artifacts deterministically; the manifest is complete.
  - **Evidence:** K5; PRD-D10-7.
- **FR-ANL-8 (risk register).** A risk register (≥15 risks, P×I, mitigation + fallback + owner-phase) is in `docs/PLAN.md` and summarized in README §8.
  - **AC:** register covers non-convergence, cloud/MCP flakiness, Gmail send, PII, Drive-`.git`, 150-LOC, figure-drift, OLoRA, citation-numbering.
  - **Evidence:** `docs/PLAN.md` §8.
- **FR-ANL-9 (§9 bonus interface spec — schema only, coordination dependency).** Ship a `bonus_report` JSON serializer + an inter-group MCP interface spec (the cross-server tool contract a partner `biu-rlNN` group's opposing server must implement). The bonus is OFF the A6 critical path (counts toward the final project); the actual match needs a partner group.
  - **AC:** the bonus schema includes `report_type:"bonus_game"`, `groups{group_1,group_2}`, `github_repo_group_1/2`, `students_group_1/2`, `totals_by_group`, `bonus_claim`, `mutual_agreement`; each bonus `sub_games[]` entry **KEEPS the §3.5 keys** `{id,start,end,moves,winner,scores{cop,thief}}` **AND adds** `cop_group` + `thief_group` (per ex06 §9.4) — the §3.5 fields are NOT dropped; the §3.5 email serializer is reusable for the bonus variant; the bonus subject string is `[MARL Cops&Robbers BONUS] {group_1} vs {group_2} | Final {num_games} sub-games | {group_1} {total_1} - {group_2} {total_2} | {date}`; the interface spec is published in `docs/` and marked as a **coordination dependency, not buildable solo**.
  - **Evidence:** `docs/schema/report.schema.json` (extensible); §9 interface spec doc; ADR-D8 §13.
- **FR-ANL-10 (§9 bonus rules — role alternation + scoring).** The §9 inter-group match is 6 sub-games with **role alternation**: sub-games **1–3 = group-1 Cop vs group-2 Thief**; sub-games **4–6 = swapped** (group-2 Cop vs group-1 Thief). Bonus scoring: **winning group +10, losing group +7, tie → +5 each**; the bonus counts **only if BOTH groups send a valid bonus email AND `mutual_agreement: true`** (otherwise both groups score 0). README §9 documents group names, final bonus scores, and screenshots.
  - **AC:** a serializer/validator test asserts `cop_group`/`thief_group` alternate per the 1–3 / 4–6 pattern; a scoring test asserts winner=+10, loser=+7, tie=+5 each; a gate test asserts the bonus is voided (0/0) unless both `bonus_claim` present and `mutual_agreement==true`.
  - **Evidence:** §9 interface spec doc; `tests/unit/test_reporting.py`; ADR-D8 §13.

### 5.9 V3 Cross-Cutting Requirements (§5 egress, §9 sensitivity, §11 cost, §13 quality)

These FRs cover the V3 sections that flip from N/A→REQUIRED for A6 (external API calls + mandatory GUI) and the §9/§11/§13 excellence sections.

- **FR-API-1 (§5 single ApiGatekeeper — flips N/A→REQUIRED).** A6 makes real runtime HTTP (peer-MCP) + Gmail + Prefect-deploy calls, so **ALL outbound egress** routes through one central `ApiGatekeeper` (`src/api/gatekeeper.py`) exposing `execute(call)` AND `get_queue_status()`. It applies a **per-channel token-bucket** read from the versioned, git-tracked `config/rate_limits.json` (channels `peer_mcp` 120/min burst 10, `gmail` 5/min burst 1, `prefect_deploy` 30/min burst 5), a **FIFO overflow queue** (`overflow_policy:"fifo_queue"`, `on_overflow:"enqueue"`, `max_queue:256` — **no crash on burst**), and logs every call (`log_all_calls:true`). When the FIFO overflow queue **itself** is full (depth == `max_queue`), the gatekeeper **rejects the call with an error and logs it** (it does **not** crash and does **not** silently drop) — the bounded queue degrades gracefully. No secrets live in `rate_limits.json` (tokens in `.env`).
  - **AC:** `test_egress_via_gatekeeper.py` asserts every peer-MCP/Gmail/Prefect call path goes through `ApiGatekeeper.execute`; a burst test exceeding `burst` enqueues (does not raise) and `get_queue_status()` reflects the depth; a test filling the queue to `max_queue`+1 asserts the gatekeeper **returns/raises a handled rejection error AND emits a log line** (no crash, no silent drop); limits are read from `config/rate_limits.json` (zero hardcoded); a grep asserts no direct `httpx`/`smtplib` egress bypasses the gatekeeper.
  - **Evidence:** `tests/architecture/test_egress_via_gatekeeper.py`; `config/rate_limits.json`; ADR-0006; NFR-9.
- **FR-RES-1 (§9 sensitivity analysis — distinct from the IQL/VDN/QMIX ablation).** A controlled **single-parameter sweep** documents the swept value's effect on capture-rate with a figure. The pinned swept parameter is **`env.view_radius_by_grid`** (other candidates `olora.rank` / `selfplay.window_k` are acceptable, all other config keys held fixed). **`reward.distance_weight` is NOT a valid sweep parameter** — potential-based shaping is provably **policy-invariant** (Ng 1999), so varying it cannot change the converged policy/capture-rate and would be a meaningless §9 sweep. This is the §9 **sensitivity** requirement — the algorithm comparison (FR-ANL-4) is an **ablation**, NOT the sensitivity sweep.
  - **AC:** `results/figures/sensitivity_view_radius.png` plots `env.view_radius_by_grid` against capture-rate over fixed seeds; `docs/ANALYSIS.md` (or README §7.2) documents the observed effect; only ONE parameter varies per sweep; no sweep uses `reward.distance_weight`.
  - **Evidence:** `results/figures/sensitivity_*.png`; ANALYSIS.md; F-sensitivity.
- **FR-RES-2 (§9 analysis notebook — SDK-only).** Ship `notebooks/analysis.ipynb` that consumes the **single `MarlSDK` only** (no direct internal imports), rendering the LaTeX Dec-POMDP/QMIX equations (eq 1, eq 7), the F1–F6 figures, and the `[1]`–`[11]` citations.
  - **AC:** the notebook imports only `src.sdk`; it regenerates/embeds the §7.3 figures and the formal equations; `tests/architecture/test_import_boundary.py` covers the notebook's import surface.
  - **Evidence:** `notebooks/analysis.ipynb`; NFR-7.
- **FR-COST-1 (§11 cost analysis).** Ship `docs/COST_ANALYSIS.md` with (a) the **AI-assisted-development token-cost breakdown** (input/output token counts, $/1M per model, total), (b) the **RL training-compute envelope** (episodes × stages × wall-clock, local-only — `training.episodes_per_stage`=5000 × `env.curriculum.stages` (4) × seeds (5)), and (c) optimization notes (OLoRA param reduction, local-only training, caching).
  - **AC:** `docs/COST_ANALYSIS.md` exists with all three parts and numeric tables; `test_required_docs_present.py` asserts its presence.
  - **Evidence:** `docs/COST_ANALYSIS.md`; ADR-0012.
- **FR-QUAL-1 (§13 ISO/IEC 25010).** Ship `docs/QUALITY.md` mapping ALL EIGHT ISO/IEC 25010 characteristics — Functional Suitability, Performance Efficiency, Compatibility, Usability, Reliability, Security, Maintainability, Portability — to concrete A6 mechanisms/evidence.
  - **AC:** `docs/QUALITY.md` addresses all 8 characteristics (each with an A6 mechanism + evidence pointer); `test_required_docs_present.py` asserts its presence.
  - **Evidence:** `docs/QUALITY.md`; ADR-0013.
- **FR-UX-1 (§10 Nielsen heuristics — flips N/A→REQUIRED via the mandatory GUI).** Ship `docs/UX.md` mapping all 10 Nielsen heuristics to the Pygame spectator GUI with a screenshot per state.
  - **AC:** `docs/UX.md` covers all 10 heuristics + a screenshot per screen/state; `test_required_docs_present.py` asserts its presence.
  - **Evidence:** `docs/UX.md`; ADR-0010; NFR-10.
- **FR-COMP-1 (§16 component contract — Input/Output/Setup + input validation).** Every public component documents its **Input / Output / Setup** in its module docstring (matching the DI + `_validate_config()` discipline already specified). In addition, the **MCP tool handlers** (`request_move`, `reveal_location`, `query_opponent`, `new_sub_game`, `send_final_report`) and the **report builder** expose a `_validate_input()` that raises **`TypeError`** on a malformed/typed-wrong payload (distinct from the value-level `_validate_config()`).
  - **AC:** a test asserts each MCP tool handler and the report builder has a `_validate_input()` that raises `TypeError` on a wrong-typed payload (e.g. non-dict observation, missing `session_id`); each documents Input/Output/Setup in its docstring.
  - **Evidence:** `tests/unit/test_mixers.py`; module docstrings.

---

## 6. Non-Functional Requirements

### 6.1 V3 hard gates → A6 mapping (NFRs)

| ID | Gate | A6 mechanism | Acceptance / Evidence |
|---|---|---|---|
| **NFR-1** | **≤150 LOC/file** (excl. blanks/comments) | `scripts/check_file_sizes.py`; split servers/tools, mixer/q_net, env modules | runs **AFTER** ruff format (MEMORY: file-size gate after format); the CI "File-size gate" step |
| **NFR-2** | **≥85% coverage** | DI + mocked `httpx` peer & Gmail; pure gatekeeper/builder/env unit tests | `pytest --cov=src --cov-fail-under=85`; the CI "Tests + coverage" step |
| **NFR-3** | **Ruff 0 violations** | `ruff check` + `ruff format --check`; `PLR2004` ignored in tests only | the CI "Ruff lint" + "Ruff format check" steps |
| **NFR-4** | **No hardcoded values → config** | all §3.6 params + ports + URLs + hyperparams in `config/config.yaml` (config single-source — verified by the exhaustive dead-config sweep: every algo-relevant key is read or removed); `ruff` + code review | `tests/unit/test_config_loader.py`; `config/config.yaml` |
| **NFR-5** | **No secrets + `.env-example` only** | tokens/OAuth/App-Password/PII in `.env`; `.env-example` committed (names only); `.gitignore` covers `.env`, `*.pem`, `*.key`, `credentials*.json`, `secrets/` | the CI "Secrets + recipient guard" step (`.github/workflows/ci.yml`) |
| **NFR-6** | **uv-only** | CI uses `uv`; no pip/conda; `uv.lock` committed; `uv sync --frozen` | the CI "Sync deps" step |
| **NFR-7** | **Single SDK entry for UIs** | `src/sdk/sdk.py::MarlSDK` is the only business-logic entry; GUI/MCP/report/scripts import only `src.sdk` (scripts exempt from the single-entry rule, NOT from size/lint) | `tests/architecture/test_import_boundary.py`, `tests/architecture/test_train_exec_split.py` |
| **NFR-8** | **Version starts at stated version** | `__version__ = "1.0.0"` in `src/__init__.py` == `config.version` (3-segment mapping of V3 "1.00", A5-accepted) | `tests/unit/test_config_loader.py`; ADR-0011 |
| **NFR-9** | **§5 External-API governance (flips N/A→REQUIRED)** | A6 makes real runtime HTTP (peer MCP) + Gmail + Prefect-deploy calls → `src/api/gatekeeper.py::ApiGatekeeper` (`execute` + `get_queue_status`; per-channel token-bucket from versioned `config/rate_limits.json`; FIFO overflow queue `max_queue:256`, no crash; all calls logged) — see FR-API-1 | `tests/architecture/test_egress_via_gatekeeper.py`; ADR-0006 |
| **NFR-10** | **§10 Nielsen heuristics (flips N/A→REQUIRED)** | GUI is now mandatory → `docs/UX.md` maps all 10 heuristics with a screenshot per state | `tests/architecture/test_required_docs_present.py`; ADR-0010 |
| **NFR-11** | **Docs-first SDLC (PRD/PLAN/TODO + per-area PRDs + ADRs)** | this PRD + `docs/PLAN.md` (C4/UML/deployment + ADRs **0001–0014**) + `docs/TODO.md` + the four per-area `docs/prd/PRD-{ENV,CTDE,MCP,REPORT}.md` exist with human §1.4 sign-off before feature code | `test_required_docs_present.py` enumerates `docs/prd/PRD-ENV.md`, `PRD-CTDE.md`, `PRD-MCP.md`, `PRD-REPORT.md` (+ COST_ANALYSIS/QUALITY/UX/PROMPTS) + ADR-count check (0001–0014) |

### 6.2 Other non-functional requirements
- **NFR-12 (partial-observability invariant — structural, not by convention).** The train-global / exec-local split is enforced **structurally**: the mixer/critic/replay/global-state code is never importable on the execution path, and only the agent-net `state_dict` is exported. Enforced by import-boundary + MCP-schema leak tests (K3).
- **NFR-13 (reproducibility).** Fixed `training.seeds` `[7,17,37,71,107]`; single seeded RNG in the replay buffer; `torch.manual_seed` per learner from each `training.seeds` entry; `results/figures/experiment_manifest.json` (`paths.experiment_manifest`) pins seeds/config-hash/git-commit so README numbers cannot drift from code (R8).
- **NFR-14 (resilience).** Cloud client `mcp.client.timeout_s`=10, `max_retries`=3 with `backoff_s`=0.5 backoff, pre-game warm-up ping (`prewarm_ping:true`); `request_move` idempotent per `(session_id, tick)`; server unreachable mid-sub-game → technical loss + replay (§3.7).
- **NFR-15 (process isolation).** Servers and referee are separate OS processes (not threads) to avoid FastMCP-async / Pygame / torch event-loop deadlock; the GUI consumes referee results via a queue and never calls servers directly.
- **NFR-16 (secrets redaction).** Unconditional redaction in `src/results/comms.py` + `src/reporting/send.py` strips `Authorization`/`Bearer`/key material from logs before any screenshot.
- **NFR-17 (Google-Drive `.git` hazard, MEMORY).** Course repos live in Google Drive; sync can silently remove `.git` mid-session — push often; keep a `/tmp` clone to restore `.git`.
- **NFR-18 (CI never gates on git-ignored PII, MEMORY).** No test asserts the git-ignored cover sheet exists — any cover-sheet check is **skip-when-absent** by design; never gate CI on a git-ignored PII artifact (this exact assert broke A5 CI).

---

## 7. User Stories

1. **As the architect (human, §1.4),** I approve the PRD/PLAN/ADRs *before* any feature code lands, so every AI-generated change traces to a signed-off spec.
2. **As the Cop agent,** I receive only my local Manhattan-radius observation each turn and return a single action, so my policy is decentralized and faithful to the Dec-POMDP O-function.
3. **As the Thief agent,** I evade capture for 25 moves using only my local view, trained adversarially by self-play, so the env is genuinely non-stationary for the Cop learner.
4. **As the referee (the environment),** I hold ground-truth state, feed each agent its masked obs, adjudicate capture/timeout/barriers, and replay technical losses to reach exactly 6 valid sub-games.
5. **As a researcher,** I run one seeded sweep (IQL/VDN/QMIX × seeds × the 2×2→5×5 ladder) and regenerate all six figures from one command, so my §7 numbers never drift from code.
6. **As a spectator (grader/contributor),** I launch the Pygame GUI and watch the live pursuit at any grid size, and capture deterministic screenshots headlessly for §7.3c.
7. **As the lecturer (`rmisegal`),** I receive exactly one Gmail at `rmisegal+marl@gmail.com` with a schema-valid JSON of all 6 sub-games, and I can hit the live cloud URLs (or view the §7.3d screenshots) to verify distributed MCP comms.
8. **As a DevOps/grader,** I confirm the V3 hard gates from CI evidence and the `software-excellence-v3` audit doc, with no PII or secrets in any tracked file.
9. **As a partner group (`biu-rlNN`, future/final-project),** I read the published inter-group MCP interface spec and point my opposing server at our cloud URLs to play the §9 bonus match.

---

## 8. Assumptions & Constraints

### 8.1 Assumptions (validate before locking)
- **A1.** The graded 5×5 match is literal 1-cop-vs-1-thief (§3); `num_cops≥2` is a 4×4 training scenario only (L4). VDN/QMIX degenerate gracefully to single-agent at N=1 (documented §7.2 caveat).
- **A2.** `PLACE_BARRIER` counts against the 25-move budget (placement IS the move, §3.3).
- **A3.** Potential-based shaping during training is acceptable to the grader because it is provably policy-invariant; the official scoreboard is computed separately (L3, ADR-002).
- **A4.** The sender Gmail is a personal account with 2FA + App Passwords enabled (else fall back to the `GmailApiSender` OAuth path via the `EmailSender` Protocol; confirm before locking — ADR-D8-1).
- **A5.** Graders accept a neutral referee as "the environment" (not a disallowed third agent) given that real cop↔thief HTTP location calls still occur each tick (ADR-D5-01; fallback = cop-driven topology documented).
- **A6.** Localhost Stage 1 is the gradable baseline; cloud Stage 2 is incremental upside, never an A6 gate (§5.3 localhost-first, R2).
- **A7.** Totals count only the 6 final valid sub-games (technical losses redone per §3.7), not all attempts.
- **A8.** Group `adrl-001` is a SOLO submission (role A; §1.3); the schema still allows 1–3 students (`minItems:1`). Real names/IDs live only in git-ignored repo-root `players.local.yaml` (placeholders in tracked repo-root `players.example.yaml`).

### 8.2 Constraints
- **C1.** Same V3 software-excellence guidelines govern grading; late penalty −5 pts/24h.
- **C2.** Identical GitHub repo URL in both members' Moodle submissions; `rmisegal@gmail.com` added as a read collaborator; branch `assignment-6`, role-A/role-B PR attribution, no direct `main` pushes.
- **C3.** Cover sheet `adrl-001-ex06.pdf` (Word template, fill fields only, save as PDF) is git-ignored, Moodle only.
- **C4.** The §9 inter-group bonus is a **coordination dependency** (needs a partner `biu-rlNN` group); the cross-group match counts toward the final project, not A6 — schema + interface spec ship now; the match defers.
- **C5.** Cloud is **inference-only**; training MUST be local (§5.2). Mixer/critic/replay/global-state never reach a deployed checkpoint.
- **C6.** Pin `fastmcp` (run the import check first — v2 vs v3 moved `BearerAuthProvider`/`RSAKeyPair`); verify `transport="http"` (Streamable HTTP, not deprecated SSE) before coding auth.

---

## 9. Milestones (strict §6 order, each with an EXIT GATE)

Build proceeds in strict §6 priority order; no phase's deliverable lands before its prerequisite phase's exit gate (PRD-D10-8). Owners per the role split (A = env + CTDE training + plots; B = MCP + GUI + Gmail + integration).

| Phase | §6 step | Deliverable | Exit gate |
|---|---|---|---|
| **P0** | pre | Scaffold repo tree; CLAUDE.md (+§1.4 table); `pyproject.toml` (`version="1.0.0"`, uv groups dev/mcp/mail); CI (ruff → format → file-size-after-format → check_* → cov 85); this PRD + PLAN + TODO + ADRs **0001–0014**; share repo read with `rmisegal@gmail.com` | docs reviewed + human §1.4 sign-off; `uv sync` green; CI skeleton |
| **P1** | Theory first | Formal Dec-POMDP tuple (eq 1) + POSG caveat (eq 3); S/A/R/Ω/O/γ encoded as config constants; reward fn unit-tested vs §3.4; README §7 outline naming every figure/experiment | tuple doc + reward tests pass; README skeleton present |
| **P2** | Basic pursuit | `CopsRobbersEnv`: dynamic grid (5×5 default), ≤25 moves, capture §3.2, barriers ≤5 §3.3, local-obs crop, simultaneous resolution | game-logic tests ≥85% cover; 2×2 capture works; leak-test green |
| **P3** | Minimal pipeline | End-to-end collect→train→export pipeline + **throwaway tabular 2×2 smoke** (Table 2 row 1, deleted in P4) + dataset/replay machinery (centralized episodic replay buffer, BC dataset/obs-encoder, Manhattan-heuristic oracles). **BC training (per-grid val-acc gate `bc.val_acc_gate_by_grid`) is P4/T4.4 — NOT P3.** | 2×2 reaches **pursuit-optimal capture from every 2×2 start over all `training.seeds`** (thief flees ⇒ naive min-Manhattan wrong; a zeroed Q fails the gate); pipeline runs NaN-free; headless `render_state()` god-view JSON + minimal sub-game JSON (GUI shot → P7) |
| **P4** | Network + training | Recurrent GRU Q (eq 8); VDN→QMIX mixer (eq 6→7); centralized episodic replay; IQL baseline (eq 2/4); OLoRA attach (eq 10-11); self-play harness; seeded sweep → `results/runs/*.jsonl` + manifest; 3×3 hyperparameter pass | loss decreasing; 3×3 converges; sweep logs present |
| **P5** | MCP protocol | Two MCP servers (cop, thief) + neutral referee; HTTP tool contracts; bearer auth; egress via gatekeeper | localhost: cop queries thief via MCP; F4 local logs captured; 401 on revoked token |
| **P6** | Full local run | Whole 6-sub-game match over localhost MCP; tally; assemble §3.5 JSON (no send); technical-loss replay | 6 valid sub-games; totals match; JSON validates |
| **P7** | GUI | Real-time Pygame board, barriers, step counter, HUD/scoreboard | F3 screenshots at all grid sizes; spectator-purity + no-leak tests green |
| **P8** | Cloud-MCP | FastMCP + Prefect Horizon deploy both servers; public URLs + revocable JWT; Render fallback smoke-tested | distributed match runs (≥1 valid cloud sub-game; shared trace_id); cloud auth/revoke proofs captured (upside per R2) |
| **P9** | Gmail report | Cop sends ONE end-of-game JSON email to `rmisegal+marl@gmail.com` (idempotent; PII from secrets only) | email received; JSON validates vs §3.5; redacted copy committed |
| **P10** | results | Run the full sweep; generate F1–F6 + the §9 sensitivity figure (FR-RES-1); write README §7 academic body; `notebooks/analysis.ipynb` (SDK-only, FR-RES-2); `docs/COST_ANALYSIS.md` (§11), `docs/QUALITY.md` (§13 ISO-25010), `docs/UX.md` (§10 Nielsen) | all 6 figures + sensitivity figure present; numbers reproducible from `runs/`; risk register in PLAN + README §8; COST/QUALITY/UX docs present |
| **P11** | gate | `software-excellence-v3` audit; self-grade; freeze 24h pre-deadline; produce git-ignored cover sheet (Moodle only) | all hard gates ✅; audit doc with real evidence; ADR count 0001–0014; `test_required_docs_present.py` green — it enumerates the four `docs/prd/PRD-{ENV,CTDE,MCP,REPORT}.md` **plus** COST_ANALYSIS/QUALITY/UX/PROMPTS present; never assert the git-ignored cover sheet in a test |
| **P-bonus** | §9 post-A6 | Inter-group interface spec published; `bonus_report` serializer (keeps §3.5 sub-game keys + adds `cop_group`/`thief_group`; role alternation 1–3/4–6; scoring +10/+7/+5; FR-ANL-9/FR-ANL-10); (match deferred — coordination dependency with a partner `biu-rlNN` group) | schema + spec present; match marked not-attempted unless a partner group + both `bonus_claim` + `mutual_agreement:true` |

**Load-bearing order:** P10 needs P4 (training) + P6 (orchestration); F4 needs P5; F3 needs P7; the §3.5 email (P9) needs the P6 tally; the F6 scale figure needs the P4 sweep across the full ladder.

---

## 10. Cross-Cutting Dependencies (digest)

- **Env → Algorithm:** env publishes obs/state dims, the view-radius schedule, channel layouts, action enums, masking semantics, and a `state()` method distinct from `step()`; the centralized replay-tuple `(obs_dict, joint_action, global_reward, reward_i, next_obs_dict, next_legal_mask, terminated, GlobalState)` lets CTDE stabilize against non-stationarity. The **`next_legal_mask`** is stored so the Double-DQN target can additively `−inf`-mask the next-state `argmax`/`max` (FR-ENV-3) — recomputing legality from `next_obs_dict` alone is unsafe (barrier-budget / per-role legality is not fully recoverable from the cropped obs). Net input dim = padded-max `W_v` (5×5) so one net spans the curriculum + OLoRA.
- **Algorithm → MCP/Cloud:** exports `cop_weights.pth` / `thief_weights.pth` = GRU agent-net `state_dict` only + a shape sidecar; the server loads `.eval()`, forward-pass only, returns `{action, used_global_state:false}`; forbidden runtime imports (mixer/buffer/learner/`GlobalState`/services) are part of the CTDE contract; the server persists per-sub-game GRU hidden state across calls.
- **MCP → GUI:** the GUI consumes the referee trace/result queue, never calls MCP servers directly (event-loop isolation).
- **MCP/Cloud → Report:** `cloud_match.py` (and the local referee) emit the §3.5 JSON body; the Gmail dimension consumes and sends; the Cop-only `send_final_report` tool fires exactly once.
- **Config/SDK (D9):** owns `config.yaml`, the single `MarlSDK` entry, `.env`/`.env-example`, the ≤150-LOC split, the `fastmcp` pin, the `ApiGatekeeper`, and the CI/arch-test harness that enforces the cross-cutting V3 contract on every dimension.
- **Analysis (D10):** owns the README §7 body, the figure pipeline, the phase gates, and the risk register; it CONSUMES env/training/MCP/GUI/email contracts and must not re-specify their internals.

---

## 11. Open Decisions (require human §1.4 sign-off before the gated phase)

These remain open per the syntheses; the recommended default is listed but the architect must confirm before the named phase.

- **OD1 (P2, ADR-001):** Move resolution = simultaneous + swap=capture (recommended) vs cop_first/thief_first. *Non-delegable rule decision.*
- **OD2 (P2, ADR-002):** Confirm potential-based shaping during training (off at eval) is acceptable to the grader; thief reward = shaped survival vs pure zero-sum.
- **OD3 (P4):** Ship QMIX as a built-and-tested deliverable (recommended per L2) — confirm the time budget; VDN+IQL are the guaranteed floor.
- **OD4 (P4): RESOLVED.** The `mixer.include_opponent_utility` lever was **dropped** (a monotone dependence on the adversary's utility would risk the forbidden cop/thief mixer boundary, ADR-0006). The genuinely-2-input mixer evidence for §7.2 comes from the **4×4 2-cop scenario** (K1) instead — clean per-team QMIX everywhere, honest N=1 caveat on the graded 5×5.
- **OD5 (P4):** OLoRA ablation REQUIRED vs stretch; standard-LoRA as a 4th arm; scale `s` default 8 vs paper's 16.
- **OD6 (P8, ADR-D6-1):** Cloud platform Prefect Horizon (recommended) vs Render; whether the grader hits live URLs or only views screenshots (affects token-revoke timing).
- **OD7 (P9, ADR-D8-1):** Sender account type (personal vs Bar-Ilan Workspace) — confirm App Passwords available before locking `smtplib`; confirm subject wording vs any cover-sheet-fixed string; confirm inline JSON vs `.json` attachment.
- **OD8 (P-bonus, C4):** Whether to pursue the §9 bonus and whether a partner `biu-rlNN` group is identified — determines whether to build the bonus serializer now (recommended: schema now, match deferred) or defer entirely.
- **OD9 (D9):** Role A/B work-stream split + real names/IDs for `.env`/ repo-root `players.local.yaml` (drives commit attribution + `students[]` order; SOLO → one entry, §1.3).

---

## 12. Glossary

- **CTDE** — Centralized Training, Decentralized Execution.
- **Dec-POMDP** — Decentralized Partially Observable MDP (cooperative, shared R).
- **POSG** — Partially Observable Stochastic Game (general-sum, role-specific R).
- **IGM** — Individual-Global-Max consistency principle (eq 5).
- **VDN / QMIX** — additive (eq 6) / monotonic-mixer (eq 7) value decomposition.
- **IQL** — Independent Q-Learning (the §7.2 non-stationarity baseline).
- **OLoRA** — Orthonormal Low-Rank Adaptation (QR-decomposition PEFT, [7]).
- **MCP** — Model Context Protocol ([11]); each agent exposes its own server.
- **Referee** — neutral orchestrator = "the environment"; sole holder of global state `s`.
- **SDK** — `src/sdk/sdk.py::MarlSDK`, the single business-logic entry for all UIs/servers/scripts.

---

## 13. Bibliography (§10)

`[1]` Bernstein et al. 2002 — Dec-POMDP complexity. `[2]` Sunehag et al. 2018 — **VDN** (1706.05296). `[3]` Rashid et al. 2018 — **QMIX** (1803.11485). `[4]` Amato 2024 — Cooperative MARL intro (2405.06161). `[5]` Lin et al. 2025 — cooperative pursuit-evasion + curriculum (MDPI Electronics). `[7]` Büyükakyüz 2024 — **OLoRA** (2406.01775). `[8]` Foerster et al. 2018 — **COMA** (1705.08926). `[9]` Rashid et al. 2020 — **Weighted QMIX** (2006.10800). `[10]` Wang et al. 2021 — **QPLEX** (2008.01062). `[11]` Anthropic 2024 — **MCP spec** (modelcontextprotocol.io).

> **Note:** MAPPO (optional §7.2 contrastive arm, L2) is **not** in the brief bibliography; if implemented, README adds a bibliography entry.
