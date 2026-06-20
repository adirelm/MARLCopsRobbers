# TODO — Assignment 6: MARL Cops & Robbers (Cloud-MCP, Vibe Coding)

> **Group:** `adrl-001` · **Version:** `1.0.0` · **Roles:** A, B (no optional C) · **V3 software-excellence rubric.**
> **Ground truth:** `planning/BRIEF.md` (ex06.pdf §0-§10) + `L10-MARL.pdf`. Bibliography `[1]`-`[11]` per BRIEF §10.
> **Companion docs:** `docs/PRD.md` (requirements), `docs/PLAN.md` (architecture/ADRs), `docs/THEORY.md`, `docs/ANALYSIS.md`, `docs/UX.md`, `README.md` (the §7 academic paper).
>
> This file is the **phased task list**, organized by the **BRIEF §6 development priority order** (P0 foundation → P11 gate, + P-bonus).
> Every task has: **ID**, **owner-role (A/B)**, **status checkbox**, and a **Definition of Done (DoD)**. The aggregate
> **Definition of Done** and the **V3 hard-gate checklist** live at the end.

---

## Locked architect decisions (human-decided, CLAUDE.md §1.4 — DO NOT re-open without sign-off)

These are settled (the user chose "maximize the grade; complexity/time acceptable"). Tasks below assume them.

1. **Scope = full ambition.** MARL+CTDE + **two FastMCP servers (localhost AND cloud)** + Pygame GUI + Gmail report + **full OLoRA ablation**. The §9 bonus-capable report schema + an **inter-group MCP interface spec** are built; the actual cross-group match is a **coordination dependency** (needs a partner `biu-rlNN` group) — not buildable solo, tracked as P-bonus.
2. **Algorithm = QMIX PRIMARY** (monotonic mixer, eq 7) + **VDN ablation** (eq 6) + **IQL baseline** (eq 4, the required §7.2 contrast). QPLEX `[10]` / Weighted-QMIX `[9]` are **§7.2 prose analysis** (not built). MAPPO is an **optional stretch** contrastive arm (not in the BRIEF bibliography → would need a README biblio addition). **Value decomposition applies ONLY within the cooperative cop team**; the **thief is a separate adversarial Double-DQN** folded into `T(s'|s,ā)`, trained by **alternating best-response self-play**. Shared recurrent GRU Q-net + **swappable Mixer ABC** seam (DRY).
3. **Reward = potential-based shaping** (provably policy-invariant, Ng 1999: `F = γ·Φ(s') − Φ(s)`, `Φ = −Manhattan(cop,thief)/d_max`, plus `−1/step`) during **TRAINING only**, toggled OFF at eval. The official **§3.4 scoreboard** (cop_win 20 / thief_win 10 / cop_loss 5 / thief_loss 5) is computed **separately** by a `Scorer` for reports — **never** the training signal.
4. **Cop-team size.** The graded **5×5 match is faithful 1-cop-vs-1-thief** (`env.num_cops:1`, BRIEF §3). A **2-cop scenario runs ONLY on the 4×4 training/sanity grid** (`env.curriculum.num_cops_by_stage:[1,1,2,1]`) so the QMIX mixer is genuinely multi-agent and §7.2 learning curves show real credit assignment. **Does NOT change the graded game rules.**

**Standing defaults:** simultaneous move resolution + **capture on same-cell OR swap** (`env.capture_on_swap:true`, a deliberate ADR-0004 rule clarification); `env.view_radius_by_grid` auto-tightens (`ceil(min(H,W)/2)−1` → 2×2:0, 3×3:1, 4×4:1, 5×5:2, with `env.view_radius_max:2`); `PLACE_BARRIER` counts as the move; barriers **cop-only** (≤5); curriculum **2×2→3×3→4×4→5×5**; seeds **`[7,17,37,71,107]`**; **Pygame god-view spectator** (reads the referee, NOT agent obs); a **neutral REFEREE = "the environment"** drives the 6 sub-games and calls each agent MCP tool per turn; cloud = **FastMCP + Prefect/Horizon**; **bearer JWT auth + jti-deny-list** revoke; Gmail via **smtplib + App-Password** with an `EmailSender` Protocol fallback to Google-API OAuth.

**PII (hard).** Real student names/IDs/emails + GitHub owner slug **NEVER** in tracked files — placeholders in tracked repo-root `players.example.yaml`, real values in git-ignored repo-root `players.local.yaml` + `.env`. Cover sheet `adrl-001-ex06.pdf` git-ignored (Moodle only); never asserted-present in a test (MEMORY: that exact assert broke A5 CI — skip-when-absent).

---

## Owner-role split (CLAUDE.md §1.4, ADR-0001/D9)

| Role | Primary ownership |
|---|---|
| **A** | Env / Dec-POMDP (`src/marl/env`), CTDE training (`src/marl`, `src/services`), OLoRA + dataset, results figures + §7 academic body (THEORY/ANALYSIS), config schema. |
| **B** | Dual-MCP servers + referee (`src/mcp`, ONE tool contract for localhost AND cloud — no `src/marl/mcp` fork), cloud deploy (`deploy/`), GUI (`src/gui`), Gmail report (`src/reporting`), API gatekeeper (`src/api`), integration + CI. |

Both sign off on every ADR before its code lands (PRD/PLAN edit first → then execute). Shared seams (SDK, config, schemas) are co-reviewed.

---

## Status legend

`[ ]` not started · `[~]` in progress · `[x]` done (DoD met + evidence captured) · `[!]` blocked (note the blocker) · `[B]` bonus / coordination-dependent.

---

# Phase 0 — Foundation & docs-first (BRIEF §6 pre-step; §1.4 contract)

> **Exit gate (P0):** docs reviewed + human sign-off; `uv sync` green; CI skeleton runs; repo shared read-only with `rmisegal@gmail.com`.

- [ ] **T0.1 — Scaffold repo tree** · _A_
  Create the flat `src/<layer>/` skeleton (D9 §1): `sdk/ marl/{env,mixers,nets,data,replay} services/ api/ mcp/ reporting/ gui/ results/ utils/`, plus `config/ scripts/ tests/{unit,integration,architecture,env} docs/{prd,adr,schema,screenshots} deploy/ results/ assets/ instructions/`.
  **DoD:** tree matches D9 §1; copy `CLAUDE.md` (+§1.4 contract table), `scripts/check_file_sizes.py`, `.gitignore`, `.github/workflows/ci.yml`, `pyproject.toml` adapted from A5; `uv groups` `dev`/`mcp`/`mail` declared; `uv.lock` committed.

- [ ] **T0.2 — Version pin = `1.0.0`** · _A_
  Set `src/__init__.py::__version__ = "1.0.0"` and `config.version: "1.0.0"`.
  **DoD:** `scripts/check_version.py` + `tests/architecture/test_version_consistency.py` assert `__version__ == config.version == "1.0.0"` (NOT `0.1.0` — V3 hard gate, ADR-0011).

- [ ] **T0.3 — Author config + schema + rate limits** · _A_
  Write the full `config/config.yaml` (D9 §4 merging D1/D2/D3/D5/D6/D8 keys, using the EXACT nested paths): `game.{grid_size:5,max_moves:25,num_games:6,max_barriers:5,scoring.{cop_win:20,thief_win:10,cop_loss:5,thief_loss:5}}`, `env.{num_cops:1,move_resolution:simultaneous,capture_on_swap:true,reward_mode:dec_pomdp,view_radius_by_grid,obs_channels:5,obs_scalars:6,actions.{a_cop:5,a_thief:4,enable_stay:false,enable_barrier:true},curriculum.{stages:[[2,2],[3,3],[4,4],[5,5]],num_cops_by_stage:[1,1,2,1],promotion_threshold,promotion_window}}`, `algo.{name:qmix,baseline:iql,double_q:true,gamma,lr_agent,lr_mixer,batch_episodes,target_update_interval,mixer.{type:qmix,embed_dim:32}}`, `nets.{hidden_dim:64,gru:true,encoder_hidden}`, `olora.{enabled:false,rank:4,rank_sweep,scale:8.0,target_layers}`, `bc.*`, `replay.*`, `selfplay.{window_k:1,episodes_per_round:100,pool_size,update_ratio,rounds:50}`, `training.{seeds:[7,17,37,71,107],episodes_per_stage:5000,eps_*}`, `reward.{...,phi_terminal_zero:true}`, `mcp.*`, `cloud.{...,workers:1}`, `gmail.*`, `gui.*`, `paths.*`. (NO top-level `algorithm`/`mixer`/`actions`/`agent`/`report`/`email`/`scoring`/scalar `view_radius` — those are the nested paths above.) Add `config/config.schema.yaml` (typed) + `config/rate_limits.json` (§5 gatekeeper, channels `peer_mcp`/`gmail`/`prefect_deploy`). Implement `src/utils/config_loader.py` with `_validate_config → ValueError` and `.env` interpolation.
  **DoD:** `load_config()` validates and round-trips; no algorithm-relevant numeric appears outside config; `tests/architecture/test_config_single_source.py` finds zero magic literals in `src/`.

- [ ] **T0.4 — `.env-example`, secrets policy, `.gitignore`** · _B_
  Tracked `.env-example` (names only, blank): `COP_MCP_TOKEN=`, `THIEF_MCP_TOKEN=`, `REFEREE_MCP_TOKEN=`, `COP_PUBLIC_URL=`, `THIEF_PUBLIC_URL=`, `MCP_PUBLIC_KEY=`, `MCP_PRIVATE_KEY=`, `REVOKED_TOKEN_JTIS=`, `PREFECT_API_KEY=`, `MODEL_PATH=`, `GMAIL_SENDER=`, `GMAIL_APP_PASSWORD=`, `REPORT_RECIPIENT_OVERRIDE=`, `GITHUB_REPO_URL=`. Tracked repo-root `players.example.yaml` (placeholder role A — SOLO, PRD §1.3); real values in git-ignored repo-root `players.local.yaml`. Git-ignore `.env`, `.env.*` (keep `!.env-example`), `*.pem`, `*.key`, `credentials*.json`, `players.local.yaml`, `secrets/`, `results/.report_sent`, `results/reports/*.real.json`, `adrl-001-ex06.pdf`/`.docx`, `instructions/`, heavy `results/` artifacts.
  **DoD:** `git ls-files` shows zero secrets/PII; `scripts/check_secrets.py` + `scripts/check_pii.py` pass; cover-sheet path ignored.

- [ ] **T0.5 — Docs-first PRD/PLAN/TODO + ADRs** · _A+B_
  Author `docs/PRD.md` (KPIs, FR/NFR, acceptance, §6-ordered milestones), `docs/PLAN.md` (C4 + UML + localhost→cloud deployment diagram + ADR index + SDK/MCP API contracts + risk register), this `docs/TODO.md`, the `docs/prd/PRD-{ENV,CTDE,MCP,REPORT}.md` set, and the numbered ADR set `ADR-0001`-`ADR-0014` (D9) **plus** the dimension ADRs (ADR-001 move-resolution, ADR-002 reward-framing, ADR-003 fixed-obs, ADR-004 barriers + the `capture_on_swap:true` rule clarification, ADR-D5-01..05 topology/auth, ADR-D6-1..8 cloud, ADR-GUI-01..05, ADR-D8-1..6 report, ADR-D10-A..G analysis).
  **DoD:** all docs exist; **human sign-off** recorded on every human-decided ADR (esp. ADR-001 move-resolution + ADR-002 reward) BEFORE any transition/reward code lands; `tests/architecture/test_required_docs_present.py` finds README 6 sections + `THEORY/ANALYSIS/QUALITY/UX/COST` docs + `docs/shared/PROMPTS.md` + the four `docs/prd/PRD-{ENV,CTDE,MCP,REPORT}.md` split docs (enumerated explicitly, NOT four separate tasks — V3 §16/§11 doc gate) (#16, #19).

- [ ] **T0.5b — Prompt log `docs/shared/PROMPTS.md` (§1.4 evidence)** · _A+B_
  Create `docs/shared/PROMPTS.md` (matches the PLAN tree mention) as the literal prompt log — the CLAUDE.md §1.4 human-architect/AI-implementer evidence trail — and **maintain it** across phases: append the verbatim prompts used per §-section as work lands (#16).
  **DoD:** `docs/shared/PROMPTS.md` exists, is non-placeholder, and is added to the `test_required_docs_present.py` enumeration (shared with T0.5); a per-section prompt entry is appended whenever a phase's code lands (kept current, not authored once).

- [ ] **T0.6 — Quality-gate tooling (RED first)** · _B_
  Implement `scripts/check_no_hardcode.py`, `check_secrets.py`, `check_pii.py`, `check_version.py`, copy `check_file_sizes.py` verbatim. `check_pii.py` MUST NOT enumerate literal deny-list tokens (real names/ids/emails/GitHub owner slug) in its own tracked source — it loads the deny-list from the git-ignored `players.local.yaml`/`secrets/` OR derives matches generically via regex shapes (A4 lesson) (#23). Write `tests/architecture/`: `test_sdk_single_entry`, `test_config_single_source`, `test_version_consistency`, `test_no_secrets_committed`, `test_no_pii_in_tracked_content`, `test_cover_sheet_gitignored` (**skip-when-absent**), `test_required_docs_present`, `test_egress_via_gatekeeper`, `test_final_gates`.
  **DoD:** gate tests written and RED where features don't yet exist; CI job order is `ruff check → ruff format --check → check_file_sizes (AFTER format) → check_no_hardcode → check_secrets → check_pii → check_version → pytest --cov-fail-under=85` (fetch-depth:0).

- [ ] **T0.7 — SDK single-entry skeleton** · _A_
  Stub `src/sdk/sdk.py::MarlSDK` with the public surface: `build_env`, `train`, `local_observation`, `select_action`, `step_match`, `play_full_game`, `build_report_json`, `send_final_report`, `build_policy(role,algo,stage)`, `spectator_session`, `export_weights`. To stay ≤150 LOC across the 9+ methods, split into a thin **facade** `src/sdk/sdk.py` (public surface only) + `src/sdk/_helpers.py` (private wiring); the public API shape is unchanged.
  **DoD:** every layer (GUI/MCP/report/scripts) routes through `src.sdk`; `sdk.py` ≤150 LOC (facade) with helpers split out; `test_sdk_single_entry.py` + `test_mcp_servers_have_no_logic.py` GREEN by construction.

- [ ] **T0.8 — Repo sharing & branch** · _B_
  Create branch `assignment-6`; add `rmisegal@gmail.com` (Dr. Yoram Segal, GitHub `rmisegal`) as read collaborator; add `.github/pull_request_template.md` (Scope/Req/Human/AI/Tests/V3-checklist); record identical `GITHUB_REPO_URL` for both members' Moodle. Keep a `/tmp` clone to recover `.git` (MEMORY: Google-Drive sync silently deletes `.git`; push often).
  **DoD:** branch exists; collaborator added; PR template present; both members have the identical repo URL.

---

# Phase 1 — Theory first (BRIEF §6.1): Dec-POMDP, S/A/R/Ω/O/γ, train-vs-exec split

> **Exit gate (P1):** formal tuple + reward tests pass; README §7 skeleton names every figure/experiment; the **local-obs-at-exec / global-state-at-train** split is encoded as a test.

- [ ] **T1.1 — Formal Dec-POMDP tuple + POSG caveat (README §7.1)** · _A_
  Write `M = ⟨N, S, {A_i}, T, R, {Ω_i}, O, γ⟩` (eq 1, cite `[1]`) for the cooperative cop team **and** the general-sum **POSG** caveat `G = ⟨I,S,{A_i},{O_i},P,Ω,{R_i},γ⟩` (eq 3, NEXP^NP) for the full game (role-specific 20/10/5/5 ⇒ deliberate Dec-POMDP proxy named as a limitation).
  **DoD:** `docs/THEORY.md` + README §7.1 contain both formalisms; **N = the cooperative COP TEAM** (the thief is folded into transition `T(s'|s,ā)`; the full game is a POSG — value decomposition NEVER crosses the cop/thief boundary; N is NOT `{cop,thief}`); `|A_cop|=5` `{UP,DOWN,LEFT,RIGHT,PLACE_BARRIER}` (`env.actions.a_cop:5`), `|A_thief|=4` `{UP,DOWN,LEFT,RIGHT}` (`env.actions.a_thief:4`) — STAY/NOOP is config-gated OFF (`env.actions.enable_stay:false`), so NO NOOP head, `γ` from `algo.gamma`; equation map footnotes **ex06 eq2 ≡ L10 eq4** (citation-numbering hazard R13).

- [ ] **T1.2 — Action enum + masking (TDD RED→GREEN)** · _A_
  Tests first for `src/marl/env/actions.py`: `IntEnum` `{UP,DOWN,LEFT,RIGHT,PLACE_BARRIER}` (STAY/NOOP defined in the enum but config-gated OFF via `env.actions.enable_stay:false`, so the live cop head = 5 `env.actions.a_cop`, thief head = 4 `env.actions.a_thief` with PLACE_BARRIER masked), `DELTAS`, `action_mask(state,agent)` (thief PLACE→masked; out-of-bounds; barrier-blocked; budget-exhausted). Then implement.
  **DoD:** mask tests pass for thief-PLACE-masked, OOB, barrier-blocked, `barriers_left==0`; `enable_stay:false` honored (no NOOP/STAY in the active action set); a_cop=5/a_thief=4; file ≤150 LOC.

- [ ] **T1.3 — Grid + types primitives (TDD)** · _A_
  `src/marl/env/grid.py` (`in_grid`, `manhattan`, `can_enter`, edge==wall, spawn sampler with `dist>radius`) + `src/marl/env/types.py` (`Pos`, `GlobalState @dataclass`, `Observation` TypedDict).
  **DoD:** `can_enter(y, actor)= y∈G and (y∉B or y==actor)` verified; `edge==wall` stays-in-place; ≤150 LOC each.

- [ ] **T1.4 — Reward model + Scorer, STRICTLY separate (TDD)** · _A_
  `src/marl/env/reward.py` (potential-based shaping `F=γΦ(s')−Φ(s)`, **team potential `Φ = −min_i manhattan(cop_i, thief) / d_max`** — the closest cop sets it — with **`d_max = (H−1)+(W−1)` derived per live grid from `state.h, state.w`** (architect decision #2, NOT a config key), grid-normalized; `−step_penalty`; terminal ±1 dominant, `Φ(terminal)=0`; per-agent dict; `dec_pomdp` ⇒ equal entries, `posg` ⇒ role-specific) + `src/marl/env/scorer.py` (Table-1 20/10/5/5, report-only).
  **DoD:** test asserts the shaping identity `F=γΦ(s')−Φ(s)`; test asserts `RewardModel` and `Scorer` are **distinct modules** and Scorer output is **never** consumed by training; `dec_pomdp` dict entries equal; shaping **toggles off at eval**.

- [ ] **T1.5 — README §7 outline + figure manifest** · _A_
  Draft README §7.1/§7.2/§7.3 skeleton naming figures **F1-F6** (D10 §A.3) and the single controlled experiment (IQL/VDN/QMIX × seeds × ladder).
  **DoD:** README skeleton present with §-anchored captions placeholders; each figure mapped to its generator + path.

---

# Phase 2 — Full basic pursuit (BRIEF §6.2): game logic + rules

> **Exit gate (P2):** game-logic tests ≥85% coverage; 2×2 capture works; `step()` proven to never leak `GlobalState`.

- [ ] **T2.1 — Pure transition function (TDD)** · _A_
  `src/marl/env/transition.py::resolve_joint_action(state, joint_a, cfg)` — **simultaneous** resolution from one pre-move barrier set; barrier placement-is-the-move + **TEAM budget cap** (the per-agent `action_mask` is permissive at `barriers_used==max-1` — both cops may show `PLACE=True`, so the joint resolver MUST cap the team total and drop/no-op any placement beyond `max_barriers`; codex P1-review F1); OOB/barrier ⇒ stay; **capture = same-cell OR swap**; `timeout = (step+1)>=max_moves`. PURE → trivially unit-testable + replay-deterministic.
  Returns a frozen **`TransitionResult(next_state, winner, capture)`**; capture is checked BEFORE timeout so a **capture on the final move beats timeout** (`winner="cop"` even at `(step+1)>=max_moves`); over-budget placement degrades to a stay **deterministically by cop index**.
  **DoD:** tests for shared-cell capture, **swap=capture**, edge stay, barrier blocks both agents, single-cop budget cap **AND a 2-cop simultaneous over-budget placement (team cap enforced, not per-agent)**, **capture-on-final-move beats timeout**, placement increments step; honors `move_resolution` flag (default `simultaneous`, ADR-001).

- [ ] **T2.2 — Observation builder, fixed 5×5 padded (TDD)** · _A_
  `src/marl/env/observation.py` — `view_radius(H,W)=max(0,ceil(min(H,W)/2)-1)` (override from `env.view_radius_by_grid`); egocentric image **always padded to max `W_v = 2*view_radius_max+1 = 5`** (`env.view_radius_max:2`) via the `out_of_bounds` channel so ONE net spans 2×2→5×5 (enables OLoRA); `env.obs_channels:5` image channels `{self, other_visible, barrier, out_of_bounds, time_norm}` + length-`env.obs_scalars:6` scalar memory vector (`other_seen_now`, `steps_since_seen_norm` aliasing hooks). If the builder + the egocentric-crop/pad encoder push past 150 LOC, split the pure crop/pad/channel-packing into `observation_encoder.py` (env imports it) so each file is ≤150 LOC.
  **Visibility-memory ownership:** `build_observation` is PURE given `memory`; the **env owns + updates** the per-role `VisibilityMemory(steps_since_seen)` (reset on `reset`; `0` when opp in view else `+1` per `step`). It is env-local state, NOT a `GlobalState` field, and NEVER on a step()/reset() payload.
  **DoD:** obs image shape `==(5,5,5)` at 2×2/3×3/4×4/5×5; `other_visible=0` beyond Manhattan radius; `steps_since_seen` resets-on-view/increments-otherwise (env test); each file ≤150 LOC.

- [ ] **T2.3 — Env class + train/exec split leak-test (TDD)** · _A_
  `src/marl/env/cops_robbers_env.py` — `reset()->(obs_dict LOCAL, info w/ action_mask + start ISO timestamp)`; `step()->4-tuple (obs_dict, reward_dict, terminated, info)`; `state()->GlobalState` (**train-only**). To keep the env ≤150 LOC across reset/step/state/render, factor the serializable `render_state()` (GUI/MCP-safe, no GlobalState) into a separate `src/marl/env/render_state.py` that the env delegates to.
  **DoD:** **leak test** (`tests/architecture/test_step_no_leak.py`) runs a full episode and **recursively scans** every value in the `reset()`/`step()` obs_dict + info — asserting NO `GlobalState` instance appears anywhere, no dict key equals any `GlobalState` field name (`{f.name for f in dataclasses.fields(GlobalState)}`), every Observation's keys `=={"image","scalars"}`, and `info` has no `state`/`global_state`/`barriers`/board-dim keys — while asserting `env.state()` **IS** a `GlobalState` (the sanctioned train-only accessor, so the global state exists yet is reachable ONLY via `state()`/`render_state()`); top-graded modeling requirement, BRIEF §4/§6.1; env + `render_state` each ≤150 LOC.

- [ ] **T2.4 — Curriculum scheduler (TDD)** · _A_
  `src/marl/env/curriculum.py` — Table-2 stages `[[2,2],[3,3],[4,4],[5,5]]`; `maybe_promote(success_rate)` on capture-rate ≥ `promotion_threshold` over `promotion_window`; spawn `dist>radius`; non-square 3×2/4×3 supported.
  **DoD:** scheduler promotes on threshold; full ≤25-move episode runs at each grid size; winner ∈ {cop,thief}; barriers ≤ max.

---

# Phase 3 — Full pipeline on minimal grid (BRIEF §6.3)

> **Exit gate (P3):** 2×2 reaches **pursuit-optimal capture from every 2×2 start over all `training.seeds`** (the thief FLEES, so naive min-Manhattan is the wrong metric; a zeroed Q-table must fail the gate); end-to-end collect→train→export runs NaN-free; a **headless `render_state()` god-view JSON** + a minimal sub-game JSON exist (the **first GUI screenshot moves to P7** — P3 emits the headless render_state JSON, NOT a Pygame shot). P3 = pipeline + tabular 2×2 smoke + dataset/replay machinery; **BC training is P4/T4.4.**

- [ ] **T3.1 — Centralized episodic replay + four §5.2 data sources (TDD RED first)** · _A_
  `src/marl/replay/episode_buffer.py::CentralizedReplayBuffer` — pre-allocated numpy ring, seeded `default_rng`, stores **whole padded episodes** (T+1 obs/state, T actions/reward/done + `filled` mask + hidden seed). **Two buffers** (cop, thief; rewards differ). "Centralized" = the only place global `s` lives, train-time only. The buffer ingests ALL FOUR §5.2 data sources into the SAME `CentralizedReplayBuffer` via one uniform `add_episode(episode, source_tag)` API: **(1) Manhattan-heuristic expert rollouts**, **(2) earlier-policy self-play episodes** (opponent-pool snapshots), **(3) random-legal coverage episodes** (legal-action-masked uniform exploration), **(4) live CTDE transitions** from the current learners.
  **Storage note (#27 readiness audit):** each step ALSO stores `next_legal_mask (T,N,n_actions) bool` (PRD:486 — P4 Double-DQN reads it; **never recomputed from cropped obs**); the `source_tag` is the `SourceTag(StrEnum){EXPERT, SELF_PLAY, RANDOM, LIVE_CTDE}` enum from `src/marl/data/schemas.py`. **N convention:** the cop buffer uses `n_agents = max(env.curriculum.num_cops_by_stage) = 2` (1-cop stages pad the agent axis with `filled=0`/zeros); the thief buffer uses `n_agents = 1` — TWO separate buffers (never one shared agent axis). **P3 scope:** the buffer adds a **fail-loud N-mismatch guard** (an episode whose agent axis ≠ `n_agents` raises, not a silent size-1→2 broadcast); the **full N=2 agent-padding convention** (zero-filled phantom slot + per-agent active mask) is **FINALIZED in P4** when num_cops=2 episodes actually exist.
  **DoD:** add/sample shape tests `(B,T+1,N,*)` for obs/scalars/state (T axis for actions/reward/done); post-done masking correct; seeded reproducibility; `next_legal_mask` round-trips per step; BPTT batch shapes verified BEFORE wiring the learner; **a test feeds one episode from EACH of the four sources** (heuristic / earlier-policy / random-legal / live-CTDE), asserts all four land in the single buffer with the correct `source_tag`, and a mixed sample draws across sources.

- [ ] **T3.2 — Schemas + heuristics + dataset (TDD)** · _A_
  `src/marl/data/schemas.py` (frozen `TransitionDTO` storing local obs + `global_state` + `next_legal_mask` + joint action; `DatasetManifest`), `heuristics.py` (barrier-aware BFS cop expert + flee thief expert), `obs_encoder.py` = a **thin BATCHING ADAPTER** over the P2 `env.observation_encoder.encode_image` (import + stack/concat into channels-first `(B,C,5,5)` f32 images + `(B,obs_scalars)` f32 scalars — **NOT a 2nd geometry impl**, DRY) **plus a pure `encode_state(state, cfg)` → stage-invariant 77-float** (`3·5·5 + 2`, masked) so P4's centralized critic gets a fixed input, `bc_dataset.py` (npz IO).
  **DoD:** TDD on a known 3×3 layout proves cop path is shortest + respects barriers; `encode_obs_batch` yields **channels-first `(B,C,5,5)`** + `(B,obs_scalars)` across all grids with OOB masked (no geometry re-implemented — reuses the env encoder); `encode_state` returns a length-77 vector at 2×2/3×3/4×4/5×5 with out-of-board cells masked; `TransitionDTO` schema test passes.

- [ ] **T3.3 — 2×2 pipeline smoke (integration)** · _A_
  Full collect→train→export on the 2×2 stage (Table-2 row 1) using a scripted/heuristic opponent, driven THROUGH the single SDK entry (`MarlSDK.run_p3_smoke` — no business logic in scripts/tests). The 2×2 learner is a **THROWAWAY tabular Q-learner** (`src/marl/replay/tabular_smoke.py`, hyperparams in the `p3_smoke:` config block, **deleted in P4** with the GRU CTDE learner) that exercises the `CentralizedReplayBuffer` end-to-end. **"Optimal" (D3) = the greedy (eval) cop achieves PURSUIT-OPTIMAL capture of the FLEEING heuristic thief from every distinct 2×2 start, for ALL `training.seeds`, NaN-free** (the thief flees, so naive min-Manhattan is the wrong target; a zeroed/untrained Q-table must FAIL this gate — non-vacuous). "Export" = the NPZ dataset + a minimal sub-game JSON (`docs/schema/subgame.schema.json`), NOT actor weights (no net at P3). **The first GUI screenshot moves to P7** — P3 emits a headless god-view JSON via `render_state()` (no GlobalState).
  **DoD:** runs NaN-free; 2×2 reaches pursuit-optimal capture from every 2×2 start (per the definition above) over ALL `training.seeds`; a minimal sub-game JSON record produced + validates vs `docs/schema/subgame.schema.json`; a headless `render_state()` god-view dict produced (no GlobalState); **≥85% coverage on `src/marl/replay` + `src/marl/data`** (the P3 machinery — NOT `src/marl/env`, which is the P2 coverage target).

---

# Phase 4 — Network quality + training (BRIEF §6.4): CTDE, QMIX/VDN/IQL, self-play, OLoRA, dataset

> **Exit gate (P4):** loss decreasing; 3×3 converges; seeded sweep logs to `results/runs/*.jsonl`; OLoRA + ablation produce §7.3 evidence.

- [ ] **T4.0a — Mixer ABC seam + VDN mixer (TDD)** · _A_ — _moved from P1 (#27): builds the network/training seam, belongs in P4 (matches PLAN P4, BRIEF §6 step 4)._
  `src/marl/mixers/base_mixer.py` (`forward(q_agents, state) -> q_tot`) as the single swappable point; `vdn_mixer.py` (sum over cop team, eq 6). IQL has no mixer — it branches in the learner (P4b), so no `iql_mixer.py`.
  **DoD:** test asserts VDN `Q_tot == Σ_i Q_i`; IQL returns per-agent independence; ABC is the only differentiator across algorithms (DRY); ≤150 LOC each.

- [ ] **T4.0b — Recurrent Q-net (TDD)** · _A_ — _moved from P1 (#27): the GRU Q-net is a training-network artifact, P4._
  Tests for `RecurrentQNet` shapes BEFORE implementing `src/marl/nets/agent_net.py` (or `src/marl/recurrent_q_net.py`): flat-MLP encoder trunk (`nets.encoder_hidden`, OLoRA-wrappable Linear) → `GRUCell(64)` → role-conditioned Q-head; `forward(obs[B,n,obs_dim], h[B,n,H]) -> (q[B,n,|A|], h')` (eq 8). All dims read from `env.obs_dim` (no hardcode).
  **DoD:** forward-shape + hidden-carry tests pass; online+target copy; agent-id one-hot for weight-sharing across same-role agents; ≤150 LOC.

- [ ] **T4.1 — QMIX mixer (TDD, monotone)** · _A_
  `src/marl/mixers/qmix_mixer.py` — monotonic hypernetwork (**softplus** weights, eq 7 — commits the abs/softplus contradiction to softplus, #3a), state-value `V(s)` head. Generic over `N` (so `num_cops≥2` is trivial).
  **DoD:** numeric test asserts `∂Q_tot/∂Q_i ≥ 0` (perturb one `Q_i` up never decreases `Q_tot`) on an explicit **N=2 mixer fixture**; IGM-consistency test (eq 5): decentralized greedy joint == argmax of mixer output; ≤150 LOC; gated behind `algo.name=qmix`.

- [x] **T4.2 — Learner base + role learners + IQL (TDD)** · _A_
  `src/marl/learner/learner_base.py::QmixLearner` (Double-DQN centralized target, BPTT, masked **Huber** loss δ=1.0, hard target sync interval 200, AdamW param-groups `lr_agent/lr_mixer`, telemetry loss/grad_norm/q_tot) + `src/marl/learner/learners.py` subclasses: `CopLearner` (mixer = `QmixMixer` when `algo.name=qmix` else `VdnMixer`, A=`env.actions.a_cop`=5), `ThiefLearner` (single-agent adversarial Double-DQN, A=`env.actions.a_thief`=4, slices the `a_cop` env mask to `a_thief`, **NOT in any mixer**), `IqlLearner` (per-agent eq 4 target, **NO mixer / NO global state**, SINGLE AdamW group at `lr_agent` only). The learner owns ALL masking: per-agent `Q_i *= active` BEFORE the mixer (both online + target paths) where `active` is the **EPISODE-CONSTANT per-slot** occupancy mask (slots `0..k-1` real, `k..N-1` zero-filled phantoms — distinct from the per-timestep `filled` length mask, and NOT time-varying); team reward = active-**MEAN** over cops (never a sum). IQL loss is masked by `filled` AND `active`.
  **DoD:** masked-loss test (padding ⇒ 0 gradient); overfit-one-episode (loss → ~0); `∂Q_tot/∂q_inactive == 0`; active-MEAN reward; IQL single param-group (no `lr_mixer`, no mixer attr); ThiefLearner A=4; IGM/monotonicity tests pass; **import-boundary test** proves the env/nets execution path cannot import learner/mixers/`episode_buffer`/`GlobalState`/`services` (MCP half extended in P5); cop & thief are two distinct learners + two buffers (POSG, ADR-D3-A).

- [ ] **T4.3 — OLoRA (TDD, paper-faithful)** · _A_
  `src/marl/olora.py` / `src/marl/nets/{olora_linear,olora_init}.py` — thin-QR on the **pretrained** weight `W0=QR` (eq 10), `B=Q_r[:, :r]`, `A=R_r[:r,:]`, re-base `W0 ← W0 − s·BA` (eq 4-of-[7]), forward `W_adapted = W0 + s·BA` (eq 11). Wraps **encoder Linears ONLY** (not Q-head, not QMIX hypernet). `rank=4` default, assert `rank ≤ min(m,n)//2`, `scale=8.0`.
  **DoD:** `‖W_adapted − W0‖ < 1e-5` at init (function-preserving); `Bᵀ·B ≈ I_r` (orthonormal); QR runs on `W0` **not a random matrix** (reject orthonormal-LoRA — §7.2 evidence); only `{A,B,head,mixer}` trainable, every `W0.requires_grad=False`; trainable-param count (full vs OLoRA, ~8× fewer) logged.

- [ ] **T4.4 — BC pretrain (Stage A)** · _A_
  `scripts/pretrain_bc.py` — generate Manhattan-heuristic `(o,a*)` pairs on 2×2/3×3 (~40k, seeded, `data/bc/<grid>.npz`, tiny seed subset vendored for CI), BC-train `CopNet`/`ThiefNet`, assert held-out **val-acc ≥ 0.9** (`bc.val_acc_gate`) before OLoRA attach.
  **DoD:** runs offline (no network); val-acc ≥ 0.9; base checkpoints under `checkpoints/base/`; `L_BC = −E[log π(a*|o)]` cross-entropy.

- [ ] **T4.5 — OLoRA attach + CTDE finetune (Stages B,C)** · _A_
  `scripts/attach_olora.py` (freeze base, QR-attach, save bundle `{base_sha, adapters, head}` with load-time `base_sha` guard) + `scripts/finetune_ctde.py` (hand OLoRA-wrapped encoders to the CTDE trainer; train up curriculum 3×3→4×4→5×5 reusing centralized replay).
  **DoD:** bundle round-trips with `base_sha` match; finetune emits loss + learning curves across stages; one frozen encoder spans the whole ladder (pad-to-5 verified).

- [ ] **T4.6 — Self-play orchestration (TDD)** · _A_
  `src/services/selfplay.py` (**alternating best-response is THE regime** — #7: one learner trains against a frozen opponent for `selfplay.window_k` rounds, then roles swap; `selfplay.window_k` is ONLY the frozen-opponent window length, NOT an update ratio; opponent pool `selfplay.pool_size` seeded by the Manhattan heuristic) + `src/services/rollout.py` (collect one episode driving both ε-greedy policies, per-role episode dicts, hidden reset) + `src/services/trainer.py`/`marl_trainer.py` (`SelfPlayTrainer`: collect→store(both)→update(both)→history; `selfplay.rounds:50` rounds with the **budget identity** `rounds(50) × episodes_per_round(100) = episodes_per_stage(5000)`; per-round ε reset/anneal from `training.eps_*`; `update_ratio` scoped ONLY to within-round learner cadence; curriculum transfer; OLoRA per stage) + `src/services/checkpoints.py` (full-vs-export split + shape sidecar).
  **DoD:** trainer logs opponent-pool sampling/snapshot (a frozen snapshot is pushed to the pool at each role-swap); `rounds × episodes_per_round == episodes_per_stage` asserted in a test; fixed-seed eval shows non-degenerate (non-cycling) convergence; `history` is the single source for §7.3 curves; thief & cop update from the SAME rollout, each its own reward.

- [ ] **T4.7 — Seeded sweep harness → JSONL** · _A_
  Build the controlled experiment (D10 §C): arms **IQL/VDN/QMIX**, ladder **2×2→3×3→4×4→5×5**, seeds **`[7,17,37,71,107]`**; identical nets/replay/ε-decay/γ/target cadence (only the mixer differs). Append per-episode records to `results/runs/*.jsonl` (schema: `{method,seed,grid,episode,reward_team,reward_cop,reward_thief,td_loss,captured,steps,stage}`). Also run the **4×4 2-cop scenario** (`env.curriculum.num_cops_by_stage:[1,1,2,1]`) for non-vacuous credit-assignment evidence (this is the genuine multi-agent mixer arm — the K1 KPI evidence, NOT the degenerate 1-cop 5×5).
  **DoD:** loss decreasing; 3×3 converges; acceptance `QMIX ≥ VDN ≥ IQL` on 5×5 capture-rate OR README honestly explains failure with the curves (a "bad" IQL curve is a §7.2 feature); `results/experiment_manifest.json` records seeds, config hashes, git commit, run IDs.

- [ ] **T4.8 — SDK training entry + thin scripts** · _A_
  Wire `MarlSDK.train(algorithm,seed)` / `build_policy(role,algo,stage)` / `export_weights(role,path)` (saves **GRUQNet state_dict ONLY** + shape sidecar JSON `{obs_dim,hidden,A,radius,P,C_obs}`; mixer/opponent-heads/global-state **never** exported). `scripts/train_ctde.py` + `scripts/train_iql_baseline.py` are thin wrappers.
  **DoD:** export artifact contains agent-net params only (load + key-set assertion); `export_weights` strips all train-only structure; scripts route through SDK.

---

# Phase 5 — MCP protocol for inter-agent comms (BRIEF §6.5): two servers + referee, localhost

> **Exit gate (P5):** localhost — cop server queries thief over real HTTP (Bearer 200) and vice-versa; F4 local comms log captured; 401 on bad/revoked token.

- [ ] **T5.1 — `mcp:` config + token placeholders** · _B_
  Author/verify the `mcp:` block (`mcp.host:127.0.0.1`, `mcp.cop_port:8001`, `mcp.thief_port:8002`, `mcp.path:/mcp`, `mcp.observation.view_radius:2`, `mcp.auth.*` env-var names + `required_scopes:[game:write]`, `mcp.transport:http`, `mcp.client.{timeout_s:10,max_retries:3,backoff_s:0.5,prewarm_ping:true}`, `mcp.auth.{cop_audience,thief_audience}`) in `config/config.yaml`; token + `*_PUBLIC_URL` placeholders to `.env-example`.
  **DoD:** zero hardcoded ports/URLs/tokens in `src/` (ports read from `mcp.cop_port`/`mcp.thief_port` = 8001/8002, NOT 8101/8102); one env-switched URL code path (localhost else `*_PUBLIC_URL`).

- [ ] **T5.2 — Wire schemas (TDD RED first)** · _B_
  `tests/unit/test_schemas.py` first: `MoveResponse`/`ActionResponse` has **no** value/logit/hidden field; `reveal_location` radius-gated; every request schema **requires `session_id`** (a request missing it is rejected — #11); `query_opponent` has a typed request/response with no policy internals; `MatchReport` key set matches §3.5 (`group_name, students[], github_repo, timezone, sub_games[6], totals`); **no** `get_state`/`get_policy`/`get_qvalues` tool registered.
  **DoD:** all schema-hiding tests written and RED (incl. the missing-`session_id` rejection + the `query_opponent` typed-schema test).

- [ ] **T5.3 — `schemas.py` + `actions.py` (single wire-JSON source)** · _B_
  `src/mcp/schemas.py` (pydantic v2: `Observation`, `MoveRequest`/`ActRequest`, `MoveResponse`/`ActResponse`, `LocationReveal`, `QueryOpponentRequest`/`QueryOpponentResponse`, `SubGameResult`, `MatchReport`, `AuthClaims`) + `src/mcp/actions.py` (enum + `legal_action_mask`). **`session_id` is a REQUIRED field on every request schema** (#11) — it keys the server-side GRU `z_t` (T5.5) and gives `(session_id, tick)` idempotency. **`query_opponent` carries a real schema** (`QueryOpponentRequest`/`QueryOpponentResponse`, evidence-only fields — no Q-values/weights/`z_t`) rather than being implicit.
  **DoD:** T5.2 tests GREEN; `session_id` present + required on all request models (a schema test asserts a request missing `session_id` is rejected); `query_opponent` round-trips via its typed schema and exposes no policy internals; `Observation` is the ONLY policy input; `select_action`/`request_move` accept ONLY local obs + legal_actions, **never** global state `s` (HARD invariant, decentralized-execution).

- [ ] **T5.4 — Auth module (local→cloud, revocable)** · _B_
  `src/mcp/auth.py` — `build_verifier(cfg)`: `StaticTokenVerifier` (local, env tokens) | `JWTVerifier` (cloud, RS256, `aud`/`exp`); `jti` deny-list check; `last4(token)` log helper; per-tool scopes (`position:read`, `game:write`, `report:send`, `admin:revoke`).
  **DoD:** unit test: missing/bad Bearer → 401; revoked `jti` → 401; valid → 200; full tokens never logged (last-4 only).

- [ ] **T5.5 — Private agent runtime + recurrent session state (DRY)** · _B_
  `src/mcp/agent_runtime.py::AgentController` — `Observation -> action` via `sdk.build_policy`; role-parametrized for both servers; policy internals **unreachable** from any tool return. Holds **server-side per-sub-game GRU hidden state** `z_t = f_phi(z_{t-1}, o_t)` (eq 8) keyed by a session id, carried across the ~25 `request_move` ticks of a sub-game and **reset on `new_sub_game`** (the canonical sub-game-lifecycle tool, #11). Do NOT run the agent servers with `FASTMCP_STATELESS_HTTP=true` (a stateless server would drop `z_t` between ticks); server-side per-session state is the chosen design (ADR-D5-04).
  **DoD:** `a_i = argmax_a mask_i(a)·Q_i(o_i, z_t, a; θ_i)`; hidden-state continuity verified across ticks within a sub-game AND reset to zero on `new_sub_game`; servers are NOT stateless-HTTP; no Q-values/weights/`z_t` reachable from returns.

- [ ] **T5.6 — Cop + Thief servers** · _B_
  `src/mcp/cop_server.py` + `thief_server.py` — `FastMCP(role, auth=verifier)`; the canonical tool set (#11) `health`, `ping`, `new_sub_game` (single sub-game-lifecycle reset tool — supersedes the old `start_sub_game`/`end_sub_game` pair), `request_move`, `reveal_location` (radius-gated), `query_opponent` (real peer HTTP, **evidence-only**, NOT fed to next `request_move`); cop-only `send_final_report`. `mcp.run(transport='http', host, port from config)`.
  **DoD:** two distinct processes on two ports; neither imports the other's policy module; `query_opponent` result decoupled from policy input (partial-observability preserved); `transport='http'` (Streamable HTTP, not deprecated SSE) verified against the installed `fastmcp` version; each tool handler documents **Input/Output/Setup** and calls `_validate_input()` that raises `TypeError` on a malformed payload (V3 §16, #20).

- [ ] **T5.7 — Typed peer clients** · _B_
  `src/mcp/clients.py` — `make_client(url, token) -> Client(BearerAuth)`; typed wrappers with `timeout_s` + `max_retries` + structured per-call logging (§7.3d).
  **DoD:** every call wrapped in timeout+retry; structured log lines suitable for the F4 capture.

- [ ] **T5.8 — Referee / MatchRunner** · _B_
  `src/mcp/referee.py` — sole holder of ground-truth `s`; `observe()=O(s,i)` ego grid; `apply()` legality + barrier + **simultaneous** capture (per human ADR-001); ISO-8601 +03:00 timing; scoring Table 1; **technical-loss replay** to reach exactly 6 valid sub-games (§3.7). Referee = the environment (CTDE global-state holder), NOT a third player.
  **DoD:** integration test plays a scripted sub-game; referee adjudicates capture/timeout + winner + scores; neither server can read global state via any tool; technical losses not counted.

- [ ] **T5.9 — API gatekeeper (§5 REQUIRED for A6)** · _B_
  `src/api/gatekeeper.py` — a SINGLE `ApiGatekeeper` with `execute(call)` AND `get_queue_status()`, a per-channel token-bucket built from versioned git-tracked `config/rate_limits.json` (`peer_mcp 120/min burst 10`, `gmail 5/min burst 1`, `prefect_deploy 30/min burst 5`), a **FIFO overflow queue** (`overflow_policy:fifo_queue`, `on_overflow:enqueue`, `max_queue:256` — NO crash on burst), and `log_all_calls:true` per-call logging. ALL outbound egress (peer-MCP HTTP, Gmail, Prefect deploy/status) routes through `ApiGatekeeper.execute()`; + `http_client.py` (bearer) is the only httpx wrapper.
  **DoD:** `test_egress_via_gatekeeper.py` fails if any module imports `httpx`/Gmail/Prefect-deploy outside `src/api` + `src/reporting/mailer.py`; `get_queue_status()` reports per-channel queue depth; burst beyond `burst` enqueues (does not crash) and drains FIFO; every call logged; ADR-0006 written (§5 flips N/A→REQUIRED).

---

# Phase 6 — Full local system run (BRIEF §6.6)

> **Exit gate (P6):** whole 6-sub-game match over localhost MCP; totals match; §3.5 JSON assembled + validates (no send yet).

- [ ] **T6.1 — `run_match.py` orchestrator** · _B_
  Top-level entry: spawn both servers (subprocess), poll `/healthz`, run referee for 6 valid sub-games, assemble `MatchReport`, call `cop.send_final_report` (dry-run at this phase). Routes through SDK (no business logic in script).
  **DoD:** `run_match.py --stage local` completes **exactly 6 valid sub-games** + emits referee trace; cold-start absorbed by health poll.

- [ ] **T6.2 — Match integration test** · _B_
  `tests/integration/test_match.py` — spin both servers (FastMCP in-memory `Client`), scripted sub-game; assert capture/timeout winners + scores + exactly 6 `sub_games` + single report; assert **401** on bad/revoked token; assert `query_opponent` result **not** in next `request_move` input.
  **DoD:** integration test GREEN; mocks the opponent for determinism (no live-net flake in CI, R11).

- [ ] **T6.3 — Assemble §3.5 JSON (no send)** · _B_
  On full local run, assemble the §3.5 body (`sub_games[6]`, scores derived Cop-side from `game.scoring`, totals) and validate against `docs/schema/report.schema.json`.
  **DoD:** JSON validates; totals `==` Σ per-sub-game scores; assemble-only (send is §6 step 9).

---

# Phase 7 — GUI (BRIEF §6.7, §5.4): Pygame god-view spectator

> **Exit gate (P7):** real-time board renders cop/thief/barriers/step counter; F3 screenshots at 2×2/3×3/4×4/5×5; spectator-purity tests pass.

- [ ] **T7.1 — `SpectatorFrame` + SDK session (TDD)** · _B_
  Frozen `SpectatorFrame` dataclass (grid_size, cop, thief, barriers, view_radius, move/max_moves, sub_game/num_games, scores, totals, winner, last_action; optional cloud fields None in-proc) + `SDK.spectator_session()` / `SpectatorSession.step()/reset()`.
  **DoD:** frame-shape test RED→GREEN; `SpectatorFrame` frozen; SDK is the single entry the GUI imports.

- [ ] **T7.2 — Headless test harness** · _B_
  `tests/conftest.py` sets `SDL_VIDEODRIVER=dummy` + `SDL_AUDIODRIVER=dummy` BEFORE pygame import; `headless_surface` fixture (A5 pattern).
  **DoD:** GUI tests import without opening a window.

- [ ] **T7.3 — Transform + palette (TDD)** · _B_
  `src/gui/transform.py::GridView` (capped square cells, letterbox, dynamic 2..5) + `src/gui/palette.py` (LOCAL literals only: colors, `GRID_W`, `TOKEN_INSET`, `CELL_PX_CAP`, `FPS`, `MOVE_ANIM_MS`, fonts — **no config import**).
  **DoD:** `test_transform` covers sizes 2/3/4/5 + bounds; `test_palette` asserts valid RGB + **no config import** (CLAUDE.md §4 local-styling rule).

- [ ] **T7.4 — Grid view + HUD + score view (TDD)** · _B_
  `src/gui/grid_view.py` (back-to-front: bg, checkerboard, gridlines, barriers, optional view-radius overlay off by default, thief, cop, capture flash), `hud.py` (sub-game i/6, move k/25, scores, last actions, winner banner), `score_view.py` (6-row results table + totals → visualizes §3.5 JSON).
  **DoD:** golden tests assert non-bg pixels for cop/thief/barrier/HUD; winner + no-winner variants covered.

- [ ] **T7.5 — Input map + state client + app (TDD)** · _B_
  `input_map.py` (space/+/-/n/r/v/esc), `state_client.py` (source-agnostic: in-proc `SpectatorSession` now + HTTP SSE/poll stub for Stage-2), `app.py` (owns SDK + StateClient; `update()/draw()`; rebuild `GridView` on grid_size change; live + replay).
  **DoD:** `test_state_client` confirms in-proc + mocked SSE/poll both yield identical `SpectatorFrame`; `test_app` drives ticks headless + rebuilds on size change.

- [ ] **T7.6 — Launcher + screenshot automation** · _B_
  `scripts/play.py` (window loop only under `__main__`; `--config --seed --live/--replay --referee-url --token`) + `scripts/capture_screens.py` (headless, fixture table per size, `pygame.image.save` to `results/screenshots/grid_{2,3,4,5}x{n}.png` with cop+thief+≥1 barrier visible).
  **DoD:** `uv run scripts/capture_screens.py` produces 4 PNGs headlessly (F3); `test_screenshot_matrix` asserts existence/dims/non-bg cop/thief/barrier/HUD pixels.

- [ ] **T7.7 — GUI architecture gates** · _B_
  `test_gui_imports_only_sdk.py` (GUI imports only pygame + `src.sdk` + `src.gui`), `test_spectator_purity.py` (frozen frame; GUI never imports referee/agent/MCP internals; `referee.build_local_obs` Manhattan-masks beyond `view_radius` — Dec-POMDP §2.1 invariant as a hard test), `test_no_agent_global_state_leak.py` (agent `request_move` payload contains ONLY local obs — no global pos/totals/`SpectatorFrame`).
  **DoD:** all three architecture gates pass before merge.

- [ ] **T7.8 — `docs/UX.md` Nielsen heuristics (§10 REQUIRED)** · _B_
  Map all 10 Nielsen heuristics to the GUI with a screenshot per state (§7.3c). §10 flips N/A→REQUIRED because the GUI is now mandatory.
  **DoD:** `test_required_docs_present.py` finds `docs/UX.md` with all 10 headings + screenshots.

---

# Phase 8 — Cloud-MCP deploy (BRIEF §6.8, §8): FastMCP + Prefect Horizon

> **Exit gate (P8):** both servers live on public HTTPS URLs; cloud cop↔thief comms with a shared `trace_id`; 401-without-token + revoke demo captured. Cloud is **upside, not an A6 grading gate** (ADR-D10-E; localhost F4 is canonical).

> **§8 four-stage cloud runbook (BRIEF §8 appendix).** T8.0→T8.4 implement the four mandated appendix stages, in order: **(1) local `uv` deps** (T8.0: prefect/fastmcp/torch/numpy via `pyproject` + `uv sync`), **(2) `mcp_server.py` `@mcp.tool` loading the local OLoRA-tuned nets** (T8.1, same tool contract as localhost), **(3) Prefect token + deploy + publish public URL** (T8.2/T8.3), **(4) Gmail auto-report** (Phase 9, triggered after the 6th cloud sub-game). `deploy/runbook.md` (T8.5) is the copy-pasteable ordered version of these four stages.

- [ ] **T8.0 — Stage 1: local uv deps + pin fastmcp + verify auth import path** · _B_
  Declare the cloud deps (`prefect`, `fastmcp`, `torch`, `numpy`) in `pyproject.toml` (the `mcp` uv group); `uv run python -c 'import fastmcp; print(fastmcp.__version__)'`; confirm exact import path for `BearerAuthProvider`/`JWTVerifier`/`RSAKeyPair` (v2 vs v3 moved it). Pin that version in `pyproject.toml` (**NO `deploy/requirements.txt`** — V3 forbids requirements.txt; if a deploy target demands a pinned file, generate it at deploy time via `uv export --no-dev > <tmp>` / `uv pip compile`, never tracked).
  **DoD:** version pinned in `pyproject.toml`; `uv.lock` committed; import path verified BEFORE any cloud auth code (do not guess); no tracked `requirements.txt` anywhere.

- [ ] **T8.1 — Stage 2: cloud server = SAME `src/mcp/` contract + JWT auth** · _B_
  Reuse the ONE tool contract from Phase 5 — there is NO divergent `src/marl/mcp` fork and NO `propose_action`/`commit_turn`/`get_public_state`/`place_barrier` tool set. The cloud entrypoints `src/mcp/cop_server.py:mcp` / `src/mcp/thief_server.py:mcp` register the SAME canonical tools (`health`, `ping`, `new_sub_game`, `request_move` [local-obs-only, **rejects any `global_state` field**], `reveal_location` [radius-gated], `query_opponent`, cop-only `send_final_report`) at the SAME `mcp.path:/mcp`. The only Stage-1→Stage-2 delta is the auth verifier: swap `StaticTokenVerifier`→`JWTVerifier` (RS256 from env; jti deny-list; client token minter) via `build_verifier(cfg)` (T5.4), wiring OLoRA-tuned actor nets through `sdk.build_policy`.
  **DoD:** localhost and cloud expose IDENTICAL tool names + `/mcp` path (one contract, ADR-D5-01); unit test: missing/revoked Bearer → 401; `request_move` rejects `global_state`; deployed `.pt` files contain actor weights only; each file ≤150 LOC.

- [ ] **T8.2 — Stage 3: deploy both servers to Horizon + publish public URL** · _B_
  Sign in to horizon.prefect.io with GitHub; create deploy `PREFECT_API_KEY` (Prefect-deploy egress routed via the `prefect_deploy` gatekeeper channel, 30/min); deploy cop (entrypoint `src/mcp/cop_server.py:mcp`, name `adrl-001-cop`, env `MCP_PUBLIC_KEY/issuer/cop_audience/REVOKED_TOKEN_JTIS/MODEL_PATH`, disable org-OAuth, rely on app JWT) → record `https://adrl-001-cop.fastmcp.app/mcp`; repeat for thief.
  **DoD:** both URLs return `health()` over the public internet from a machine outside the dev host; record both URLs in `cloud.cop_url`/`cloud.thief_url`.

- [ ] **T8.3 — Wire peers + cloud match** · _B_
  Set each server's `PEER_MCP_URL`+`PEER_MCP_TOKEN`; redeploy; run `python -m mcp.cloud_match --num-games 6 --max-moves 25` over the two cloud URLs → `full_match.json` (the §3.5 body). Cloud client uses `mcp.client.timeout_s:10`, `mcp.client.max_retries:3`+backoff, `mcp.client.prewarm_ping:true`; the run reuses the SAME `request_move`/`new_sub_game` contract as localhost (no `commit_turn` fork).
  **DoD:** ≥1 valid cloud sub-game; `full_match.json` produced; **does NOT send email** (that is §6 step 9).

- [ ] **T8.4 — Cloud proofs (§7.3d) + redaction** · _B_
  `scripts/smoke_cloud.py` (3-turn cross-server, shared `trace_id` in BOTH servers' logs) → `cloud_comms.png`; `scripts/smoke_auth.py` (no/bad token → `cloud_auth_401.png`); revoke a jti + redeploy + re-run → old token 401 / new token 200 → `cloud_revoke.png`. Run `scripts/redact_logs.py` (strip Authorization/Bearer/key material) BEFORE every screenshot.
  **DoD:** four proof captures exist (local comms, cloud comms w/ shared trace_id, 401-without-token, revoke demo); no secret/PII in any capture.

- [ ] **T8.5 — Render fallback (tested, not theoretical)** · _B_
  `deploy/render.yaml` + ONE real Render deploy + warm-up `health()` ping smoke test; `deploy/runbook.md` (copy-pasteable ordered runbook).
  **DoD:** `render.yaml` exists; a captured log shows one successful Render deploy + warm-up ping; runbook written.

- [ ] **T8.6 — Cloud GUI spectator path** · _B_
  Wire `state_client.py` HTTP path to referee `/spectator/events` (SSE) + `/spectator/state` (poll fallback) behind a SEPARATE spectator bearer token (distinct from agent tokens; NOT in any agent MCP tool contract).
  **DoD:** GUI can watch a distributed cloud match; spectator token ≠ agent tokens; `test_state_client` HTTP path yields identical `SpectatorFrame`.

---

# Phase 9 — Gmail report (BRIEF §6.9, §3.5, §5.5)

> **Exit gate (P9):** Cop sends ONE end-of-game JSON email to `rmisegal+marl@gmail.com`; body validates vs §3.5 schema; idempotent; zero PII/secrets tracked.

- [ ] **T9.1 — Report config + secrets + players** · _B_
  Author/verify the `gmail.*` block (`gmail.{to:"rmisegal+marl@gmail.com", mechanism:smtplib_app_password, smtp_host:smtp.gmail.com, smtp_port:587, subject_template, output_dir:"results/reports", sentinel:"results/.report_sent", schema_path:"docs/schema/report.schema.json"}`; `project.timezone:"Asia/Jerusalem"`; `github_repo` injected from `players.local.yaml`) in config — there is NO top-level `report:`/`email:` block, all live under `gmail.*`. Create git-ignored repo-root `players.local.yaml` (real names/ids) + tracked repo-root `players.example.yaml` (placeholders).
  **DoD:** recipient + `+marl` plus-tag live in `gmail.to` (never inlined); no PII/secret in tracked files.

- [ ] **T9.2 — Schema + validators (TDD RED first)** · _B_
  Author `docs/schema/report.schema.json` (draft-2020-12, `additionalProperties:false`, ISO-8601 regex `+0[23]:00`, `sub_games` minItems=maxItems=6, `students` **minItems:1** maxItems:3 unique roles — SOLO submission is role A only, so exactly 1 entry is valid). Tests first: `validate()` rejects `len!=6`, ids `!=[1..6]`, bad winner, `end<start`, totals mismatch, AND `students` empty (0 entries); a 1-entry `students` array PASSES; jsonschema test against the committed schema.
  **DoD:** schema committed; runtime `src/reporting/schema.py` (stdlib `@dataclass` `Student/SubGame/MatchReport` + `to_dict()` + `validate()`, **no Pydantic** runtime dep); validation tests GREEN.

- [ ] **T9.3 — Collector + builder + players + redact + idempotency** · _B_
  `collector.py` (`MatchRecorder` zoneinfo timestamps, `isoformat(timespec='seconds')` → +03:00 IDT), `builder.py` (winner→scores from `game.scoring`, `build_report()`, `format_subject()` from `gmail.subject_template`), `players.py` (`load_players()` from `players.local.yaml`), `redact.py` (role-only JSON → `results/reports/*.redacted.json`), `idempotency.py` (sha256 canonical hash + `gmail.sentinel` written only AFTER successful send).
  **DoD:** scores derived solely from winner+config (`cop`→{20,5}, `thief`→{5,10}); subject formats correctly; timestamps end `+03:00`; redacted copy has no `full_name`/`id`; `builder.py` documents **Input/Output/Setup** and `build_report()` calls `_validate_input()` raising `TypeError` on a malformed sub-game list (V3 §16, #20).

- [ ] **T9.4 — Mailer (TDD, smtplib default)** · _B_
  `mailer.py` — `EmailSender` Protocol + `GmailMailer` (smtplib STARTTLS + App Password, injectable `smtp_factory`) + `FakeEmailSender`. Routes egress through the gatekeeper.
  **DoD:** unit test with MagicMock `smtp_factory` asserts `starttls→login→send_message` order + UTF-8 body (Hebrew names survive `ensure_ascii=False`), zero live sends; ≥85% coverage with no network.

- [ ] **T9.5 — SDK send + Cop trigger (idempotent)** · _B_
  `MarlSDK.send_final_report(recorder)`: load players → build → validate → `dumps(ensure_ascii=False)` → subject → hash → idempotent env-cred send → write redacted copy. Wire Cop MCP `record_sub_game_result` + orchestrator to call it **exactly once after the 6th valid sub-game**; Thief never sends.
  **DoD:** calling twice with the same report sends once (FakeEmailSender call count == 1); Thief has no report tool/credentials; `scripts/smtp_smoke.py` validates the App Password days before the demo (Hebrew sample name).

- [ ] **T9.6 — Report CI gates + evidence** · _B_
  CI greps `src/` for literal `rmisegal+marl` + score magnitudes (fail if inlined); deny-list grep for real surnames/id patterns/secret values; assert no tracked `results/reports` file has `full_name`/`id`. README §7.3 commits `match_report.redacted.json` + screenshot of the received email.
  **DoD:** all report CI gates pass; redacted JSON + email screenshot committed as evidence.

---

# Phase 10 — Results & §7 academic analysis + figures (BRIEF §7)

> **Exit gate (P10):** all six figures present + non-placeholder + reproducible from `results/runs/`; README §7 body complete; every number traces to code.

- [ ] **T10.1 — Aggregator + plots + make_figures** · _A_
  `src/results/aggregate.py` (per-method per-seed mean±SE, win-rate, time-to-capture), `src/results/plots.py` (matplotlib; alpha/fontsize/dpi LOCAL literals OK), `src/results/make_figures.py` (single entry reading `results/runs/*.jsonl`).
  **DoD:** `uv run python -m src.results.make_figures` regenerates committed figures; `experiment_manifest.json` pins seeds/config-hash/commit (figure-drift R8 eliminated).

- [ ] **T10.2 — Generate F1-F6** · _A_
  **Plotted from `results/runs/*.jsonl`** via `make_figures`: **F1** both-agent learning curves (reward vs episode, mean±SE) → `results/figures/learning_curves.png`; **F2** per-stage loss curves → `loss_curves.png`; **F5** IQL vs VDN vs QMIX win-rate/convergence → `baseline_comparison.png`; **F6** scale effect (capture-rate vs grid size) → `scaling.png`. **Deterministically CAPTURED, not plotted** (#26): **F3** GUI screenshots 2×2/3×3/4×4/5×5 (from T7.6) + **F4** MCP-comms proof (localhost CANONICAL CLI log, cloud if P8) → `mcp_comms_local.png`. Plus the OLoRA ablation chart (OLoRA vs full-finetune vs frozen-base) + trainable-param table (plotted from runs).
  **DoD:** all six figures non-placeholder; F1/F2/F5/F6 regenerate from `results/runs/` (F3/F4 are captured artifacts checked for existence/dims, not regenerated from runs); IQL-vs-VDN divergence/convergence contrast visible in F5; capture-rate above Manhattan floor + above IQL on held-out seeds.

- [ ] **T10.3 — README §7.1 formalisms** · _A_
  Dec-POMDP tuple (eq 1) + POSG caveat (eq 3); chosen value function (QMIX eq 7; VDN eq 6 special case; IGM eq 5); "assumptions-in-code" module-boundary evidence (train reads global `s` in mixer; MCP exec reads ONLY `o_i`); equation map appendix (**ex06 eq2 ≡ L10 eq4**; eqs 3,5,6,7,8,10,11).
  **DoD:** both formalisms present; the deliberate Dec-POMDP-proxy collapse is named as a limitation; N=1 QMIX "trivial/lossy decomposition" caveat documented.

- [ ] **T10.4 — README §7.2 critical analysis (4 sub-parts)** · _A_
  (1) Non-stationarity + CTDE fix — both targets side-by-side: independent `y_i = r_i + γ(1−d)max Q_i(o'_i,a'_i)` vs centralized `y_tot = r_team + γ(1−d)max Q_tot(s',ā')`, with the drift equation `P_i(s'|s,a_i)=Σ_{a_-i} π_-i·T`. (2) IQL-vs-CTDE baseline — **empirical** (F5), identical nets/replay/seeds. (3) IGM monotonicity / lossy decomposition + the non-monotonic pincer example + **QPLEX `[10]` / Weighted-QMIX `[9]`** (exact arXiv IDs); reproduce L10 Table 3. (4) Pursuit-evasion / curriculum (Lin `[5]`, MAPPO, MADDPG/COMA `[8]`). Plus the OLoRA honest-limitations note (stability aid, not the non-stationarity cure, citing `[7]`§III + `[4]`,`[8]`) and the rejected-readings list (random-matrix-QR, LLM-bolt-on, r≥dim).
  **DoD:** all four sub-parts present with exact equations; CTDE claimed to improve **stability, not optimality**; citation-numbering hazard footnoted.

- [ ] **T10.5 — README §7.3 figures + risk register + remaining docs** · _A_
  Embed F1-F6 with §-anchored captions (caption ties the spectator/partial-observability train-global vs exec-local distinction). Add risk register to `docs/PLAN.md` + summary to README §8 (≥15 risks, P×I, mitigation+fallback+owner-phase). Author `docs/QUALITY.md` (ISO 25010), `docs/COST_ANALYSIS.md`, finalize `docs/ANALYSIS.md` (OLoRA math + rejected readings + param table).
  **DoD:** every figure referenced; risk register ≥15 entries; all required docs present and README-linked.

- [ ] **T10.6 — §9 single-parameter sensitivity sweep (DISTINCT from the ablation)** · _A_
  A controlled SINGLE-PARAMETER sweep — sweep ONE config key over a small grid while holding everything else fixed (seeds/nets/replay/γ/target cadence identical), e.g. `env.view_radius_by_grid[5]` ∈ {1,2}, OR `olora.rank` ∈ `olora.rank_sweep` {2,4,8}, OR `selfplay.window_k` ∈ {1,2,5}, OR `reward.distance_weight` ∈ {0.5,1.0,2.0}. `scripts/sensitivity_sweep.py` (thin SDK wrapper) appends to `results/runs/sensitivity_*.jsonl`; one figure plots the swept parameter vs capture-rate. This is the §9 **sensitivity analysis** and is explicitly NOT the IQL/VDN/QMIX **ablation** of T4.7 (the ablation swaps the algorithm; this sweeps one hyperparameter).
  **DoD:** exactly ONE parameter varies (others pinned, documented); `results/figures/sensitivity_<param>.png` produced + reproducible from `results/runs/`; the swept parameter's effect on capture-rate is documented in `docs/ANALYSIS.md` §9; a test asserts the sweep touched only the one config key.

- [ ] **T10.7 — §9 analysis notebook (SDK-only)** · _A_
  `notebooks/analysis.ipynb` consuming the **SDK only** (`import src.sdk` — NO direct env/marl/mcp/learner imports): renders the LaTeX Dec-POMDP/POSG + QMIX (eq 7) / VDN (eq 6) / IGM (eq 5) equations, loads + displays the F1–F6 figures from `results/figures/`, and lists the bibliography `[1]`-`[11]` citations.
  **DoD:** notebook runs top-to-bottom headless (`uv run jupyter nbconvert --execute --to notebook`); imports ONLY `src.sdk` (a test/grep asserts no `src.marl`/`src.mcp`/learner imports in the notebook); LaTeX equations + all six figures + citations present.

- [ ] **T10.8 — §11 cost analysis (`docs/COST_ANALYSIS.md`)** · _A_
  Author `docs/COST_ANALYSIS.md`: (a) AI-assisted-development token-cost breakdown — input/output token counts × `$/1M` per model used × total `$`; (b) the RL **training-compute envelope** — episodes × stages × wall-clock, LOCAL-only (no cloud training spend, BRIEF §5.2 training-must-be-local); (c) optimization notes (OLoRA ~8× fewer trainable params, seeded reuse, cache).
  **DoD:** `docs/COST_ANALYSIS.md` exists with the per-model token table (counts + $/1M + total) AND the local training-compute envelope AND optimization notes; `test_required_docs_present.py` finds `docs/COST_ANALYSIS.md`.

- [ ] **T10.9 — §13 ISO/IEC 25010 quality (`docs/QUALITY.md`)** · _A_
  Author `docs/QUALITY.md` mapping ALL EIGHT ISO/IEC 25010 characteristics to A6 evidence: **Functional Suitability, Performance Efficiency, Compatibility, Usability, Reliability, Security, Maintainability, Portability** (each with a concrete A6 artifact — e.g. Security→bearer-JWT+jti-revoke+redaction, Maintainability→≤150-LOC+SDK seam+ADRs, Portability→uv+pyproject+cloud/Render fallback).
  **DoD:** `docs/QUALITY.md` has all 8 characteristic headings, each with ≥1 concrete A6 evidence link; `test_required_docs_present.py` finds all 8 headings.

---

# Phase 11 — V3 gate audit & submission (BRIEF §11 quality)

> **Exit gate (P11):** all V3 hard gates ✅; self-grade written (to git-ignored cover sheet); code frozen 24h pre-deadline; cover sheet produced for Moodle (never asserted in a test).

- [ ] **T11.1 — Run software-excellence-v3 audit** · _A+B_
  Walk all 20 V3 sections; gather FILE-PATH/grep/command evidence (never prose). Paste real evidence into `instructions/assignment-6/submission_guidelines_audit.md`; confirm §5 (ApiGatekeeper) + §10 (UX Nielsen) now REQUIRED-and-satisfied.
  **DoD:** every hard gate has command evidence; ✅/⚠️/❌ verdict per section; all Blocker/Material gaps fixed.

- [ ] **T11.2 — Final gate sweep** · _A+B_
  Run `uv run ruff check src/ tests/ scripts/ main.py` (0), `ruff format --check`, `python scripts/check_file_sizes.py` (AFTER format), `check_no_hardcode/secrets/pii/version`, `check_no_requirements_txt` (no tracked `requirements.txt` anywhere), `uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=85`.
  **DoD:** ruff 0; every `.py` ≤150 LOC (LOC metric, excl blanks/comments); coverage ≥85%; uv-only (zero tracked `requirements.txt`); all `check_*` green.

- [ ] **T11.4 — Per-module test matrix (V3 §6)** · _A+B_
  Beyond the global ≥85% coverage gate, assert STRUCTURAL test coverage: every importable module under `src/` (every `src/<layer>/*.py`, excluding `__init__.py`) has a matching `tests/.../test_<module>.py`. `scripts/check_test_matrix.py` walks `src/` and fails listing any module with no test file; `tests/architecture/test_module_test_matrix.py` enforces it in CI.
  **DoD:** `scripts/check_test_matrix.py` reports zero un-tested `src/` modules; `test_module_test_matrix.py` GREEN; the check runs in the CI gate order before `pytest --cov`.

- [ ] **T11.3 — Self-grade + freeze + cover sheet** · _A+B_
  Write the self-grade recommendation (to git-ignored cover sheet) (drives grading strictness). Freeze code 24h before the deadline. Produce git-ignored `adrl-001-ex06.pdf` cover sheet (fill template fields only, save as PDF, no extra text) → Moodle only; **never assert it exists in a test** (skip-when-absent, MEMORY: broke A5 CI). Each member submits the identical repo URL on Moodle.
  **DoD:** self-grade written (to git-ignored cover sheet); freeze in effect; cover sheet on Moodle only; late-penalty risk (−5/24h) mitigated by P11 buffer.

---

# P-bonus — §9 inter-group +10 (post-A6, off critical path, coordination-dependent)

> **Status `[B]`:** the cross-group match needs a partner `biu-rlNN` group → **coordination dependency, not buildable solo.** The schema + interface spec ARE built now; the match is deferred. Counts toward the FINAL PROJECT, not A6.

- [B] **TB.1 — Bonus-capable report schema** · _B_
  Extend the report serializer (off by default, A6 critical path untouched): `report_type:"bonus_game"`, `groups{group_1,group_2}`, `github_repo_group_1/2`, `students_group_1/2`, per-sub-game `cop_group`/`thief_group`, `totals_by_group`, `bonus_claim`, `mutual_agreement:bool`. Subject (exact): `[MARL Bonus Game] <GroupA> vs <GroupB> – Final Report`.
  **DoD:** `build_report(..., bonus: BonusContext | None)` injects bonus keys when set; schema leaves room so no rework; A6 single-group path unchanged.

- [B] **TB.2 — Inter-group MCP interface spec** · _B_
  Publish the cop↔thief tool contracts (the SAME single canonical contract used localhost+cloud: `health`, `ping`, `new_sub_game`, `request_move` [local-obs-only], `reveal_location`, `query_opponent`, cop-only `send_final_report`, GameRef shape, `mcp.protocol_version`, Bearer/issuer/audience convention) as an **inter-group interface spec** doc so a partner group can run the opposing server against ours.
  **DoD:** `docs/INTERGROUP_MCP_SPEC.md` exists with the full wire contract + auth/token-exchange convention + which referee is authoritative; marked as a **coordination dependency** pending a partner group.

- [B] **TB.3 — Bonus match (deferred)** · _B_
  When a partner `biu-rlNN` is identified: written pre-game agreement on result-recording + a shared signed `bonus_results.json` BEFORE either sends; **freeze both policies** (no training during the bonus game); play 6 alternating-role sub-games (1-3 group1 Cop / group2 Thief; 4-6 swapped) over cloud MCP; one group sends the bonus email; document opponent/final-score/claim/screenshots in README "Bonus Game" section.
  **DoD:** counts ONLY if both groups email valid bonus reports AND `mutual_agreement:true` (else both 0); README Bonus Game section present only if the match completed (else explicitly marked not-attempted).

---

# Definition of Done (aggregate)

A task is **done** only when ALL of the following hold (not "code compiles"):

1. **TDD honored** — tests written BEFORE implementation (RED→GREEN→REFACTOR); the task's named assertions hold.
2. **Coverage** — the touched module is ≥85% covered (`uv run pytest --cov=src --cov-report=term-missing`).
3. **File size** — every `.py` it adds/edits is ≤150 LOC (LOC metric, excl blanks/comments), verified AFTER `ruff format`.
4. **Lint** — `uv run ruff check` reports 0 violations on the touched files.
5. **No hardcoding** — every algorithm-relevant value lives in `config/config.yaml` (UI-styling literals stay local in `palette.py` per CLAUDE.md §4); `check_no_hardcode.py` passes.
6. **No secrets / PII** — nothing in tracked files (`check_secrets.py` + `check_pii.py` pass); real values only in git-ignored `.env` / repo-root `players.local.yaml`.
7. **Single SDK entry** — UIs/servers/scripts import only `src.sdk` (`test_sdk_single_entry.py`); env/marl/mcp are library code.
8. **CTDE invariant** — execution path (policy / MCP `request_move`) receives ONLY local obs + legal actions; global state `s` is unreachable at execution (import-boundary + leak tests).
9. **Evidence captured** — any §7.3 deliverable (figure / screenshot / log) is produced, committed to the named path, and reproducible from `results/runs/` or a script.
10. **Docs/ADRs** — any human-decided change (new public API, changed assertion, rule decision) has an approved ADR/PRD/PLAN edit FIRST (CLAUDE.md §1.4).
11. **Phase gate** — the phase's exit gate is met before the next phase's deliverables land (BRIEF §6 order, ADR-D10-F).

---

# V3 hard-gate checklist (map to this project)

| # | V3 hard gate | How A6 satisfies it | Enforced by |
|---|---|---|---|
| 1 | **≤150 LOC/file** (excl blanks/comments) | env/mixers/learner/mcp/gui/reporting all split into ≤150-LOC modules; cop/thief servers split from `tools.py`; mixer split from q-net | `scripts/check_file_sizes.py` (AFTER `ruff format`); `test_final_gates.py` |
| 2 | **≥85% coverage** | DI + mocked `httpx` peer + `FakeEmailSender` + headless SDL; pure `transition.py`/`reward.py`/`gatekeeper`/`builder` unit-tested | `pytest --cov=src --cov-fail-under=85` |
| 3 | **Ruff 0** | `ruff check` + `ruff format --check`; PLR2004 ignored in tests only | CI steps 4-5 |
| 4 | **0 hardcoded values → config** | grid/moves/games/barriers/scoring/ports/hyperparams/seeds in `config.yaml`; UI literals local in `palette.py` (CLAUDE.md §4) | `check_no_hardcode.py`; `test_config_single_source.py` |
| 5 | **0 secrets + `.env-example`** | tokens/JWT keys/App-Password/PII in `.env` + repo-root `players.local.yaml`; `.env-example` names-only committed; `.gitignore` covers `.env *.pem *.key credentials*.json players.local.yaml secrets/` | `check_secrets.py`; `test_no_secrets_committed.py` |
| 6 | **uv-only (NO requirements.txt)** | CI uses `uv`; no pip/conda; `uv.lock` committed; deploy deps come from `pyproject.toml` / `uv export` — **NO `deploy/requirements.txt`** (V3 forbids requirements.txt) | CI `uv sync --frozen`; `check_no_requirements_txt.py` |
| 7 | **Single SDK entry for UIs** (scripts exempt) | GUI/MCP/report import only `src.sdk`; scripts are thin wrappers | `test_sdk_single_entry.py`; `test_mcp_servers_have_no_logic.py` |
| 8 | **Version = `1.0.0`** | `src/__init__.py::__version__` == `config.version` == `"1.0.0"` (3-segment mapping of V3 "1.00", ADR-0011) | `check_version.py`; `test_version_consistency.py` |
| 9 | **§5 External-API governance** (N/A→**REQUIRED** for A6) | all peer-MCP + Gmail egress via `src/api/gatekeeper.py` + `config/rate_limits.json` | `test_egress_via_gatekeeper.py`; ADR-0006 |
| 10 | **§10 UX / Nielsen** (N/A→**REQUIRED** — GUI mandatory) | `docs/UX.md` maps all 10 heuristics + screenshot per state | `test_required_docs_present.py` |
| 11 | **PRD/PLAN/TODO + ADRs** | `docs/PRD.md`, `docs/PLAN.md`, this `docs/TODO.md`, `docs/prd/*`, ADRs 0001-0014 + dimension ADRs; human §1.4 sign-off before code | `test_required_docs_present.py` |
| 12 | **Cover sheet git-ignored** | `adrl-001-ex06.pdf` ignored, Moodle-only | `test_cover_sheet_gitignored.py` (**skip-when-absent** — MEMORY) |
| 13 | **CI hygiene** | gate order `ruff check → ruff format --check → check_file_sizes → check_* → pytest --cov`; `fetch-depth:0`; never pin mutable git SHAs in tests; CI never sends Gmail / calls cloud MCP (dry-run + mocks) | `.github/workflows/ci.yml` |
| 14 | **Docstring gate (Ruff `D`)** (#18) | Ruff `D` added to the lint `select` (per-file `D`-ignore for `tests/`); every package `__init__.py` carries a one-line module docstring | `ruff check` (`D` rules); CI step 4 |

**Carried lessons (MEMORY):** add `rmisegal` as read collaborator on the repo; never assert a git-ignored cover sheet exists (skip-when-absent); Google Drive can silently delete `.git` mid-session — push often and keep a `/tmp` clone to restore `.git`.
