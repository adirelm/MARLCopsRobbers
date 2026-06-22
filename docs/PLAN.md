# PLAN — Assignment 6: MARL Cops & Robbers (adrl-001) — v1.0.0

> **Group code:** `adrl-001` · **Version:** `1.0.0` · **Solo submission** (work-streams WS1 / WS2, §10 R16)
> **Source of truth:** `planning/BRIEF.md` (distilled `ex06.pdf`, Dr. Yoram Segal, 2026) +
> `L10-MARL.pdf`. Governed by the V3 software-excellence guidelines and `CLAUDE.md` §1.4.
>
> This is the **foundational architecture & plan** document. It carries the C4 component
> view, the full repository tree, the train-vs-execute data-flow, the ADR index, the
> localhost→cloud deployment diagram, and the strict §6-ordered phased build sequence.
> It is the human-signed-off *architecture* artifact required before any feature code lands
> (`CLAUDE.md` §1.4: human = architect, AI = implementer).
>
> **Companion docs:** `docs/PRD.md` (requirements/KPIs), `docs/TODO.md` (phase-tagged tasks),
> `docs/THEORY.md` (Dec-POMDP/CTDE math), `docs/ANALYSIS.md` (OLoRA + §7.2 critique),
> `docs/UX.md` (Nielsen heuristics), `docs/QUALITY.md` (ISO 25010), `docs/COST_ANALYSIS.md`,
> the ADR index in §6 of this doc (ADR-0001..0014 + supporting-policy), `docs/schema/report.schema.json`.

---

## 1. Scope & success criteria (human-decided per §1.4 — non-delegable)

**Locked scope = full ambition** (architect directive: *maximize the grade; complexity/time
acceptable*). The following are all in scope for the A6 build:

| In scope | Detail | Brief anchor |
|---|---|---|
| MARL + CTDE | QMIX primary (monotonic mixer) + VDN ablation + IQL baseline; recurrent GRU Q-nets; centralized episodic replay | §2.1, §5.2, §7.2 |
| Adversarial self-play | Cop team (cooperative, value-decomposed) vs Thief (separate adversarial Double-DQN folded into `T`); alternating best-response | §2.1, §5.2 |
| OLoRA | Full QR-on-pretrained-weights ablation (OLoRA vs full-finetune vs frozen-base) | §5.2, [7] |
| Two FastMCP servers | Cop + Thief, separate OS processes, HTTP comms; **localhost (Stage 1) AND cloud deploy (Stage 2)** | §5.3, §6.8, §8 |
| Pygame GUI | Real-time god-view spectator at 2×2/3×3/4×4/5×5 | §5.4, §7.3c |
| Gmail report | One auto email by the Cop at whole-game end (smtplib + App-Password, OAuth fallback) | §3.5, §5.5, §6.9 |
| §9 bonus capability | Bonus-capable report schema **built** + inter-group MCP interface spec **written** | §9 |

**Coordination dependency (not buildable solo):** the actual §9 cross-group match needs a
partner `biu-rlNN` group. We ship the *schema* and the *interface spec*; the live bonus match
is marked a coordination dependency, off the A6 critical path.

**Success criteria (KPIs):**
1. A full 6-sub-game match completes over **localhost** MCP (Stage 1) and again over **cloud**
   MCP URLs (Stage 2), each emitting a schema-valid §3.5 JSON body.
2. On the final 5×5 1-cop-vs-1-thief match, the trained Cop's capture rate **strictly exceeds**
   both the Manhattan-heuristic floor and the IQL baseline on held-out seeds `[7,17,37,71,107]`.
3. The plotted §7.3 figures (F1/F2/F5/F6) regenerate from one command, traceable to
   `results/runs/*.jsonl`; F3 (GUI) and F4 (MCP-comms) are deterministically **captured**, not plotted.
4. All V3 hard gates green (see §9 of this PLAN).

**Cost-budget envelope (human-decided):** the full training envelope is **~100k episodes**
(5000 episodes/stage × 4 curriculum stages × 5 seeds, per PRD FR-COST-1) and is **local-only**
(§5.2); cloud is used only for the demo match (a 6-sub-game match is <500 requests, well inside
Prefect Horizon free tier); tear down public services after the grading proof is captured to save
compute minutes. **Defer-order (if behind schedule):** drop in order **P8 cloud → OLoRA ablation
arm → P-bonus** (see R2 / R12 / R16 / P-bonus) — the localhost match + QMIX/VDN/IQL comparison
stay on the critical path.

---

## 2. C4-style architecture (component view)

### 2.1 Level 1 — System context

```
                         ┌─────────────────────────────────────────────┐
        ┌──────────┐     │      MARL Cops & Robbers System (adrl-001)    │     ┌──────────────┐
        │  Student │────▶│  trains locally, plays a 6-sub-game match,    │────▶│  Lecturer    │
        │ (solo)   │ uv  │  visualizes it, emails the final report       │ SMTP│  rmisegal     │
        └──────────┘ run │                                               │ 587 │ +marl@gmail  │
                         └───────────────┬─────────────────┬────────────┘     └──────────────┘
                                         │ HTTPS (JWT)      │ git push-to-deploy
                                         ▼                  ▼
                              ┌────────────────┐   ┌──────────────────────┐
                              │ Prefect Horizon │   │  GitHub (shared repo, │
                              │ (FastMCP Cloud) │   │  read collab: rmisegal)│
                              └────────────────┘   └──────────────────────┘
```

### 2.2 Level 2 — Container view (the 3 runtime processes + the single SDK seam)

The system runs as **three OS processes** (processes, not threads — avoids the FastMCP-async ↔
Pygame ↔ torch event-loop deadlock, ADR-0011) plus offline trainers. **Every container imports
the single `MarlSDK` facade — never `src/env` or `src/marl` directly** (V3 single-entry; ADR-0002).

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                          PROCESS A — Referee / Orchestrator (run_match.py)              │
│  THE ENVIRONMENT + CTDE global-state holder. Owns ground-truth s (positions,           │
│  barriers, clock, capture rule, scoring, technical-loss replay → exactly 6 valid).     │
│  Builds O(s,i) = bounded-Manhattan local obs per agent. Drives the turn loop.          │
│  Holds the spectator read path (in-proc Stage 1 / HTTP /spectator Stage 2).            │
└───────┬───────────────────────────────────────┬───────────────────────────────┬───────┘
        │ HTTP  request_move(Observation o_cop)  │ HTTP request_move(o_thief)    │ read-only
        ▼  (Bearer JWT)                          ▼  (Bearer JWT)                 ▼ (SDK / SSE)
┌────────────────────────┐            ┌────────────────────────┐      ┌─────────────────────┐
│ PROCESS B — Cop MCP     │◀──HTTP────▶│ PROCESS C — Thief MCP   │      │ Pygame GUI          │
│ server (FastMCP :8001)  │ reveal_    │ server (FastMCP :8002)  │      │ (god-view spectator)│
│ private CopNet (OLoRA)  │ location   │ private ThiefNet (OLoRA)│      │ SpectatorFrame only │
│ cop-only send_final_    │ (radius-   │                         │      │ never calls servers │
│ report → Gmail          │  gated)    │                         │      └─────────────────────┘
└────────────────────────┘            └────────────────────────┘
        ▲ loads actor-only .pt + sidecar       ▲ loads actor-only .pt + sidecar
        │  (NO mixer / NO replay / NO global s)  │
┌───────┴────────────────────────────────────────┴───────────────────────────────────────┐
│  OFFLINE (local-only): Trainers — BC pretrain → OLoRA-attach → CTDE self-play up the     │
│  curriculum. Mixer + replay + global state live HERE and are NEVER exported. Emits        │
│  cop_weights.pth / thief_weights.pth + results/runs/*.jsonl (telemetry for §7.3).        │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Level 3 — Component view (inside the boxes)

```
src/sdk/sdk.py :: MarlSDK  ── SINGLE business-logic entry; every UI/server/script depends on it
   build_env · train(algorithm,seed) · build_policy(role,algo,stage) · eval_policy · export_weights
   spectator_session · play_full_game · build_report_json · send_final_report

ENVIRONMENT (src/marl/env/, pure library)              LEARNER (src/marl/, src/services/)
  actions  types  grid  transition(PURE)                 recurrent_q_net (GRU, eq 8)
  observation(O, 5×5 padded)  reward(potential)          mixers/{base,vdn,qmix} + learner branches (seam)
  scorer(Table-1, SEPARATE)  curriculum                  learner_base → cop/thief/iql learners
  cops_robbers_env(reset/step→TransitionResult/state)    episode_buffer(centralized, train-only)
  render_state(ONLY other global accessor, spectator)
                                                          olora(QR init, eq 10-11)
DATA PLANE (src/marl/data/, src/marl/nets/)              selfplay(best-response)  trainer(curriculum)
  schemas(TransitionDTO)  heuristics(A*/flee experts)    referee = THE ENVIRONMENT (MatchRunner)
  obs_encoder(pad-to-5×5)  bc_dataset  replay           MCP (src/mcp/)
  encoder  olora_linear  olora_init  agent_net  bundles   schemas  actions(re-export env)  auth(Static→JWT+jti)
                                                          agent_runtime(private AgentController)
EGRESS GOVERNANCE (src/api/, §5 REQUIRED — ADR-0009)      cop_server  thief_server  clients
  ApiGatekeeper.execute(call) + .get_queue_status()        (per-channel token-bucket: peer_mcp/gmail/
  FIFO overflow queue (no crash on burst), logs all calls   prefect_deploy from config/rate_limits.json)
  http_client(bearer)  ── ALL outbound egress routes here
                                                          GUI (src/gui/)  REPORTING (src/reporting/)
RESULTS (src/results/)                                     app transform grid_view hud score_view   schema collector players
  aggregate  plots  make_figures (F1-F6, 1 command)       palette input_map state_client            builder mailer redact idempotency
```

**Hard architectural seams (machine-enforced by `tests/architecture/`):**
- **Single SDK entry:** GUI / MCP servers / reporting / scripts import only `src.sdk` (regex test).
- **CTDE structural split:** runtime/MCP code **cannot import** `mixer`, `episode_buffer`,
  `learner_base`, `GlobalState`, or anything under `services/` (import-boundary test). Exported
  `.pth` contains **agent-net params only** + a shape sidecar — global state is unreachable at exec.
- **Algorithm seam** across IQL/VDN/QMIX: one shared `RecurrentQNet` + one buffer; VDN/QMIX swap only
  the Mixer ABC, while **IQL additionally drops the mixer + global state** and uses an independent
  per-agent target (`IqlLearner._compute_target`, eq 4) — the seam is mixer-OR-learner-branch, not mixer-only.
- **Egress via gatekeeper:** no module calls `httpx`/Gmail/Prefect outside `src/api/` + `src/reporting/mailer.py`; ALL peer-MCP, Gmail, and Prefect-deploy egress routes through `ApiGatekeeper.execute()`.
- **Spectator purity:** GUI never imports referee/agent/MCP internals; `SpectatorFrame` frozen;
  `referee.build_local_obs` Manhattan-masks beyond `view_radius` (encodes the §2.1 Dec-POMDP invariant).

---

## 3. Repository directory tree

Flat `src/<layer>/` layout (matches A1–A5 grader muscle memory; ADR-0001). **Every `.py` ≤150 LOC**
(LOC = code lines excl. blanks/comments, measured *after* `ruff format`). One-line purpose per module.

```
Assignment6-MARLCopsRobbers/
├── README.md                       # §7 academic paper body + 6 user-manual sections + F1-F6
├── CLAUDE.md                       # global standards + §1.4 Human↔AI contract (adapted from A1)
├── LICENSE                         # MIT
├── pyproject.toml                  # name=marl-cops-robbers, version="1.0.0", uv groups dev/mcp/mail
├── uv.lock                         # committed (uv-only hard gate)
├── .python-version                 # 3.11
├── .env-example                    # tokens + Gmail + (PII handled via secrets/, see ADR-0013) — committed
├── .gitignore                      # .env .env.* *.pem *.key credentials*.json secrets/ adrl-001-ex*.pdf instructions/ artifacts/
│                                    #   + allow-list exception: !deploy/model/*.pt (reserved for the Stage-2 actor checkpoint, committed at deploy time)
├── .github/
│   ├── pull_request_template.md    # Scope/Req/Human/AI/Tests/V3 checklist
│   └── workflows/ci.yml            # ruff check → ruff format --check → file-size → check_* → pytest --cov
├── config/
│   ├── config.yaml                 # ALL numeric params (single source of truth, §4 below)
│   ├── config.schema.yaml          # typed schema; loader _validate→ValueError
│   └── rate_limits.json            # §5 gatekeeper limits {peer_mcp, gmail, prefect_deploy} (versioned)
├── players.example.yaml            # repo-root placeholder role A — SOLO (tracked, NO real PII; PRD §1.3)
├── src/
│   ├── __init__.py                 # __version__ = "1.0.0"
│   ├── sdk/
│   │   ├── sdk.py                  # MarlSDK facade — SINGLE entry (delegates to _helpers)     ~120
│   │   └── _helpers.py            # SDK build/train/eval/report helper fns (split for ≤150)    ~110
│   ├── utils/
│   │   └── config_loader.py        # load_config(): yaml + .env interpolation + validation     ~110
│   ├── marl/
│   │   ├── env/
│   │   │   ├── actions.py          # SINGLE source: Action IntEnum, DELTAS, action_mask(state, agent) ~70
│   │   │   ├── types.py            # Pos, GlobalState @dataclass, Observation TypedDict        ~60
│   │   │   ├── grid.py             # bounds, manhattan(), in_grid(), can_enter(), spawn        ~90
│   │   │   ├── transition.py       # resolve_joint_action(...) -> TransitionResult [PURE]       ~120
│   │   │   ├── observation.py      # O-function: builds 5×5-padded egocentric image + 6 scalars ~110
│   │   │   ├── obs_channels.py     # channel/scalar encoders split out of observation.py (≤150) ~90
│   │   │   ├── reward.py           # potential-based shaped R + per-agent R_i (Ng 1999)        ~110
│   │   │   ├── scorer.py           # Table-1 game points 20/10/5/5 (SEPARATE from reward)      ~60
│   │   │   ├── curriculum.py       # CurriculumScheduler: Table-2 stages + promotion gate      ~110
│   │   │   ├── cops_robbers_env.py # CopsRobbersEnv: reset/step(4-tuple)/state                  ~120
│   │   │   └── render_state.py     # render_state() spectator snapshot split out of env (≤150)  ~70
│   │   ├── nets/
│   │   │   ├── encoder.py          # flat-obs Linear trunk (encoder_hidden, OLoRA-wrapped); NO conv ~70
│   │   │   ├── olora_linear.py     # OLoRALinear: frozen W0 + Parameter A,B; forward eq 11     ~90
│   │   │   ├── olora_init.py       # qr_init(W0,r)→(A,B); rebase eq 10; wrap_encoder()         ~100
│   │   │   ├── agent_net.py        # AgentNet = wrapped Encoder + plain Q-head (cop 5/thief 4) ~120
│   │   │   └── bundles.py          # save/load {base_sha, adapters, head}; base_sha guard      ~90
│   │   ├── recurrent_q_net.py      # GRUQNet: flat-obs Linear trunk → GRUCell → role Q-head, eq 8 ~120
│   │   ├── mixers/
│   │   │   ├── base_mixer.py       # Mixer ABC: forward(q_agents, state) → q_tot               ~30
│   │   │   ├── vdn_mixer.py        # sum over cop team, eq 6                                    ~30
│   │   │   └── qmix_mixer.py       # monotonic hypernetwork, softplus(weights)≥0, eq 7         ~130
│   │   │                           #   (NO iql_mixer.py — IQL has no mixer; it branches in IqlLearner)
│   │   ├── learner_base.py         # QmixLearner: update/_compute_target/_sync/save/load       ~145
│   │   ├── cop_learner.py          # CopLearner(QmixLearner): num_cops, A=5, reward hook       ~45
│   │   ├── thief_learner.py        # ThiefLearner(QmixLearner): num_agents=1, A=4             ~40
│   │   ├── iql_learner.py          # IqlLearner: override _compute_target → eq 4, no mixer;     ~55
│   │   │                           #   single-param-group optimizer (drops QMIX's lr_mixer group)
│   │   ├── policy.py               # ε-greedy + greedy DECENTRALIZED exec path (local obs)     ~80
│   │   ├── episode_buffer.py       # CentralizedEpisodeBuffer: padded episodes + global s      ~140
│   │   └── data/
│   │       ├── schemas.py          # frozen TransitionDTO + DatasetManifest                    ~80
│   │       ├── heuristics.py       # barrier-aware A*/BFS cop expert + flee thief expert       ~130
│   │       ├── obs_encoder.py      # local Manhattan-radius patch → tensor (pad-to-5×5, mask)  ~110
│   │       ├── bc_dataset.py       # generate + load (o,a*) BC pairs; npz IO                   ~100
│   │       └── replay.py           # CentralizedReplayBuffer (local obs + global s + joint a)  ~140
│   ├── services/
│   │   ├── rollout.py              # collect ONE episode driving both ε-greedy policies        ~110
│   │   ├── trainer.py              # SelfPlayTrainer: curriculum + OLoRA per stage + history   ~145
│   │   ├── selfplay.py             # alternating best-response, frozen window, opponent pool   ~120
│   │   ├── checkpoints.py          # full(train) vs export(deploy) split + shape sidecar       ~70
│   │   └── referee.py              # MatchRunner = THE ENVIRONMENT: ground-truth s, observe()=O(s,i), apply() ~145
│   ├── api/
│   │   ├── gatekeeper.py           # ApiGatekeeper.execute(call)+get_queue_status(): per-channel  ~120
│   │   │                           #   token-bucket (peer_mcp/gmail/prefect_deploy) + FIFO overflow
│   │   │                           #   queue (no crash on burst) + call log (§5, ADR-0009)
│   │   └── http_client.py          # bearer-auth HTTP wrapper (timeout/retry/backoff)         ~90
│   ├── mcp/
│   │   ├── schemas.py              # pydantic v2: Observation/MoveResponse/LocationReveal/...  ~140
│   │   │                           #   every request carries session_id (GRU keying + idempotency)
│   │   ├── actions.py              # re-exports Action/DELTAS/mask from marl.env.actions (no dup) ~15
│   │   ├── auth.py                 # build_verifier: Static(local)|JWT+jti-denylist(cloud)     ~120
│   │   ├── agent_runtime.py        # private AgentController: Observation→action via SDK;       ~110
│   │   │                           #   holds per-session GRU z_t (reset on new_sub_game)
│   │   ├── cop_server.py           # FastMCP("cop"): request_move/reveal_location/query_opponent ~130
│   │   │                           #   /new_sub_game/send_final_report (SAME contract localhost==cloud)
│   │   ├── thief_server.py         # FastMCP("thief"): request_move/reveal_location/query_opponent/new_sub_game ~100
│   │   └── clients.py              # make_client(url,token); typed wrappers + structured log   ~120
│   │                               #   (referee.py lives in src/services/ — it is THE ENVIRONMENT,
│   │                               #    so it is OUT of the test_mcp_servers_have_no_logic scope)
│   ├── gui/
│   │   ├── app.py                  # owns SDK + StateClient; update()/draw(); rebuild on resize ~110
│   │   ├── transform.py            # GridView (row,col)→pixel Rect; capped square cells 2..5    ~50
│   │   ├── grid_view.py            # board back-to-front: cells/barriers/tokens/capture flash   ~95
│   │   ├── hud.py                  # sub-game i/6, move k/25, scores, last actions, banner       ~70
│   │   ├── score_view.py           # right rail: 6-row results table + running totals            ~70
│   │   ├── palette.py              # LOCAL render literals (colors/px/FPS); NO config import      ~35
│   │   ├── input_map.py            # keys→commands (pause/speed/next/replay/view-radius)         ~35
│   │   └── state_client.py         # source-agnostic SpectatorFrame: in-proc OR SSE/poll         ~90
│   ├── reporting/
│   │   ├── schema.py               # @dataclass Student/SubGame/MatchReport + to_dict/validate  ~130
│   │   ├── collector.py            # MatchRecorder: zoneinfo timestamps, per-sub-game capture    ~110
│   │   ├── players.py              # load_players() from git-ignored secrets/players.local.yaml  ~70
│   │   ├── builder.py              # build_report() + winner→scores map + format_subject()       ~110
│   │   ├── mailer.py               # EmailSender Protocol + GmailMailer(smtplib) + Fake          ~120
│   │   ├── redact.py               # write role-only redacted JSON to results/reports/           ~60
│   │   └── idempotency.py          # sha256 canonical hash + sentinel read/write                 ~50
│   └── results/
│       ├── aggregate.py            # per-method per-seed mean±SE, win-rate, time-to-capture      ~140
│       ├── plots.py                # matplotlib helpers (alpha/fontsize/dpi local literals OK)   ~120
│       └── make_figures.py         # SINGLE entrypoint: runs/*.jsonl → F1/F2/F5/F6 (F3/F4 captured) ~140
├── scripts/                        # uv-run entrypoints; SDK-only; exempt from SDK rule, NOT size/lint
│   ├── check_file_sizes.py         # ≤150 LOC gate (copy A5 verbatim)
│   ├── check_no_hardcode.py        # grep magic game literals in src/
│   ├── check_secrets.py            # git ls-files secret scan
│   ├── check_pii.py                # deny-list scan over tracked files
│   ├── check_version.py            # __version__ == config.version == "1.0.0"
│   ├── pretrain_bc.py              # Stage A: BC on Manhattan-heuristic data → base checkpoints
│   ├── attach_olora.py             # Stage B: freeze + QR-attach → wrapped bundles
│   ├── finetune_ctde.py            # Stage C: hand wrapped encoders to CTDE trainer
│   ├── train_ctde.py               # thin wrapper → sdk.train (qmix/vdn)
│   ├── train_iql_baseline.py       # thin wrapper → sdk.train (iql)
│   ├── run_cop_server.py           # launch Cop MCP server
│   ├── run_thief_server.py         # launch Thief MCP server
│   ├── play.py                     # GUI launcher (window loop only under __main__)
│   ├── capture_screens.py          # headless deterministic GUI screenshots (F3)
│   ├── mcp_tokens.py               # CLI: issue | list | revoke JWTs (by jti/subject)
│   ├── smoke_local.sh              # Stage-1 two-port bearer round-trip probe
│   ├── smoke_cloud.py              # Stage-2 cross-server request_move/reveal_location proof
│   ├── smoke_auth.py              # bad-token + revoked-token 401 proof
│   ├── smtp_smoke.py               # one-line Gmail App-Password pre-flight
│   └── redact_logs.py              # strip Authorization/Bearer before screenshots
├── tests/
│   ├── conftest.py                 # SDL_VIDEODRIVER=dummy; tmp config; fake MCP peer; mock httpx/Gmail
│   ├── unit/                       # env, reward, mixer, olora, gatekeeper, report-builder, config
│   ├── integration/               # full 6-sub-game localhost match (dry-run Gmail); train smoke
│   └── architecture/               # V3 GATE TESTS (§2.3 seams)
├── deploy/                         # NO requirements.txt (uv-only gate); Horizon/Render read
│   │                               #   pyproject.toml + uv.lock — deps generated on demand via `uv export`
│   ├── render.yaml                 # fallback host blueprint (two web services; `uv sync --frozen` build cmd)
│   ├── runbook.md                  # ordered, copy-pasteable cloud runbook (incl. the `uv export` step)
│   └── model/                      # ONE committed actor-only inference checkpoint (MODEL_PATH target)
│       ├── cop_actor.pt            # GRU+MLP state_dict, actor weights ONLY (sub-MB; .gitignore allow-listed)
│       └── cop_actor.shape.json    # shape sidecar (obs/action dims) — load-time guard
├── docs/
│   ├── PRD.md  PLAN.md  TODO.md
│   ├── THEORY.md  ANALYSIS.md  QUALITY.md  UX.md  COST_ANALYSIS.md
│   ├── prd/{PRD-ENV,PRD-CTDE,PRD-MCP,PRD-REPORT}.md
│   ├── adr/0001..0014.md
│   ├── schema/report.schema.json   # draft-2020-12 §3.5 report contract (grading evidence)
│   ├── interfaces/intergroup_mcp.md# §9 inter-group MCP interface spec (coordination dependency)
│   └── shared/PROMPTS.md           # literal prompts used (§1.4 evidence)
├── results/
│   ├── README.md                   # what lives here; heavy artifacts git-ignored
│   ├── runs/*.jsonl                # append-only per-episode telemetry (TRACKED: small, needed for figure regen)
│   ├── figures/                    # F1-F6 PNGs (committed for README)
│   ├── screenshots/                # GUI @2/3/4/5 + MCP CLI-log proof (§7.3 c,d)
│   ├── reports/*.redacted.json     # role-only report evidence (real *.real.json git-ignored)
│   └── experiment_manifest.json    # seeds, config hashes, git commit, run IDs
├── secrets/                        # GIT-IGNORED: players.local.yaml (real names/ids)
├── instructions/                   # GIT-IGNORED: brief PDFs, lecturer feedback, audit, cover sheet
└── notebooks/analysis.ipynb        # consumes MarlSDK ONLY; LaTeX equations
```

---

## 4. Config — single source of truth (`config/config.yaml`, §3.6, no hardcoding)

All algorithm-relevant numerics live here and are read via `config_loader`. UI render literals
(colors, cell px, FPS, matplotlib dpi/alpha) stay LOCAL in their render modules (`CLAUDE.md` §4).

This sample MIRRORS `config/config.yaml` (the frozen single source of truth) — the SAME nested
keys, ports, and values. Inline comments are abbreviated; the file itself carries the §-anchors.

```yaml
version: "1.0.0"
project: { group_code: "adrl-001", timezone: "Asia/Jerusalem" }
game:                                                                         # §3 graded 5×5 match
  grid_size: 5
  max_moves: 25
  num_games: 6
  max_barriers: 5
  scoring: { cop_win: 20, thief_win: 10, cop_loss: 5, thief_loss: 5 }         # §3.4 Table 1 (report-only)
env:
  num_cops: 1                  # graded match = 1 cop; curriculum.num_cops_by_stage overrides per stage
  move_resolution: simultaneous# simultaneous | cop_first | thief_first  (ADR-0004)
  capture_on_swap: true        # cop↔thief one-tick swap = CAPTURE (ADR-0004, architect rule call)
  reward_mode: dec_pomdp       # dec_pomdp | posg
  view_radius_by_grid: { 2: 0, 3: 1, 4: 1, 5: 2 }   # per-grid view radius (auto = ceil(min(H,W)/2)-1)
  view_radius_max: 2           # padding footprint = 2*view_radius_max+1 = 5×5
  obs_channels: 5              # self, other_visible, barrier, out_of_bounds, time_norm
  obs_scalars: 6               # own_r/c_norm, step_norm, barriers_left_norm, other_seen_now, steps_since_seen_norm
  actions:                     # brief: both move one cell each turn
    enable_stay: false         # STAY config-gated OFF
    enable_barrier: true       # §3.3 PLACE_BARRIER cop-only; placement IS the move
    a_cop: 5                   # {UP,DOWN,LEFT,RIGHT,PLACE_BARRIER}
    a_thief: 4                 # {UP,DOWN,LEFT,RIGHT}
  curriculum:
    stages: [[2,2],[3,3],[4,4],[5,5]]
    num_cops_by_stage: [1, 1, 2, 1]   # ADR-0007: 4×4 trains a 2-cop team (non-trivial mixer); 5×5 stays 1
    promotion_threshold: 0.8
    promotion_window: 100
algo:                          # was top-level `marl:` — now nested under `algo:`
  name: "qmix"                 # qmix(primary) | vdn(ablation) | iql(baseline)  (ADR-0003)
  double_q: true
  gamma: 0.99
  lr_agent: 3.0e-4
  lr_mixer: 5.0e-4
  weight_decay: 1.0e-5
  grad_clip_norm: 10.0
  huber_delta: 1.0
  batch_episodes: 32
  target_update_interval: 200            # hard target sync (the only mode implemented)
  mixer: { type: qmix, embed_dim: 32 }   # 2-cop 4×4 stage supplies the non-trivial-mixer evidence
nets:        { hidden_dim: 64, encoder_hidden: [64, 64] }   # flat-obs MLP trunk; NO conv (#5); recurrent GRU
olora:       { rank: 4, rank_sweep: [2, 4, 8], scale: 8.0, target_layers: ["encoder"] }  # assert rank≤min(m,n)//2; default OFF (enabled via the SDK olora arg)
bc:          { epochs: 30, lr: 1.0e-3, epsilon: 0.1, pretrain_grids: [[2,2],[3,3]], n_pairs: 40000, val_fraction: 0.2, split_seed: 7, val_acc_gate_by_grid: {2: 0.50, 3: 0.78} }  # per-grid honest gate
replay:      { buffer_episodes: 5000, min_replay_episodes: 256 }   # cadence = selfplay.update_ratio
selfplay:    { window_k: 1, pool_size: 5, update_ratio: 1, rounds: 50 }      # window_k=update-ratio tick; rounds/stage
training:
  seeds: [7, 17, 37, 71, 107]
  eps_start: 1.0
  eps_end: 0.05
  eps_anneal_env_steps: 20000
reward:                        # RL TRAINING SIGNAL ONLY (potential-based; SEPARATE from game.scoring)
  shaping_enabled: true
  shaping_eval_enabled: false
  capture_bonus: 1.0           # terminal +1 dominant signal on capture
  timeout_penalty: 1.0         # terminal -1 cop / +1 thief on timeout
  step_penalty: 0.05
  distance_weight: 1.0
  barrier_cost: 0.25
  invalid_penalty: 0.1
  evade_step: 0.1              # posg-mode thief per-step survival
  timeout_bonus: 1.0          # posg-mode thief terminal evade
mcp:                           # was nested cop/thief/cloud blocks — now flat, matching config.yaml
  host: "127.0.0.1"
  cop_port: 8001               # Cop FastMCP server
  thief_port: 8002             # Thief FastMCP server
  path: "/mcp"
  transport: http
  protocol_version: "adrl-001-mcp-v1"
  observation: { view_radius: 2 }
  client: { timeout_s: 10, max_retries: 3, backoff_s: 0.5, prewarm_ping: true }
  auth:
    scheme: bearer
    issuer: "adrl-001-mcp-auth"
    algorithm: RS256
    cop_audience: "marl-cop"
    thief_audience: "marl-thief"
    required_scopes: ["game:write"]
cloud:                         # was `mcp.cloud` — now top-level `cloud:`
  platform: prefect_horizon
  cop_url: ""                  # e.g. https://adrl-001-cop.fastmcp.app/mcp (set at deploy)
  thief_url: ""                # e.g. https://adrl-001-thief.fastmcp.app/mcp (set at deploy)
  fallback_platform: render
  rate_limits: { peer_mcp_per_minute: 120, peer_mcp_burst: 10, gmail_per_minute: 5, gmail_burst: 1 }
gmail:                         # was `report:` + `email:` — now unified under `gmail:`
  to: "rmisegal+marl@gmail.com"
  smtp_host: "smtp.gmail.com"
  smtp_port: 587
  subject_template: "[MARL Cops&Robbers] {group_name} | Final {num_games} sub-games | Cop {cop_total} - Thief {thief_total} | {date}"
  output_dir: "results/reports"
  sentinel: "results/.report_sent"
gui:         { screenshot_sizes: [2, 3, 4, 5], screenshot_dir: "results/screenshots" }
paths:
  runs_dir: "results/runs"
  figures_dir: "results/figures"
  experiment_manifest: "results/figures/experiment_manifest.json"
logging:     { level: "INFO" }
```

`config/rate_limits.json` (versioned, git-tracked, no secrets) →
`{ "version":"1.0.0", "limits": { "peer_mcp":{"per_minute":120,"burst":10},
"gmail":{"per_minute":5,"burst":1}, "prefect_deploy":{"per_minute":30,"burst":5} },
"overflow_policy":"fifo_queue", "on_overflow":"enqueue", "max_queue":256, "log_all_calls":true }`
`.env-example` (names only): `COP_MCP_TOKEN= THIEF_MCP_TOKEN= REFEREE_MCP_TOKEN= PEER_MCP_TOKEN= MCP_PUBLIC_KEY= MCP_PRIVATE_KEY= REVOKED_TOKEN_JTIS= PREFECT_API_KEY= GMAIL_SENDER= GMAIL_APP_PASSWORD= REPORT_RECIPIENT_OVERRIDE=`
Secrets/PII NEVER in `config.yaml`: transport secrets + RSA keys → `.env`; real names/ids +
GitHub owner slug → git-ignored `players.local.yaml` (placeholders in repo-root `players.example.yaml`).

---

## 5. Data-flow — train-time global state vs exec-time local obs

This split is the single most-graded modeling requirement (BRIEF §4, §6.1, §7.1; L10 §4). It is
made **structural**, not a matter of discipline: the global state `s` exists only in the
referee/trainer; it is physically unreachable on the execution path.

### 5.1 Train-time (CTDE — global state allowed)

```
CurriculumScheduler ── grid_size ──▶ CopsRobbersEnv
                                       │ env.step(joint_a) → (LOCAL obs_dict, reward_i, done, info)
                                       │ env.state()       → GlobalState s   ◀── TRAIN ONLY
                                       ▼
rollout.collect ── per-role episode dicts ⟨o, a, r, o', done, GlobalState s, active, hidden seed⟩
   (producer widens a k-cop episode to the N-wide buffer: real cops in slots 0..k-1, slots k..N-1
    zero-filled, and stamps the EPISODE-CONSTANT per-slot mask active[j] = (j < k) — P4d)
                                       ▼
CentralizedEpisodeBuffer (the ONLY place s lives) ── sample episode-batch (BPTT, padded; filled-mask
   = per-TIMESTEP episode length, active-mask = per-SLOT occupancy, fixed for the whole episode —
   NOT time-varying: a whole episode is uniformly N=1 or N=2)
                                       ▼
Learner (Cop / Thief / IQL)
   per-agent  Q_i(z_t, o_i)  [RecurrentQNet, GRU eq 8]      ──┐
   QMIX mixer f_mix(Q_1..Q_n ; s),  ∂Q_tot/∂Q_i ≥ 0 (eq 7) ◀─┘ consumes GLOBAL s
   GLOBAL s is encoded to a STAGE-INVARIANT 77-float footprint (out-of-board cells masked),
   mirroring the obs pad-to-5×5 — so the mixer/critic input width is fixed across 2×2..5×5 stages
   target  y_tot = r_team + γ(1-d)·Q_tot⁻(s', greedy(o'))      (Double-DQN action selection)
   loss    masked Huber(Q_tot(s,a) − stopgrad(y_tot))
                                       ▼
export_weights() ── writes cop_weights.pth / thief_weights.pth = AGENT-NET state_dict ONLY + sidecar
                    (mixer, opponent heads, buffer, global-state code NEVER exported)
                                       ▼
results/runs/*.jsonl  ── per-episode {method, seed, grid, reward_cop, reward_thief, td_loss, captured, steps}
```

### 5.2 Exec-time (decentralized — local obs only, no mixer, no `s`)

```
Referee (sole holder of ground-truth s) ── O(s,i) = bounded-Manhattan crop ──▶ Observation (local)
                                       │  request_move(Observation)  [HTTP, Bearer JWT]
                                       ▼
MCP server (Cop / Thief) ── AgentController loads actor-only .pt, .eval()
   a_i = argmax_a  mask_i(a) · Q_i(o_i, a ; θ_i)        ◀── LOCAL obs + legal_actions ONLY
   returns MoveResponse{action, policy_version}          (NO q_values / logits / hidden / s)
                                       ▼
Referee.apply(a_cop, a_thief)  ── simultaneous resolution, swap=capture, barrier=move (ADR-0004)
   (separately, evidence-only) cop_server → thief.reveal_location over real HTTP, and vice-versa,
   radius-gated, LOGGED for §5.3/§7.3(d) — but NEVER fed back into request_move (partial obs preserved)
                                       ▼
Spectator read path ── SpectatorFrame (full ground truth) ──▶ Pygame GUI  (god-view, never leaks into any o_i)
```

**Invariant tests:** `select_action` / `request_move` accept only the local
`Observation` schema (ONE tool contract — localhost == cloud, no `propose_action`/`commit_turn`
fork); an import-boundary test forbids runtime modules from importing
`mixer`/`episode_buffer`/`learner_base`/`GlobalState`/`services`; an MCP-schema test asserts no
global field appears in any request/response; `query_opponent` has its OWN request/response schema in
`mcp/schemas.py` (returns only the last radius-gated `LocationReveal`, keyed by `session_id`) and its
result is decoupled from the next `request_move`. The Thief is folded into `T(s'|s,ā)` and is a *separate* adversarial learner — it
never shares a cooperative mixer with the Cop (POSG, eq 3; ADR-0006).

---

## 6. Architecture Decision Records (index)

This index IS the canonical ADR record — each entry below is **title — decision — rationale**
(ADR-0001..0014 + the supporting-policy ADRs noted after the table). All ADRs that alter a
human-decided column (`CLAUDE.md` §1.4: rules, architecture, test acceptance) require explicit
human sign-off before the corresponding code lands.

| ADR | Title | Decision | Rationale |
|---|---|---|---|
| **0001** | Repo layout | Flat `src/<layer>/`; reject nested `src/marl_cops_robbers/` | Matches A1–A5 the same grader accepted 4×; rename buys nothing; single-entry enforced by import-regex tests. |
| **0002** | Single SDK entry | `MarlSDK` is the only business-logic surface; GUI/MCP/report/scripts import only `src.sdk` | V3 single-entry for UIs; scripts are thin wrappers (exempt). Enforced by `test_sdk_single_entry`. |
| **0003** | CTDE method = QMIX primary | QMIX (monotonic mixer, eq 7) primary; VDN (eq 6) ablation; IQL (eq 4) required baseline; **MAPPO optional prose-only** | BRIEF §1.1/§2.1/§4/§7.2 make CTDE + value-decomposition + IGM/monotonicity the graded outcome; MAPPO lacks a brief citation. Discrete 4–5 action, ≤25-move, tiny grid favors value-based. (Human sign-off before P4.) |
| **0004** | Move resolution & capture | SIMULTANEOUS joint-move; a cop↔thief **swap counts as capture**; `cop_first/thief_first` selectable via `move_resolution` flag. **Extended:** `transition.resolve_joint_action` returns a frozen `TransitionResult(next_state, winner, capture)`; capture is checked **BEFORE** timeout so **a capture on the final move beats the `(step+1)≥max_moves` timeout** (`winner="cop"`, not `"thief"`); **any** cop reaching the thief captures; over-budget barrier placement degrades to a stay **deterministically by cop index** (codex F1). | Brief is silent on move order; simultaneous is the principled default; swap-as-capture prevents a degenerate thief exploit. Capture-beats-timeout keeps a last-tick tag a cop win (the pursuer is rewarded for closing the case on move 25, not penalized by the clock). **Rule decision → human sign-off** before `transition.py`. |
| **0005** | Reward = potential-based shaping (train-only) | `R = −step + distance_weight·(γΦ(s')−Φ(s))`, **team potential `Φ = −min_i manhattan(cop_i, thief) / d_max`** (the closest cop sets it; Ng 1999, policy-invariant) with **`d_max = (H−1)+(W−1)` derived per live grid from `state.h, state.w` (architect decision #2, NOT a config key)**; terminal ±1 dominant, `Φ(terminal)=0`; shaping OFF at eval; official 20/10/5/5 scoreboard SEPARATE (`Scorer`, report-only) | Sparse terminal-only signal stalls on a 25-step horizon; potential-based shaping is provably policy-invariant, preserving faithfulness while curves converge. `−min` rewards the team for ANY cop closing in; deriving `d_max` keeps one `Φ` correct across the 2×2→5×5 ladder. Keeps graded scores authentic. **Test-acceptance change → human sign-off.** |
| **0006** | Adversarial boundary (POSG) | Value decomposition applies ONLY within the cooperative Cop team; Thief is a separate adversarial Double-DQN folded into `T(s'|s,ā)`; NO mixer over {cop,thief} | Opposed rewards violate `∂Q_tot/∂Q_thief≥0`; full game is POSG (NEXP^NP, eq 3). Both planning legs independently confirmed; load-bearing. |
| **0007** | Cop-team size & mixer non-triviality | Graded 5×5 match = literal 1-cop-vs-1-thief (degenerates gracefully); `num_cops≥2` runs ONLY on the 4×4 train scenario, which supplies a genuinely 2-input (non-trivial) mixer for non-vacuous decomposition | Faithful to BRIEF §3 while giving §7.2 real credit-assignment evidence; N=1 'trivial decomposition' is itself the rubric-required IGM-limit discussion. **Scope decision → human sign-off.** |
| **0008** | Shared net + swappable mixer/learner seam | One `RecurrentQNet` (GRU) + one centralized episode buffer; VDN/QMIX vary ONLY the Mixer ABC, while IQL additionally **drops the mixer + global state** and overrides `_compute_target` (independent per-agent target, eq 4) — the variation point is mixer-OR-learner-branch, NOT mixer-only | DRY + single-entry; keeps the algorithm comparison apples-to-apples (IQL = a ~15-line `_compute_target` diff the §7.2 write-up quotes; there is no `iql_mixer.py` identity module). |
| **0009** | §5 External-API governance REQUIRED | Add `src/api/gatekeeper.py` — one `ApiGatekeeper` with `execute(call)` + `get_queue_status()`, per-channel token-bucket (peer_mcp / gmail / prefect_deploy) from versioned `config/rate_limits.json`, a FIFO overflow queue (no crash on burst), call logging; ALL peer-MCP + Gmail + Prefect egress routes through it | §5 was N/A in A1–A5 (no runtime external calls); A6 makes real HTTP + Gmail + cloud-deploy calls → §5 flips N/A→REQUIRED. Enforced by `test_egress_via_gatekeeper`. |
| **0010** | OLoRA scope & init | Pre-trained model = locally BC-cloned per-role encoder; OLoRA = paper-exact QR on the **pretrained** weight `W0` (eq 10-11, function-preserving rebase), wraps **encoder Linears only**, `rank=4` (assert ≤min(m,n)//2), `scale=8`; default OFF for from-scratch 5×5, ON for curriculum transfer | Only honest reading of §5.2 "pre-trained models + OLoRA (QR on pretrained weight matrices)" satisfying "training MUST be local"; reject QR-of-random-matrix (that is orthonormal-LoRA, not OLoRA) — the rejection is §7.2 grade evidence. |
| **0011** | Referee-mediated dual-MCP topology, process isolation, per-session GRU state | 3 OS processes: 2 peer FastMCP servers + 1 neutral referee = THE ENVIRONMENT (CTDE global-state holder), NOT a third player; real cop↔thief HTTP `reveal_location` each tick (evidence-only); each agent server keeps **server-side per-sub-game GRU hidden state** `z_t` (keyed by session id, reset on `new_sub_game`) — agent servers are NOT `FASTMCP_STATELESS_HTTP` | Both planning legs converged; removes cop-as-self-referee conflict the §9 `mutual_agreement` exposes; makes train-global/exec-local literal in code. Processes (not threads) avoid FastMCP-async/Pygame/torch deadlock. The recurrent policy (eq 8) MUST carry `z_t` across the ~25 `request_move` ticks, so stateless HTTP is rejected for the agent servers (session-state holds it server-side; same contract localhost==cloud). Server-side `z_t` therefore requires a **single worker** (`cloud.workers: 1`); the free single-instance tier already provides this, and **client-carried hidden state** is the documented fallback if the host ever scales beyond one worker. |
| **0012** | Cloud platform & auth | Primary host = **Prefect Horizon / FastMCP Cloud** (push-to-deploy, `*.fastmcp.app/mcp`); Render free-tier fallback (pre-tested); auth = app-level **revocable Bearer JWT** (RS256) inside FastMCP, NOT Horizon OAuth; revoke via jti deny-list (primary) + key rotation (hard lever) | §8.3 names this infra; identical auth gate runs on localhost AND cloud (never rewritten); §5.3 demands a revocable token + screenshottable 401. Inference-only deploy (actor nets; mixer/replay/`s` stay local). |
| **0013** | Gmail mechanism & PII boundary | smtplib + STARTTLS + Gmail **App-Password** (default) behind an `EmailSender` Protocol (OAuth `GmailApiSender` drop-in fallback); real names/ids ONLY in git-ignored `secrets/players.local.yaml`, injected at send time; only role-only redacted JSON committed | Autonomous no-human-in-the-loop Cop (§3.5) can't do OAuth's interactive consent; App-Password is one revocable `.env` secret vs a tracked `token.json` hazard; §5.5 explicitly blesses it. PII deny-list is a hard gate. |
| **0014** | GUI = Pygame god-view spectator | Pygame (over Tkinter/Streamlit); strict read-only god-view fed by referee ground truth via frozen `SpectatorFrame`; dual read path (in-proc Stage-1 default, HTTP `/spectator` SSE Stage-2, separate spectator token); render literals local in `palette.py` | True real-time loop + clean headless `SDL_VIDEODRIVER=dummy` capture for §7.3c; reuses A5's proven ≤150-LOC module split; spectator never affects partial observability (hard test). |

Supporting policy ADRs documented in the same index: version `1.0.0` (3-segment mapping of V3
"1.00", A5-accepted; `test_version_consistency`), §10 Nielsen-heuristics UX doc REQUIRED (GUI now
mandatory), equation-numbering policy (ex06/BRIEF numbering primary, footnote L10 equivalence —
e.g. ex06 eq 2 ≡ L10 eq 4; citation hazard: BRIEF [2]=VDN vs L10 [7]=VDN).

---

## 7. Deployment diagram (localhost → cloud)

Strict §6 ordering: **localhost (Stage 1, §6.6) must work before cloud (Stage 2, §6.8).** The
auth gate and the tool contracts are *identical* across both stages — only the verifier (Static→JWT)
and the URLs (env-switched) change, so auth is never rewritten.

### 7.1 Stage 1 — localhost (the gradable baseline)

```
┌──────────────────────────── developer machine (uv run) ────────────────────────────┐
│                                                                                      │
│   run_match.py ──spawn──▶ cop_server (FastMCP http://127.0.0.1:8001)                  │
│        │ (referee)  ──spawn──▶ thief_server (FastMCP http://127.0.0.1:8002)            │
│        │                                                                              │
│        │  StaticTokenVerifier (tokens from .env; scopes per tool)                     │
│        │  referee → cop.request_move / thief.request_move  (Bearer, loopback HTTP)    │
│        │  cop ⇄ thief  reveal_location  (real server-to-server HTTP, §7.3d evidence)  │
│        │  poll /healthz before tick loop; technical-loss replay → exactly 6 sub-games │
│        ▼                                                                              │
│   Pygame GUI ◀── SpectatorFrame (in-process via SDK.spectator_session())             │
│   cop.send_final_report ──▶ gatekeeper ──▶ smtplib:587 ──▶ rmisegal+marl@gmail.com    │
└──────────────────────────────────────────────────────────────────────────────────────┘
artifacts: results/screenshots/{local_comms.png, cloud_auth_401.png}, full_match.json
```

### 7.2 Stage 2 — cloud (incremental upside; localhost logs remain canonical F4)

```
GitHub main ──push-to-deploy──▶  Prefect Horizon / FastMCP Cloud   (single worker — cloud.workers: 1)
   (commit ONE actor-only ──┐     ├─ entrypoint src/mcp/cop_server.py:mcp   → https://adrl-001-cop.fastmcp.app/mcp
    inference checkpoint:    │     └─ entrypoint src/mcp/thief_server.py:mcp → https://adrl-001-thief.fastmcp.app/mcp
    GRU+MLP state_dict +     │     env: MCP_PUBLIC_KEY, issuer, audience, REVOKED_TOKEN_JTIS,
    shape sidecar, sub-MB,   │          MODEL_PATH = deploy/model/cop_actor.pt (Stage-2; actor weights ONLY —
    under deploy/model/)  ───┘          NO mixer / NO critic / NO replay / NO global state)
                                  NOT stateless: each server keeps per-sub-game server-side GRU
                                       hidden state z_t (keyed by session id), reset on new_sub_game.
                                  Single worker REQUIRED: server-side z_t needs one process (free
                                       single-instance tier already gives this); client-carried hidden
                                       state is the fallback if the host ever scales >1 worker.

   referee / smoke_cloud.py (local or CI)
        │  JWTVerifier (RS256, exp + jti deny-list); BearerAuth clients; warm-up ping
        │  (jti deny-list = a CUSTOM verifier wrapper around FastMCP's JWTVerifier, which itself
        │   validates only sig/exp/aud/scopes — the per-jti revocation check is ours, not built in)
        │  cop(cloud) ⇄ thief(cloud)  request_move / reveal_location  (SAME tool contract as Stage 1)
        │     → SAME trace_id appears in BOTH servers' Horizon logs (§7.3d cloud proof)
        ▼
   cloud_match.py ── 6 sub-games over the two public URLs → emits §3.5 JSON body (does NOT send)

   Revoke proof: mcp_tokens.py revoke --jti <jti> → redeploy → old token 401 / new token 200
   Fallback: deploy/render.yaml (two web services, pre-tested warm-up ping)
   Teardown after grading proof captured (save free-tier compute minutes).
```

**CTDE deploy invariant:** cloud servers run **inference only**. `request_move` rejects any
`global_state` field; deployed model files contain actor weights only — no trainer, mixer, replay,
or global-state code reaches the cloud (`test_egress_via_gatekeeper` + actor-only checkpoint test).
**Model delivery mechanism:** ONE small actor-only inference checkpoint (the GRU+MLP `state_dict` +
its shape sidecar, sub-MB) **will be committed at deploy time** under the non-ignored `deploy/model/`
path (a `.gitignore` allow-list exception reserves it); `MODEL_PATH` points the deployed entrypoints at it. The trainer's
`export_weights()` writes exactly this actor-net `state_dict` (mixer/critic/replay/`s` excluded).
Per-sub-game GRU hidden state is held **server-side** (keyed by session id, reset on `new_sub_game`)
so the recurrent policy (eq 8) carries `z_t` across the ~25 `request_move` ticks — the agent servers
are deliberately **NOT** `FASTMCP_STATELESS_HTTP` (ADR-0011 session-state; same contract on localhost).

---

## 8. Phased build sequence (strict BRIEF §6 order, exit-gated)

Milestones P0–P11 follow §6 verbatim: theory → pursuit → pipeline → training → MCP → local →
GUI → cloud → Gmail. Each phase has an **exit gate** that must pass before the next begins.
`docs/TODO.md` tags every task with its phase + §-anchor. TDD throughout (RED→GREEN→REFACTOR).

| Phase | §6 step | Deliverable | Exit gate |
|---|---|---|---|
| **P0** | pre | Scaffold repo tree (§3); `pyproject` v1.0.0 + uv groups; `config.yaml` + `config.schema.yaml` + `rate_limits.json` + `.env-example`; `config_loader`; gate scripts + `tests/architecture/` (RED); CI (ruff → format → file-size → check_* → pytest cov); PRD/PLAN/TODO + ADRs 0001–0014 with **human §1.4 sign-off**; add `rmisegal@gmail.com` read collaborator | docs reviewed + signed; `uv sync` green; CI skeleton green; arch tests present |
| **P1** | Theory first | Formal Dec-POMDP tuple ⟨N,S,A,T,R,Ω,O,γ⟩ (eq 1) + POSG caveat (eq 3) in `docs/THEORY.md`/README §7.1; S/A/R/Ω/O/γ as config constants; `actions.py`/`types.py`/`grid.py`; `reward.py` (potential-shaping, asserts `F=γΦ(s')−Φ(s)`) + `scorer.py` separate; README §7 outline names every figure/experiment | tuple doc present; reward + scorer tests pass; ADR-0004/0005 signed off |
| **P2** | Basic pursuit | `transition.py` (PURE: simultaneous, swap=capture, barrier-is-move, budget cap, edge=wall, timeout); `observation.py` (5×5-padded egocentric image + 6 scalars, view-radius auto formula, other_visible masked beyond radius); `cops_robbers_env.py` (reset/step 4-tuple/state/render_state); **leak test**: `step()` never returns `GlobalState` | game-logic tests ≥85% cover; 2×2 capture works; obs shape fixed at 2/3/4/5; leak test green |
| **P3** | Minimal pipeline | `curriculum.py` (Table-2 stages + promotion gate); `episode_buffer.py` (padded episodes + global s, post-done masking, four §5.2 source tags); `data/{schemas,obs_encoder,heuristics,bc_dataset}.py` (DTOs + stage-invariant `encode_state` + Manhattan oracles + npz IO); **throwaway tabular 2×2 smoke** (`replay/tabular_smoke.py`, deleted in P4) collect→train→export via `MarlSDK.run_p3_smoke`; headless `render_state()` god-view JSON + minimal sub-game JSON (`docs/schema/subgame.schema.json`). **GUI shot → P7; BC training → P4/T4.4.** | 2×2 reaches **pursuit-optimal capture from every 2×2 start over all `training.seeds`** (thief flees ⇒ naive min-Manhattan wrong; a zeroed Q fails the gate); pipeline runs NaN-free; buffer mask + BPTT shape tests pass; sub-game JSON validates vs schema |
| **P4** | Network + training | `recurrent_q_net.py` (GRU, eq 8); mixers `base/vdn/qmix` (IGM + monotonicity tests; IQL has no mixer — it branches in `iql_learner`); `learner_base` + cop/thief/iql learners; `olora_*` (QR init eq 10, forward eq 11; orthonormal + function-preserving tests); BC pretrain → OLoRA-attach → CTDE-finetune; `selfplay`/`trainer` curriculum; seeded sweep harness logging to `results/runs/*.jsonl` | loss decreasing; 3×3 converges; QMIX/VDN/IQL arms run on identical nets/seeds; sweep logs written |
| **P5** | MCP protocol | `mcp/schemas.py`/`actions.py`/`auth.py`(Static)/`agent_runtime.py`; `cop_server.py`/`thief_server.py` (tool surface, NO get_state/get_policy); `clients.py`; `services/referee.py` (THE ENVIRONMENT, OUTSIDE `src/mcp/`: observe=O(s,i), apply, simultaneous rule, scoring, technical-loss replay); `api/gatekeeper.py`+`http_client.py` | localhost: cop queries thief via MCP over HTTP; 401 on bad/revoked token; F4 local logs captured; no-global-leak + egress arch tests pass |
| **P6** | Full local run | `run_match.py` (spawn both servers, poll /healthz, run referee 6 sub-games, assemble `MatchReport`); 6 valid sub-games; tally; §3.5 JSON assembled (NOT sent) | 6 sub-games complete; totals match Scorer; JSON validates vs `report.schema.json`; integration test green |
| **P7** | GUI | `gui/*` (transform/palette/grid_view/hud/score_view/input_map/state_client/app); `play.py`; `capture_screens.py` (headless deterministic fixtures); architecture gates (spectator purity, imports-only-SDK, no-leak) | F3 screenshots at 2×2/3×3/4×4/5×5 with cop+thief+≥1 barrier; spectator-purity + no-leak tests pass |
| **P8** | Cloud-MCP | §8 4-stage runbook: (1) local `uv export` deploy deps (NO requirements.txt — uv-only); (2) `@mcp.tool` entrypoints load the committed actor-only checkpoint from `MODEL_PATH` (deploy/model/, OLoRA actor nets) + per-session GRU state (single worker, NOT stateless); (3) Prefect token + push-to-deploy + capture public URLs; (4) confirm Gmail auto-report path. Swap to `JWTVerifier`+jti deny-list; pin fastmcp; deploy both entrypoints to Horizon with the SAME tool contract as Stage 1 (`request_move`/`reveal_location`/`query_opponent`/`new_sub_game`); wire peers; `smoke_cloud.py` (shared trace_id), `smoke_auth.py` (401 + revoke); `cloud_match.py`; `render.yaml` fallback pre-tested; `runbook.md`; redact before every screenshot | distributed match runs (≥1 valid cloud sub-game); cloud comms + 401 + revoke screenshots captured (cloud = upside, localhost canonical per ADR / R2) |
| **P9** | Gmail report | `reporting/*` (schema/collector/players/builder/mailer/redact/idempotency); `docs/schema/report.schema.json`; `sdk.send_final_report` (idempotent, sha256 sentinel, PII from secrets); wire cop's record→send after 6th valid sub-game; `smtp_smoke.py` | one email received at `rmisegal+marl@gmail.com`; JSON validates vs §3.5; idempotency + no-PII-in-tracked tests pass |
| **P10** | Results | `results/aggregate.py`+`plots.py`+`make_figures.py`; run sweep (IQL/VDN/QMIX × 5 seeds × ladder); generate F1–F6; **§9 single-parameter sensitivity sweep** (e.g. `view_radius` / `olora.rank` / `selfplay.window_k` / `reward.distance_weight` → capture-rate effect + figure — DISTINCT from the algorithm ablation); **`notebooks/analysis.ipynb`** (SDK-only: LaTeX Dec-POMDP/QMIX equations, F1–F6, citations); `experiment_manifest.json`; write README §7 academic body (7.1 formalisms + eq map; 7.2 four sub-parts incl. IQL-vs-CTDE F5, IGM limits + QPLEX[10]/WQMIX[9], curriculum[5]/MAPPO; 7.3 captioned figures); risk register in PLAN + README §8 | all 6 figures present + non-placeholder; sensitivity figure present; notebook runs via SDK; numbers reproducible from `runs/`; README §7 complete |
| **P11** | Gate | `software-excellence-v3` audit; confirm all hard gates; **`docs/COST_ANALYSIS.md`** (§11: AI-dev token-cost breakdown — input/output tokens, $/1M per model, total — plus the local-only RL training-compute envelope: episodes × stages × wall-clock + optimization notes); **`docs/QUALITY.md`** (§13: ISO/IEC 25010 mapping of ALL EIGHT characteristics — Functional Suitability, Performance Efficiency, Compatibility, Usability, Reliability, Security, Maintainability, Portability); self-grade; produce git-ignored cover sheet `adrl-001-ex06.pdf` (Moodle only, never asserted in a test); freeze 24h pre-deadline | all V3 hard gates ✅; COST_ANALYSIS + QUALITY present; audit doc + self-grade written; code frozen |
| **P-bonus** | §9 post-A6 | `docs/interfaces/intergroup_mcp.md` interface spec (built now); bonus-capable report schema branch (`report_type:bonus_game`, `groups`, `students_group_1/2`, per-sub-game `cop_group`/`thief_group`, `totals_by_group`, `bonus_claim`, `mutual_agreement`) — **off A6 critical path; live match needs a partner `biu-rlNN` group** | spec + schema present; if a partner is found: both groups email + `mutual_agreement:true` |

**Load-bearing ordering:** P10 needs P4 (training) + P6 (orchestration); F4 needs P5; F3 needs P7;
email JSON (P9) needs P6 tally; the §11.3 scale figure (F6) needs the full-ladder P4 sweep; cloud
(P8) is decoupled from and precedes Gmail (P9) — `cloud_match.py` emits the JSON body but never sends.

---

## 9. V3 hard-gate → A6 mapping

| Gate | How A6 satisfies it | Enforcement |
|---|---|---|
| ≤150 LOC/file (excl blanks/comments) | every module budgeted in §3; cop/thief servers split from `tools`, mixer from q-net, env across 9 files | `scripts/check_file_sizes.py` **after** `ruff format`; `test_final_gates` |
| ≥85% coverage | DI + mocked httpx peer & Gmail; pure `transition`/`gatekeeper`/`builder` unit-tested; FakeEmailSender | `pytest --cov=src --cov-fail-under=85` |
| Ruff 0 violations | `ruff check` + `ruff format --check`; PLR2004 ignored in tests only | CI steps 4–5 |
| Docstring gate (every module/class/public fn) | Ruff `D` rules in the lint select (`D` per-file-ignored for `tests/`); every package `__init__.py` carries a one-line module docstring | `ruff check` (Ruff `D`); CI step 4 |
| 0 hardcoded values | grid/moves/games/barriers/scoring/ports/hyperparams/recipient/templates in `config.yaml`; render literals local | `scripts/check_no_hardcode.py`; `test_config_single_source` |
| 0 secrets + `.env-example` | tokens/keys/App-Password/PII in `.env` + `secrets/`; `.env-example` names-only; `.gitignore` covers `.env *.pem *.key credentials*.json secrets/` | `scripts/check_secrets.py`; redaction middleware + `redact_logs.py` before screenshots |
| uv-only | CI uses `uv`; no pip/conda; `uv.lock` committed | CI step 3 `uv sync --frozen` |
| Single SDK entry (UIs) | GUI/MCP/report import only `src.sdk`; scripts are thin wrappers (exempt) | `test_sdk_single_entry`, `test_mcp_servers_have_no_logic` |
| Version starts at 1.00 | `1.0.0` in `src/__init__.py` + `config.version` | `scripts/check_version.py`; `test_version_consistency` |
| PRD/PLAN/TODO + ADRs | this PLAN + PRD + TODO + ADRs 0001–0014 with §1.4 sign-off | `test_required_docs_present` + ADR-count check |
| No PII in tracked content | real names/ids only in git-ignored `secrets/players.local.yaml`, injected at send time; cover sheet git-ignored, **skip-when-absent** (never asserted present — MEMORY: this broke A5 CI) | `scripts/check_pii.py`; `test_no_pii_in_tracked_content`; `test_cover_sheet_gitignored` (skip-when-absent) |
| §5 External-API governance (flips REQUIRED) | `ApiGatekeeper` + `rate_limits.json`; all egress routed through it | ADR-0009; `test_egress_via_gatekeeper` |
| §10 Nielsen heuristics (flips REQUIRED) | `docs/UX.md` maps all 10 heuristics to the mandatory GUI + screenshot per state | ADR-0014; `test_required_docs_present` |

**Carried A1–A5 lessons (CI hygiene):** never pin mutable git SHAs in tests; file-size gate runs
**after** `ruff format`; CI never sends Gmail or hits cloud MCP (`report.dry_run` + mocked clients);
CI never asserts the git-ignored cover sheet *exists* (skip-when-absent); push often and keep a
`/tmp` clone — Google Drive can silently delete `.git` mid-session.

---

## 10. Risk register (P×I; mitigation + fallback; owner-phase)

| ID | Risk | P×I | Mitigation | Fallback |
|---|---|---|---|---|
| R1 | MARL non-convergence/instability on tiny grids (the studied phenomenon) | H×H | start 2×2 (P3); 5 seeds; potential-based shaping; OLoRA QR-init; VDN fallback arm | heuristic Thief for early Cop training; report instability honestly with curves |
| R2 | Cloud-MCP deploy fails / Prefect quota late | M×H | §5.3 localhost-first is sufficient for grading comms; localhost logs = CANONICAL F4; warm-up ping + Render fallback pre-tested | submit local proof + cloud smoke proof; cloud is upside not gate |
| R3 | Gmail single mandatory send fails (App-Password/OAuth) | M×H | `smtp_smoke.py` pre-flight days early; `EmailSender` Protocol → OAuth drop-in; idempotent sentinel | never block match completion on send; manual resend at most one extra email |
| R4 | QMIX/POSG formalism mismatch | H×M | document Dec-POMDP proxy (shared R) vs full POSG (role-specific R) in §7.1; claim CTDE improves stability not optimality | present CTDE as assignment-aligned approximation + POSG limitation |
| R5 | PII leak into tracked repo | M×H | cover sheet git-ignored Moodle-only; CI deny-list grep; placeholders; identity from `secrets/` | NEVER assert git-ignored cover sheet exists in a test (skip-when-absent — broke A5 CI) |
| R6 | Google Drive deletes `.git` mid-session | M×H | push often; keep `/tmp` clone to restore `.git` | re-clone, copy `.git` back |
| R7 | File >150 LOC as env/mixer/MCP grow | H×M | split early (9-file env, 4-file mixers, server/tools split); file-size gate AFTER ruff format | extract submodules |
| R8 | Figures drift from code / stale README numbers | M×M | single `make_figures` entry reading `results/runs/*.jsonl`; manifest with commit hash | CI regenerates figures |
| R9 | §7.2 IGM lossy-decomposition critique too shallow | M×M | use L10 pincer example + Table 3; cite QPLEX[10]/WQMIX[9] with exact arXiv IDs; N=1 caveat | — |
| R10 | Bonus inter-group dispute (no mutual agreement → both 0) | L×M | written pre-game agreement + shared signed JSON before send | keep entirely off A6 critical path |
| R11 | Live-opponent MCP flakes CI (non-stationarity in test) | M×M | mock opponent MCP in unit tests; seeded-deterministic opponent in integration | — |
| R12 | OLoRA ambiguity (HARD req vs marginal at 5×5) | M×M | implement faithfully behind config; ablation = §7.2 evidence; cite [7] | disable OLoRA, document plain GRU baseline |
| R13 | Citation-numbering hazard (ex06 vs L10) | M×M | README uses ex06/BRIEF numbering primary, footnote L10 equivalence | — |
| R14 | FastMCP API churn (auth import path moved v2→v3) | M×M | run import check + pin fastmcp version BEFORE writing `auth.py` (P0/P8) | pin closest-to-brief v2.11.x |
| R15 | Late penalty −5/24h | L×H | P11 V3 audit with buffer; freeze code 24h before deadline | — |
| R16 | Solo-build overload (single submitter, large scope) | M×M | sequence as solo work-streams — WS1 env+CTDE+plots, WS2 MCP+GUI+Gmail+integration — one at a time, with a daily artifact checklist | freeze scope to env+IQL+QMIX+report figures (defer-order in §1) |

---

## 11. Bibliography cross-reference (BRIEF §10)

[1] Bernstein+ 2002 Dec-POMDP complexity (eq 1 tuple, eq 3 POSG NEXP^NP) · [2] Sunehag+ 2018 **VDN**
(1706.05296, eq 6) · [3] Rashid+ 2018 **QMIX** (1803.11485, eq 7 monotonicity) · [4] Amato 2024
Cooperative MARL intro (2405.06161, IQL limits) · [5] Lin+ 2025 cooperative pursuit-evasion +
curriculum (MDPI Electronics, §7.2(4)) · [7] Büyükakyüz 2024 **OLoRA** (2406.01775, eq 10-11 QR PEFT) ·
[8] Foerster+ 2018 **COMA** (1705.08926, IQL/centralized-critic contrast) · [9] Rashid+ 2020
**Weighted QMIX** (2006.10800, IGM-limit fix) · [10] Wang+ 2021 **QPLEX** (2008.01062, duplex dueling) ·
[11] Anthropic 2024 **MCP spec** (modelcontextprotocol.io, §5.3/§8).
*Citation hazard:* README uses ex06/BRIEF numbering as primary and footnotes L10 equivalences
(e.g. ex06 §7.2 "eq 2" ≡ L10 "eq 4" IQL target; BRIEF [2]=VDN vs L10 [7]=VDN).
