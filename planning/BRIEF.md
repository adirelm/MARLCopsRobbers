# Assignment 6 — MARL Cops & Robbers (Vibe Coding, Cloud-MCP) — Distilled Brief

> Source of truth: `planning/source_pdfs/ex06.pdf` (23 pp, Dr. Yoram Segal, 2026) +
> `L10-MARL.pdf` (Lecture 10). This file is the distilled spec all planning agents
> share. Group code: **adrl-001**. Cover sheet template: `adrl-001-ex06.pdf` (Moodle only).

## 0. One-paragraph summary
Build a **Multi-Agent RL (MARL)** "Cops and Robbers" pursuit game between **two fully
autonomous AI agents** (one **Cop**, one **Thief**) on a dynamic grid, under the
**Vibe Coding** methodology. Each agent has only **partial/local observation** (bounded
Manhattan-radius view; no global state at execution). Model it as a **Dec-POMDP**
(cooperative framing) / **POSG** (general-sum). Train with **CTDE** (Centralized Training,
Decentralized Execution) + **value decomposition** (VDN/QMIX, IGM principle). The two
agents are **separate processes each exposing its own MCP server**; they communicate over
**HTTP** (localhost first, then **cloud**). Mandatory **GUI**. At game end, the Cop sends
**one automatic Gmail report** (JSON of all 6 sub-games) to `rmisegal+marl@gmail.com`.

## 1. Learning outcomes (§1.1)
(1) model partial-observability decision problems as Dec-POMDP; (2) design+implement
CTDE + value decomposition between distributed agents; (3) practical MCP + Vibe-Coding;
(4) critically analyze the algorithms' limits and their real-world implications.

## 2. Scenario (§2) — cooperative-competitive framing
Autonomous agents act in shared space, reasoning in real time about other agents'
actions (allies or rivals). Cop team tries to catch a Thief on a shared urban grid; each
agent has **limited view radius only** (bounded Manhattan distance), **no global state**.
- §2.1: Pursuit modeled as **Dec-POMDP** (cooperative); reward `R(s, ā)` is **global,
  shared** for the pursuer team (e.g. capture success / time-to-capture). Thief is part
  of transition dynamics `T(s'|s,ā)` → environment becomes **non-stationary** from any
  single learner's view. CTDE + value decomposition (VDN[2], QMIX[3]) on the **IGM
  (Individual-Global-Max)** principle directly address this. Fuller model = **POSG**
  (general-sum, separate reward per agent). Students must address modeling implications
  (incl. limits) in the §7 academic analysis.

## 3. Game rules, scoring, reporting (§3)
Implement the inter-agent comms (MCP) + async comms via Gmail API, **no human in the loop**.
- **Game** = full match of **6 sub-games**. **Sub-game** = one Cop-vs-Thief pursuit,
  capped at a number of moves. The pair of AI agents plays all 6.
- **Board:** 5×5 grid. **Sub-game length:** ≤ **25 moves**. **Roles:** Cop / Thief.
- **# sub-games:** 6. Each move: both Cop and Thief move one cell.
- **Win (§3.2):** Cop wins if at some move it lands on the Thief's cell (capture). Thief
  wins if not caught within 25 moves (never shared a cell with the Cop for the whole game).
- **Barriers (§3.3, optional advanced):** Cop may, instead of moving, place a **Barrier**
  on its current cell (the placement is the move). Barrier = impassable (like a wall /
  board edge for step-computation). Limit **≤ 5 barriers per sub-game** (config max_barriers).
- **Scoring (§3.4, Table 1):** Cop wins=**20**, Thief wins=**10**, Cop loses=**5**, Thief loses=**5**.
- **Email report (§3.5):** sent **once**, by the **Cop**, at end of the **whole game**
  (after all 6 sub-games), to `rmisegal+marl@gmail.com`, with a structured **Subject** and a
  **JSON body** of all 6 sub-games. JSON keys: `group_name`; `students[]` (role A/B/C,
  full_name, id); `github_repo`; `timezone` ("Asia/Jerusalem"); `sub_games[]`
  (id, start, end ISO-8601, moves, winner, scores{cop,thief}); `totals{cop,thief}`.
- **Config file (§3.6):** all numeric params in `config.yaml`/`config.json`, **no hardcoding**.
  Keys: `grid_size`(5×5), `max_moves`(25), `num_games`(6), `max_barriers`(5),
  `scoring`{cop_win:20, thief_win:10, cop_loss:5, thief_loss:5}.
- **Technical losses (§3.7):** sub-games that fail technically don't count; redo to reach 6.

## 4. Link to Lecture 10 (§4)
Builds directly on L10 MARL theory; extends A5 (DDPG, single-agent, stationary) to the
**non-stationarity** of multiple simultaneously-learning agents. Implement L10 §2.2
(cooperative MDP), §3 (non-stationarity instability + value decomposition), §4 (CTDE).
Report MUST faithfully reflect train-time (**global state access**) vs execution
(**local observation only**) distinction.

## 5. Task definition + implementation requirements (§5)
- **Group = TWO students** (roles A, B; optional C). No league/inter-group collab; each
  group ships its own pipeline.
- **Formal Dec-POMDP** tuple **⟨N, S, A, T, R, Ω, O, γ⟩** (eq 1), citing primary source [1].
- **§5.1 Env + dynamic grid:** custom Cops&Robbers env; generic architecture supporting
  **dynamic grid sizes** + board boundaries; **graduated Sanity-Check validation** (Table 2):
  - **2×2** very-low: algorithm correctness + full-Pipeline validation.
  - **3×3 / 3×2** low: entry test + first hyperparameter pass.
  - **4×4 / 4×3** high: partial-observability effects (view radius scales with Manhattan dist).
  - **5×5** final: final test + **Learning Curves**.
- **§5.2 Algorithms / dataset / training:** free choice of MARL algo (VDN/QMIX/MADDPG/
  MAPPO/IQL). **Training MUST be LOCAL.** Build a **Dataset** OR generate synthetic data /
  advanced simulation / earlier policies / Manhattan-heuristics. **Centralized Replay
  Buffer** (CTDE) recommended. Use **pre-trained models + OLoRA** (Orthonormal Low-Rank
  Adaptation [7], QR-decomposition PEFT) for efficient fine-tuning + stable RL in
  non-stationary settings.
- **§5.3 Cloud-MCP comms architecture:** each agent (Cop, Thief) is a **fully autonomous
  AI agent with its OWN separate MCP server**. They talk via **HTTP** to each other's MCP
  server; each reveals position/actions over HTTP only as needed for mutual location checks.
  - **Two separate MCP servers** — each hides its state/policy; other queries via HTTP.
  - **Stage 1 — localhost:** run both MCP servers locally (different ports); confirm whole
    game flows via inter-agent comms.
  - **Stage 2 — cloud:** only after local works, deploy both to cloud; confirm distributed comms.
  - **Public access + auth:** bearer **Authentication Token** (revocable).
  - **Recommended infra:** **FastMCP (Prefect)** — recommendation only, details are yours.
- **§5.4 GUI (mandatory):** Pygame / Tkinter / Streamlit; real-time Cop-vs-Thief dynamics on
  the grid, barrier positions, step progression.
- **§5.5 Gmail API:** at whole-game end, send **one** auto summary email to
  `rmisegal+marl@gmail.com` with the full 6-sub-game JSON (per §3). API choice free
  (Google API Client / App Password+smtplib / dedicated MCP tool).

## 6. Development priority order (§6)
1. **Theory first:** L10 RL principles + algorithm — agent logic, reward fns, state space
   S, action space A; full Dec-POMDP tuple; in code: **local obs only at execution, global
   state only at training**. 2. Full basic pursuit (game logic+rules). 3. Full Pipeline on
   minimal grid. 4. Network quality + training (hyperparams, Dataset). 5. MCP protocol for
   inter-agent comms. 6. Full local system run. 7. GUI. 8. Cloud-MCP deploy. 9. Gmail API report.

## 7. Academic proof + results analysis (§7) — README.md at repo root
- **§7.1 Correct formalisms:** formal Dec-POMDP in the report body; write the **Value
  Function** (or Policy) under the chosen MARL method; explain how the assumptions surface in code.
- **§7.2 Critical analysis:**
  - **Non-stationarity + CTDE fix:** explicit; contrast single-agent independent target
    **eq 2: y_i = r_i + γ·max_{a'_i} Q_i(o'_i, a'_i)** vs the **centralized critic** with
    access to global state + joint actions.
  - **Baseline:** compare **IQL** (Independent Q-Learning) vs CTDE; note IQL limits in
    non-stationary env [4],[8].
  - **IGM limits:** short discussion of **QMIX monotonicity** ("lossy decomposition") +
    fixes (**QPLEX**[10], **Weighted QMIX**[9]).
  - **Optional:** compare to pursuit-evasion MARL works (Curriculum Learning[5], MAPPO).
- **§7.3 Graphs/viz/docs (MANDATORY in report):** (a) **Learning Curves** of both agents
  over episodes (expected reward); (b) **Loss graphs** of deep nets across local training
  stages; (c) clear **GUI screenshots** at different grid sizes; (d) **screenshots proving
  correct MCP-server comms** (e.g. CLI logs).

## 8. Technical appendix (§8) — MCP-on-cloud step-by-step
FastMCP + Prefect Cloud. Stage 1 local env (CLI + `uv` deps: prefect, fastmcp, torch,
numpy). Stage 2 write `mcp_server.py` (`@mcp.tool` defining the tools; load local
OLoRA-tuned nets; each tool remotely callable). Stage 3 deploy to cloud + Prefect Cloud
(horizon.prefect.io, API token, deploy, publish public URL). Stage 4 Gmail API auto-report.

## 9. Bonus +10 pts (§9) — inter-group game (counts to FINAL PROJECT, not A6)
Two DIFFERENT groups play a cross-group match: 6 sub-games, roles alternate (first 3:
group-1 Cop vs group-2 Thief; next 3: swapped). One group sends the bonus email. Bonus
scoring: winning group +10, losing +7; tie → +5 each. Counts only if **both** groups sent
a valid bonus email AND agreed on results (`mutual_agreement: true`), else both 0. README
must document the bonus game (group names, final bonus scores, screenshots). JSON adds
`github_repo_group_1/2`, `students_group_1/2`, per-sub-game `cop_group`/`thief_group`,
`totals_by_group`, `bonus_claim`, `mutual_agreement`.

## 10. Bibliography (§10)
[1] Bernstein+ 2002 Dec-POMDP complexity. [2] Sunehag+ 2018 **VDN** (1706.05296).
[3] Rashid+ 2018 **QMIX** (1803.11485). [4] Amato 2024 Cooperative MARL intro (2405.06161).
[5] Lin+ 2025 cooperative pursuit-evasion + curriculum (MDPI Electronics).
[7] Büyükakyüz 2024 **OLoRA** (2406.01775). [8] Foerster+ 2018 **COMA** (1705.08926).
[9] Rashid+ 2020 **Weighted QMIX** (2006.10800). [10] Wang+ 2021 **QPLEX** (2008.01062).
[11] Anthropic 2024 **MCP spec** (modelcontextprotocol.io).

## Standing submission rules (course-wide)
- Code on GitHub; **share repo with `rmisegal@gmail.com`** (read collaborator).
- Each member submits separately on Moodle; **identical repo URL** for all members.
- Group code: 8 chars no spaces (**adrl-001**). Cover-sheet PDF `adrl-001-ex06.pdf`
  (Word template provided; fill fields only, save as PDF, no extra text) — **Moodle only,
  git-ignored**. Self-grade recommendation drives strictness of grading.
- Late: −5 pts / 24h. V3 software-excellence guidelines (course intro doc) govern grading.

## Hard constraints carried from prior assignments (A1–A5 lessons)
- **PII deny-list** (NEVER in tracked/public content — not even when *describing* the list):
  full name, student IDs, personal email handles, the personal app/vendor handle, and the
  home-dir path. Real values live only in git-ignored `players.local.yaml` / `.env`; the
  cover sheet is git-ignored (Moodle only).
- V3 hard gates: ≤150 LOC/file (LOC metric, excl blanks/comments), ≥85% coverage, ruff 0,
  no hardcoded values (→config), no secrets (.env-example only), **uv-only**, single SDK
  entry for UIs (scripts exempt), version starts at stated version, PRD/PLAN/TODO + ADRs.
- CI: never pin mutable git SHAs in tests; run file-size gate AFTER ruff format.
