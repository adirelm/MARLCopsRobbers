# CLAUDE.md — Global Coding Standards for MARLCopsRobbers (Assignment 6)

## Project Context

MARLCopsRobbers is a multi-agent RL project for the Bar-Ilan *Vibe Coding & RL*
workshop (Lecture 10, MARL / CTDE). It builds — **from scratch** — a custom
`CopsRobbersEnv` Dec-POMDP/POSG pursuit on a dynamic grid, a **CTDE** training
stack (**QMIX** primary monotonic mixer, **VDN** ablation arm, **IQL** the §7.2
non-stationarity baseline) over ONE shared recurrent (GRU) Q-net, two autonomous
**FastMCP** servers (cop + thief over HTTP), a Pygame god-view GUI, and an
end-of-game Gmail report. Single source of truth for every algorithm-relevant
value: `config/config.yaml` (read via `src/utils/` config loader).

## Human ↔ AI Responsibility Contract (§1.4)

§1.4 of the submission guidelines frames the developer as **architect**, the AI
as **implementer**. This table says **who decides** before any code is generated.

| Concern | Human-decided (non-delegable) | AI-delegated |
|---|---|---|
| Requirements (PRD, scope, success criteria, KPIs) | ✅ | — |
| Architecture (ADRs, layer boundaries, public API/SDK shape) | ✅ | — |
| Test acceptance criteria + the assertions that must hold | ✅ | — |
| Final code-review sign-off + commit message intent | ✅ | — |
| Self-score / grade claim against the rubric | ✅ | — |
| Cost-budget envelope (what we're willing to spend) | ✅ | — |
| Code generation against an approved spec | — | ✅ |
| Refactoring within an existing public API | — | ✅ |
| Test scaffolding + boilerplate from a written spec | — | ✅ |
| Docstring drafts (human edits before commit) | — | ✅ |
| Routine doc maintenance (link fixes, freshness sweeps) | — | ✅ |
| Lint / format auto-fixes | — | ✅ |

**Operating rule.** If any AI-generated change would alter a human-decided
column above (a new public SDK method, a changed test assertion, a weakened
quality gate, a choice between two architectures), the human signs off
*explicitly first* — typically by approving the PRD/PLAN edit, then letting the
AI execute against it. This trail is evidenced in `docs/shared/PROMPTS.md` (the
literal prompts) and in the per-section commit messages naming the § addressed.

## Hard Constraints (apply to ALL files)

1. **File size ≤150 LOC** per `.py` (excl. blanks/comments), **tests included**.
   Split into modules before hitting the cap; never compress to dodge it.
   Run the file-size gate AFTER `ruff format` (format can expand a file).
2. **TDD** — RED→GREEN→REFACTOR. Tests before implementation. **≥85% coverage**
   (`fail_under=85`). Run: `uv run pytest tests/ --cov=src --cov-report=term-missing`.
3. **OOP / SDK single entry** — all business logic reachable via the SDK
   (`src/sdk/`); CLI / scripts / GUI / MCP import only the SDK (no logic in UIs
   or MCP servers). Use inheritance + a Mixer ABC seam (IQL/VDN/QMIX share ONE
   recurrent Q-net). No code duplication (DRY).
4. **No hardcoded values** — every **algorithm-relevant** parameter (MARL
   hyperparameters, reward weights, the §3.4 scoreboard, grid/curriculum dims,
   seeds, MCP hosts/ports, auth config) lives in `config/config.yaml` via the
   config loader. **Local UI-styling literals** (Pygame colours/widths/fonts/FPS
   in `src/gui/palette.py`, matplotlib `dpi`/`alpha`/`fontsize`) stay in their
   rendering modules. The test for "should this be in config" is: *"would a
   grader, contributor, or future-me ever want to change this without editing
   source?"* If yes → config. If no → keep it local.
5. **Zero Ruff violations** — `uv run ruff check src/ tests/ scripts/`. The lint
   select includes `D` (pydocstyle, google convention); every module/component
   carries a docstring (§16). Tests are exempt from `D`.
6. **uv only** — `uv sync --dev`, `uv run …`. No pip / conda / venv / requirements.txt.

## Canonical Values (from config/config.yaml — never duplicate, cite)

- **Graded match (BRIEF §3):** grid **5×5**, **≤25 moves** per sub-game, a full
  game = **6 sub-games**, **1 cop vs 1 thief** (the 4×4 curriculum stage trains a
  2-cop team so the QMIX mixer is genuinely multi-agent — does NOT change the
  graded rules).
- **Scoring (§3.4, REPORT-ONLY — never the RL signal):** cop_win **20**,
  thief_win **10**, cop_loss **5**, thief_loss **5**.
- **Algorithms (§5.2 / §7.2):** **QMIX** primary (monotonic mixer), **VDN**
  ablation arm, **IQL** the mandated non-stationarity baseline.
- **Seeds (§7.3 mean±SE):** **7, 17, 37, 71, 107**.
- **Identity:** group code **adrl-001**; version **1.1.0** (`src/__init__.py`
  `__version__` + `config.version` + `pyproject`).

## Secrets / PII Deny-List Rule

NO secrets and NO PII in tracked files. Transport secrets (MCP tokens, RSA keys,
Gmail App Password) live in `.env` (git-ignored; placeholders in
`.env-example`). Real student names/IDs + the GitHub owner slug live in
`players.local.yaml` (git-ignored; placeholders in `players.example.yaml`). The
numeric self-grade lives ONLY on the git-ignored Moodle cover sheet
`adrl-001-ex06.pdf` — the tracked repo claims no self-grade. `check_pii.py` MUST
**load deny-list patterns from the git-ignored sources or derive them generically
(regex shapes)** — it must NOT enumerate literal deny-list tokens in tracked
code (A4 lesson).

## Version Control

- Group code `adrl-001`; share read access with the lecturer's GitHub handle
  `@rmisegal`. Version starts `1.0.0`. Prompt log in `docs/shared/PROMPTS.md`.
- Cover sheet `adrl-001-ex06.pdf` via the official Moodle template (self-grade
  ONLY on the PDF). Push often — course repos live in Google Drive (sync can
  silently remove `.git` mid-session).

## Config Structure

`config/config.yaml`: `version`, `project`, `game`, `env` (+ `curriculum`),
`algo` (+ `mixer`), `nets`, `olora`, `bc`, `replay`, `selfplay`, `training`,
`reward`, `mcp`, `cloud`, `gmail`, `gui`, `paths`, `logging`. Egress rate limits
live in `config/rate_limits.json` (single source). Accessed via the config loader.
