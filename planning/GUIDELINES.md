# V3 Software-Excellence Guidelines — distilled rubric (for A6 plan review)

> Faithful distillation of `software_submission_guidelines-V3.pdf` (Dr. Y. Segal,
> v3.00, ~20 sections). Graded INDEPENDENTLY of the per-assignment brief. §19: it's
> an *excellence* bar (more criteria met ⇒ higher grade), but the **hard gates are
> non-negotiable**. Use this to verify the A6 plan (PRD/PLAN/TODO/config) covers each.

## THE 8 HARD GATES (must all be green)
| Gate | Threshold | § |
|---|---|---|
| File size | **≤150 LOC** (excl blanks/comments), **tests too** | §3.2/§6.1 |
| Coverage | **≥85%**, suite fails below (`fail_under=85`) | §6.2 |
| Ruff | **0 violations** (`select=["E","F","W","I","N","UP","B","C4","SIM"]`, line 100/py310 spec; stricter 110/py311/+PL/+RUF is an accepted documented deviation) | §7.1 |
| Hardcoded values | **0** in source — all configurable values via `config/` loader | §7.2 |
| Secrets | **0** tracked + **`.env-example` committed**; `.gitignore` covers `.env *.pem *.key credentials*` | §7.4 |
| Package mgr | **uv only** — no pip/python -m/venv/requirements.txt; `uv.lock` committed | §8.4 |
| SDK entry | all business logic via a single **SDK**; CLI/GUI/notebook import only the SDK | §4.1 |
| Version | starts at the stated version (here **1.0.0**) in code + config | §8.1 |

## SECTION CHECKLIST (§1–§20)
- **§1 SDLC + §1.4 Human↔AI contract.** Order: PRD→PLAN→TODO→dedicated PRDs→approve→code(TDD)→results→README. Plan-before-code (docs predate code). §1.4 architect(human)/implementer(AI) contract documented (CLAUDE.md).
- **§2 Mandatory structure + docs.** Root **README.md** = full manual with ALL SIX: **Installation · Usage · Examples/Screenshots · Configuration · Contribution Guidelines · License & Credits** (Contributing + Config are the most-missed). **docs/** with PRD (overview, KPIs, **acceptance criteria**, functional+non-functional, user stories, assumptions, milestones), PLAN (**C4 + UML + deployment diagrams**, **ADRs** with trade-offs, API/schemas), TODO (phased, status, **definition-of-done**). **Dedicated PRD per algorithm/mechanism** (`docs/prd/PRD-<NAME>.md`). Dirs: src/ tests/ docs/ config/ data/ results/ assets/ notebooks/ (or justified N/A).
- **§3 Code docs + 150-line cap.** Every .py ≤150 LOC (split, never compress; tests included). Docstring on every module/class/function; comments explain "why"; DRY.
- **§4 SDK architecture + OOP.** Single SDK entry; no business logic in CLI/GUI/controllers. OOP, no duplication (extract at 2 copies).
- **§5 API Gatekeeper + rate control.** *N/A only if NO external API calls at runtime.* If external APIs ARE called (HTTP/openai/anthropic/requests/httpx/smtplib/Gmail) → one central **ApiGatekeeper** (`execute`, `get_queue_status`), rate limits from versioned **`config/rate_limits.json`**, **FIFO overflow queue** (no crash), all calls logged.
- **§6 TDD + QA.** RED→GREEN→REFACTOR; every module has a test file; every public fn ≥1 test; happy+error paths; ≥85% cov with `fail_under=85`; conftest fixtures; external deps mocked; test files ≤150 LOC.
- **§7 Lint, config, security.** Ruff 0. No hardcoded configurable values (allowed: physical/math constants, defaults, constants.py, Enum). Versioned config under config/ (.yaml ok). `.env` git-ignored; **`.env-example` committed**; no secrets in code.
- **§8 Version control + uv.** Version starts **1.00/1.0.0** in code + config. Git: meaningful commits, branches, PRs/review, **tags for major versions**. **Prompt Log** (`docs/PROMPTS.md`). uv only; `uv.lock` committed.
- **§9 Research + results.** **Sensitivity analysis** (controlled parameter sweep, each param's effect documented). **Analysis notebook** consuming the SDK only (LaTeX equations, comparisons, citations). Quality visualizations (bar/line/scatter/heatmap/box) with labels/legend/captions, high-resolution.
- **§10 UI/UX.** *N/A/partial if CLI-only.* If a **GUI exists** → usability criteria + **Nielsen's 10 heuristics** + **screenshot of every screen/state** + workflow + accessibility (typically `docs/UX.md`).
- **§11 Costs.** Token cost breakdown (input/output counts, $/M, total per model) + optimization strategies + budget mgmt. For RL: input-corpus token volume + training runtime.
- **§12 Extensibility + maintainability.** Extension points / plugin or adapter architecture (interfaces, hooks, middleware); ADRs + adapters satisfy this.
- **§13 ISO/IEC 25010.** Address all **8** characteristics (Functional Suitability, Performance Efficiency, Compatibility, Usability, Reliability, Security, Maintainability, Portability) — typically `docs/QUALITY.md`.
- **§14 Project as a package.** `pyproject.toml` name/version/deps; **`__init__.py` in package root + every source subdir**; `__all__` + `__version__`. Imports use package/relative paths, not filesystem-absolute.
- **§15 Parallelism.** **multiprocessing for CPU-bound**, **multithreading/async for I/O-bound** — chosen correctly + documented; thread safety (locks, queue.Queue, context managers, no deadlock) where threads used.
- **§16 Modular building blocks.** Each component documents Input/Output/Setup; SRP; `_validate_config()`→ValueError, `_validate_input()`→TypeError; DI for testability.
- **§17 Final checklist.** Re-tick §2 docs, §4 architecture, §6 tests, §7 config/security, §9 research, §12/§13 standards as a pre-submission pass.
- **§18/§19/§20.** Reference ISO 25010 / Google Eng Practices / Nielsen / MS REST where relevant; depth > box-ticking; appendix restates §1–§17.

## A6-SPECIFIC FLIPS (vs RL-only A1–A5, where these were N/A)
- **§5 ApiGatekeeper → REQUIRED**: A6 makes external calls (peer-MCP HTTP, Gmail, cloud) → needs a central gatekeeper + `config/rate_limits.json` + FIFO queue + logging.
- **§10 Nielsen heuristics → REQUIRED**: A6 has a MANDATORY GUI → `docs/UX.md` mapping all 10 heuristics + a screenshot per screen/state.
- **§15 parallelism → REQUIRED**: two MCP servers + referee = multiple processes → document the process model + I/O-async vs CPU-multiprocessing choice + thread safety.

## N/A / deviation patterns to record explicitly
- §7.3 single `config/config.yaml` instead of split files → acceptable (§20.3).
- Stricter Ruff (110/py311/+PL/+RUF) → accepted documented deviation; never loosen below spec.
