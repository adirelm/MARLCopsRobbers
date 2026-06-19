# MARL Cops & Robbers — Multi-Agent RL on a Dec-POMDP pursuit (Assignment 6)

Bar-Ilan University — *Vibe Coding & Reinforcement Learning* workshop, **Assignment 6**
(group **adrl-001**, solo). Two autonomous AI agents — a **Cop** and a **Thief** — pursue
each other on a dynamic grid under **partial observation**. Trained with **CTDE + value
decomposition** (QMIX primary, VDN ablation, IQL baseline), each agent runs behind its own
**FastMCP server** (localhost → cloud), visualized live in a **Pygame** GUI, with an
end-of-game **Gmail** report.

> **Status: PLANNING COMPLETE — implementation pending.** This README is the submission
> report (brief §7); it is currently a skeleton and fills in as the phased build (PLAN §8,
> TODO P0→P11) lands. Design is locked: see [`docs/PRD.md`](docs/PRD.md),
> [`docs/PLAN.md`](docs/PLAN.md), [`docs/TODO.md`](docs/TODO.md).

## Installation

```bash
uv sync --dev          # uv-only (no pip/conda). add --group mcp --group mail for the servers + report.
uv run pytest tests/ --cov=src   # quality gates (≥85% coverage)
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
uv run python scripts/check_file_sizes.py   # every .py ≤150 LOC
```

## Usage

*(pending implementation — P5/P6/P8)*. Planned single-SDK entry + thin surfaces:

```bash
uv run python -m src.cli train --algo qmix      # local CTDE training (curriculum 2×2→5×5)
uv run python -m src.cli play                    # run a 6-sub-game match over the two MCP servers
uv run python -m src.gui                         # Pygame god-view spectator
```

## Examples

The graded deliverables (brief §7.3) — before/after pursuit, learning curves, loss curves,
GUI screenshots at 2×2/3×3/4×4/5×5, and MCP-comms proof — will appear in §3–§7 below as the
build completes.

## Configuration

All algorithm-relevant parameters live in [`config/config.yaml`](config/config.yaml) (the
single source of truth; brief §3.6, no hardcoding) — game rules (5×5, ≤25 moves, 6 sub-games,
scoring 20/10/5/5), env/observation, QMIX/VDN/IQL, OLoRA, self-play, MCP ports/auth, cloud,
Gmail. Egress rate limits in [`config/rate_limits.json`](config/rate_limits.json). Secrets in
`.env` (see `.env-example`); real identities in `players.local.yaml` (git-ignored) — see
`players.example.yaml`.

## Contributing

Solo project (role A). Workflow: human = architect, AI = implementer (CLAUDE.md §1.4) — the
PRD/PLAN are signed off before code lands. TDD (RED→GREEN→REFACTOR), conventional commits,
every `.py` ≤150 LOC, ruff clean, ≥85% coverage on every push (CI).

## License & Credits

MIT (see `LICENSE`). Built for Dr. Yoram Segal's Vibe Coding & RL workshop. Key references:
Dec-POMDP (Bernstein 2002), VDN (Sunehag 2018), QMIX (Rashid 2018), OLoRA (Büyükakyüz 2024),
MCP (Anthropic 2024) — full bibliography in `docs/PRD.md` §10.

---

## 1.–7. Submission report (brief §7) — *to be authored during the build (README is the §7 paper)*

§1 Game · §2 Skills/architecture · §3 before · §4 after · §5 metrics/ablation/**sensitivity** ·
§6 bug/limitations · §7 academic analysis (Dec-POMDP formalism, non-stationarity, IQL-vs-CTDE,
IGM/monotonicity → QPLEX/Weighted-QMIX). Figures F1–F6 + GUI + MCP-comms screenshots land here.

---

## 7. Academic analysis (brief §7) — outline + figure manifest

> **Skeleton only (T1.5).** §7.1/§7.2/§7.3 below are placeholders filled in as the build lands.
> Equation/citation numbering follows **ex06/BRIEF as primary** (R13): ex06 "eq 2" ≡ L10 "eq 4";
> BRIEF `[2]` (VDN) ≡ L10 `[7]`. Full formalism in [`docs/THEORY.md`](docs/THEORY.md).

### 7.1 Formalism — Dec-POMDP `M` + POSG caveat *(pending)*

The cooperative cop team is a **Dec-POMDP** `M = ⟨N, S, {A_i}, T, R, {Ω_i}, O, γ⟩` (eq 1, cite
`[1]`), with **N = the cooperative COP TEAM** (the Thief is folded into `T`; value decomposition
never crosses the cop/thief boundary). The faithful full game is a general-sum **POSG**
`G = ⟨I, S, {A_i}, {O_i}, P, Ω, {R_i}, γ⟩` (eq 3, NEXP^NP) with `R_cop ≠ R_thief`. Each tuple
symbol is mapped 1:1 to code (`GlobalState`, `Observation`, `actions.py`, `reward.py`, config).
See [`docs/THEORY.md`](docs/THEORY.md).

### 7.2 Analysis — non-stationarity, IQL-vs-CTDE, IGM/monotonicity *(pending)*

(a) Why independent learners (IQL, eq 2/4) face a drifting target vs centralized CTDE; (b) the
IQL/VDN/QMIX credit-assignment ordering on the 4×4 2-cop scenario; (c) IGM + monotonicity limits
(eq 5, eq 7) → QPLEX `[10]` / Weighted-QMIX `[9]`; (d) OLoRA `[7]` as a stability aid, not the
non-stationarity cure; curriculum `[5]`. Evidenced by F5.

### 7.3 Results — the controlled experiment + figure manifest *(pending)*

**Single controlled experiment** (D10 §C): arms **IQL / VDN / QMIX** × seeds **`[7, 17, 37, 71,
107]`** (`training.seeds`) × ladder **`[2, 3, 4, 5]`** (`env.curriculum.stages`), with **identical**
nets / replay / ε-decay / γ / target cadence — only the mixer (or the IQL learner branch) differs.
Per-episode records append to `results/runs/*.jsonl`; `results/experiment_manifest.json` pins seeds,
config hashes, git commit, and run IDs (zero README↔code drift).

| Fig | Content | Generator | Path |
|---|---|---|---|
| **F1** | Both-agent learning curves (reward vs episode, mean±SE) | `python -m src.results.make_figures` (plots `results/runs/*.jsonl`) | `results/figures/learning_curves.png` |
| **F2** | Per-stage TD-loss curves | `python -m src.results.make_figures` | `results/figures/loss_curves.png` |
| **F3** | GUI screenshots at 2×2/3×3/4×4/5×5 (CAPTURED, not plotted) | `scripts/capture_screens.py` (headless) | `results/screenshots/grid_{2,3,4,5}x{n}.png` |
| **F4** | MCP-comms proof — localhost canonical (cloud if P8) (CAPTURED) | redacted cop↔thief comms log / `scripts/smoke_cloud.py` | `results/figures/mcp_comms_local.png` (+ `_cloud.png`) |
| **F5** | IQL vs VDN vs QMIX win-rate/convergence (incl. 4×4 2-cop panel) | `python -m src.results.make_figures` | `results/figures/baseline_comparison.png` |
| **F6** | Scale effect — capture-rate vs grid size | `python -m src.results.make_figures` | `results/figures/scaling.png` |

F1/F2/F5/F6 regenerate from one command (`uv run python -m src.results.make_figures`); F3/F4 are
deterministically captured by their seeded scripts. The OLoRA ablation chart + trainable-param
table also land here (§7.2 evidence).
