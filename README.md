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
