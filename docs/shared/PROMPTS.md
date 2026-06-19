# PROMPTS — MARL Cops & Robbers (architect → implementer trail)

> Evidence for the **Human ↔ AI Responsibility Contract** (CLAUDE.md §1.4): the
> developer is the **architect** (decides PRD/architecture/ADRs/acceptance/
> sign-off), the AI is the **implementer** (codes against an approved spec).
> This log is maintained **per V3 §8** as the §1.4 evidence trail. It is
> enumerated in `test_required_docs_present`, so the file must exist on every
> commit. Each row records the verbatim prompt, the landing commit(s) it
> produced, and the human-judgment call that gated it. The full trail is
> `git log --oneline`; the SHAs below resolve in a fresh clone.

## How to read this log
- **Prompt** — the literal instruction given to the implementer.
- **Commit** — the resulting commit hash(es) (per-phase landing commits).
- **Human-judgment annotation** — the architect-decided, non-delegable call
  (CLAUDE.md §1.4 table) that gated or shaped the prompt.

## Phase 0 — Bootstrap (scaffold + gates)
| Prompt | Commit | Human-judgment annotation |
|---|---|---|
| _(to be filled at landing)_ | _(SHA)_ | Architect chose the gate thresholds (≤150 LOC, ≥85% cov, ruff-0, version 1.0.0) and the layer boundaries before any code. |

## Phase 1 — Theory / env / reward / scorer
| Prompt | Commit | Human-judgment annotation |
|---|---|---|
| _(to be filled at landing)_ | _(SHA)_ | Architect fixed the Dec-POMDP/POSG tuple, the §3.4 scoreboard (20/10/5/5), and the Ng-1999 shaping potential. |

## Phase 4 — CTDE learners (QMIX primary · VDN · IQL baseline)
| Prompt | Commit | Human-judgment annotation |
|---|---|---|
| _(to be filled at landing)_ | _(SHA)_ | Architect chose QMIX primary + VDN ablation + IQL §7.2 baseline, the Mixer ABC seam, and the seeds [7, 17, 37, 71, 107]. |
