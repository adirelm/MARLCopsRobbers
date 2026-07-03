# PROMPTS — MARL Cops & Robbers (architect → implementer trail)

> Evidence for the **Human ↔ AI Responsibility Contract** (CLAUDE.md §1.4): the
> developer is the **architect** (decides PRD/architecture/ADRs/acceptance/
> sign-off), the AI is the **implementer** (codes against an approved spec).
> This log is maintained **per V3 §8** as the §1.4 evidence trail. It is
> enumerated in `test_required_docs_present`, so the file must exist on every
> commit. Each row records the prompt — **verbatim where captured, otherwise a
> faithful summary marked "(summary)"** — the landing commit(s) it produced, and
> the human-judgment call that gated it. The full literal trail is
> `git log --oneline`; the SHAs below resolve in a fresh clone.

## How to read this log
- **Prompt** — the literal instruction given to the implementer.
- **Commit** — the resulting commit hash(es) (per-phase landing commits).
- **Human-judgment annotation** — the architect-decided, non-delegable call
  (CLAUDE.md §1.4 table) that gated or shaped the prompt.

## Phase 0 — Bootstrap (scaffold + gates)
| Prompt | Commit | Human-judgment annotation |
|---|---|---|
| _Scaffold the A6 repo tree + V3 gate tooling + PRD/PLAN/TODO/ADRs from the distilled brief_ (summary; literal trail in `git log`) | `c155eb8`, `18e98f6` | Architect chose the gate thresholds (≤150 LOC, ≥85% cov, ruff-0, version 1.0.0) and the layer boundaries before any code. |

## Phase 1 — Theory / env / reward / scorer
| Prompt | Commit | Human-judgment annotation |
|---|---|---|
| _Implement the theory-first env: Dec-POMDP primitives + reward/scorer + config loader (TDD)_ (summary) | `4398e15` | Architect fixed the Dec-POMDP/POSG tuple, the §3.4 scoreboard (20/10/5/5), and the Ng-1999 shaping potential. |

## Phase 2–3 — pursuit rules + minimal-grid pipeline
| Prompt | Commit | Human-judgment annotation |
|---|---|---|
| _Implement the full basic pursuit (transition/observation/env/curriculum) then the full pipeline on a minimal grid (replay, data sources, heuristics, 2×2 smoke) — TDD_ (summary) | `8ecb5a0`, `ced40cc` | Architect fixed the §3 rules ADRs (simultaneous resolution, swap=capture, barrier-as-move) before the transition code landed. |

## Phase 4 — CTDE learners (QMIX primary · VDN · IQL baseline)
| Prompt | Commit | Human-judgment annotation |
|---|---|---|
| _Build nets/mixers, the CTDE learners, OLoRA+BC, self-play trainer + thin SDK-routed train/sweep scripts (TDD)_ (summary) | `b7d700d` … `06a4bd2` | Architect chose QMIX primary + VDN ablation + IQL §7.2 baseline, the Mixer ABC seam, paper-exact OLoRA (QR on pretrained W0), and the seeds [7, 17, 37, 71, 107]. |

## Phase 5–6 — MCP layer + full local match
| Prompt | Commit | Human-judgment annotation |
|---|---|---|
| _Wire the MCP protocol (schemas/auth/controller), dual FastMCP servers + typed clients + referee, the §5 gatekeeper, then the canonical tool set + §3.5 report assembly + full local match_ (summary) | `890cee6` … `813903e` | Architect fixed the canonical 5-tool contract (no propose/commit fork), the referee-mediated topology (ADR-0011), and the §5 egress-governance requirement (ADR-0009). |

## Phase 7–9 — GUI, cloud Stage-2, Gmail report
| Prompt | Commit | Human-judgment annotation |
|---|---|---|
| _Build the Pygame god-view spectator (purity-gated) + screenshots; the RS256-JWT cloud entrypoints/runbook; the idempotent §3.5 Gmail sender_ (summary) | `0e345d8` … `1c7e414` | Architect chose Pygame (ADR-0014), Prefect-Horizon-primary/Render-fallback (ADR-0012), App-Password-over-OAuth (ADR-0013), and the HARD no-send-without-explicit-go email gate. |

## Phase 10–11 — results, README §7, gates
| Prompt | Commit | Human-judgment annotation |
|---|---|---|
| _Generate the figure pipeline + the 45-run matrix figures, the V3-§9 sensitivity sweep, ISO-25010/cost docs, the SDK-only notebook; author README §7; run the V3 gate audit_ (summary) | `17af07c` … `0f66a16` | Architect signed off the honest §7.2 result (QMIX least stable — reported, not idealized) and the figure-manifest drift gate (R8). |

## Phase P-bonus — Minimax-Q equilibrium baseline (v1.1.0)
| Prompt | Commit | Human-judgment annotation |
|---|---|---|
| "been reset, continue" → execute the deferred **L11 §5 self-challenge** per `planning/P-BONUS_NASHQ_IMPL_SPEC.md` (verbatim) | `f2118e4` … `177c7d9` | Architect deferred the bonus to a limits-reset session, then authorized the autonomous TDD build. The convergence fix (decaying α + GLIE) and the −γ^(H−1) escape-floor framing were implementer calls, reported back honestly (ANALYSIS §10). |
| "Merge to main + tag v1.1.0" (verbatim) | `4b5349f` (tag `v1.1.0`) | Architect's §1.4 final code-review / merge / version-release sign-off. The §3.5 report email remains a HARD no-send-until-explicit-"send" gate. |
