# PRD — Environment (Dec-POMDP Cops & Robbers grid)

> **Per-area PRD (V3 §2 "dedicated PRD per mechanism").** This decomposes the **area-relevant
> requirements of the human-approved master `docs/PRD.md`** — it adds no new requirements, scope, or
> KPIs. Per `CLAUDE.md` §1.4 requirements are human-decided; sign-off is **inherited** from the master
> PRD approval. The canonical, authoritative requirement text + acceptance criteria live in `docs/PRD.md`
> (the `FR-ENV-*` block); this file is the per-area index a grader follows for the environment.

## Scope

The custom, grid-agnostic `CopsRobbersEnv` (`src/marl/env/`) realizing the formal **Dec-POMDP**
⟨N, S, A, T, R, Ω, O, γ⟩ (BRIEF eq 1) for the pursuit game: simultaneous joint moves, barriers, capture,
a 2×2→5×5 curriculum, and the strict **train-global / execute-local** split that makes CTDE faithful.
Formal model + equation map: `docs/THEORY.md`.

## Functional requirements (canonical text in `docs/PRD.md`)

| ID | Requirement (essence) |
|---|---|
| **FR-ENV-1** | Formal Dec-POMDP tuple ⟨N,S,A,T,R,Ω,O,γ⟩, grid-agnostic env (eq 1). |
| **FR-ENV-2** | Train/exec split (top-graded): `step()` returns LOCAL obs only; global `S` only via `env.state()`. |
| **FR-ENV-3** | Per-agent actions; `PLACE_BARRIER` cop-only; **additive −inf** legal masking (not multiplicative). |
| **FR-ENV-4** | Simultaneous-resolution transition; barriers/edges impassable to both; barrier consumes the move (≤5). |
| **FR-ENV-5** | Capture on shared cell **incl. one-step swap** (architect rule, §1.4); terminate on capture or step≥25. |
| **FR-ENV-6** | Potential-based shaped reward (Ng-1999), team `Φ=−min_i Manhattan/d_max`, derived `d_max`, `Φ(terminal)=0`; §3.4 scoreboard is a SEPARATE `Scorer`. |
| **FR-ENV-7 / 7b** | 5×5 egocentric obs (5 ch + scalars), radius-gated opponent channel; stage-invariant 77-float centralized `S` pad (train-only). |
| **FR-ENV-8** | Dynamic-grid curriculum `[[2,2],[3,3],[4,4],[5,5]]`, per-stage cop counts `[1,1,2,1]`, success-gated promotion. |
| **FR-ENV-9** | No-hardcode: all env numerics in `config/config.yaml`. |

## Key architect decisions (human-decided, §1.4)

- **Swap-as-capture** + **capture-beats-timeout** — ADR-0004 (BRIEF silent; deliberate rule call).
- **Potential-based shaping, train-only, `Φ(absorbing)=0`, derived `d_max=(H−1)+(W−1)`** — ADR-0005
  (test-acceptance change, signed off before `transition.py`/`reward.py`).
- The environment is **deterministic + RNG-free by design** (reproducible CI); partial observability comes
  from the bounded **view radius**, not observation noise — there is no stochastic-transition/obs option.

## Acceptance & evidence

ACs are the per-`FR-ENV-*` AC lines in `docs/PRD.md`. Evidence: `tests/unit/test_*` over
`src/marl/env/` (transition, reward potential-invariance, capture/swap, masking, curriculum, obs/state
pad) and the architecture leak tests (`tests/architecture/` — no global state on the execution path).
