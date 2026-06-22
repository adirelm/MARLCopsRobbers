# PRD — CTDE learning (QMIX / VDN / IQL + OLoRA)

> **Per-area PRD (V3 §2 "dedicated PRD per mechanism").** Decomposes the area-relevant requirements of
> the human-approved master `docs/PRD.md` (`FR-ALG-*` + `FR-OLoRA-*`) — no new requirements/scope/KPIs.
> Per `CLAUDE.md` §1.4 requirements are human-decided; sign-off is **inherited** from the master PRD.
> Canonical text + ACs: `docs/PRD.md`. Theory + equation map: `docs/THEORY.md`; honest results: `docs/ANALYSIS.md`.

## Scope

Centralized-Training / Decentralized-Execution value-based MARL for the **cooperative Cop team**:
**QMIX** (graded primary, monotonic hypernetwork mixer, eq 7) with **VDN** (additive, eq 6) and **IQL**
(independent per-agent Double-DQN, eq 4) arms; one shared `RecurrentQNet` (GRU) trunk + a swappable
(learner × mixer) seam; centralized episodic replay; alternating-best-response self-play; and **OLoRA**
(paper-exact QR rebase of pretrained encoders) for parameter-efficient curriculum transfer.

## Functional requirements (canonical text in `docs/PRD.md`)

| ID | Requirement (essence) |
|---|---|
| **FR-ALG-1** | Value decomposition; **QMIX primary** (`algo.name`), VDN ablation, IQL baseline. |
| **FR-ALG-2** | IQL baseline = independent recurrent Double-DQN per agent (no mixer, no global state). |
| **FR-ALG-3** | Adversarial boundary: Cop/Thief never share a mixer; Thief folded into `T` (POSG). |
| **FR-ALG-4** | Shared net + swappable seam (DRY); IQL is a learner-branch (mixer+global-state drop), not a mixer swap. |
| **FR-ALG-5** | Execution = local obs only — no global state / mixer / opponent injected into the policy. |
| **FR-ALG-6** | Centralized episodic replay; masked loss so post-termination padding contributes zero gradient. |
| **FR-ALG-7** | Self-play: alternating best-response, frozen-opponent windows, heuristic-seeded opponent pool. |
| **FR-ALG-8** | Double-DQN target, masked Huber loss, AdamW param groups, hard target sync (`target_update_interval`). |
| **FR-ALG-9** | Reproducible LOCAL training (seeded RNG + `torch.manual_seed` per learner per `training.seeds`). |
| **FR-OLoRA-1..2** | Locally BC-pretrained encoders; **paper-exact OLoRA** QR rebase of `W0` (eq 10-11) — reject QR-of-random. |
| **FR-OLoRA-3..5** | Freeze `W0`; train only `{A,B}`+head+mixer; wrap encoder Linears only; `rank=4` (assert ≤min(m,n)//2); one encoder spans 2×2→5×5. |
| **FR-OLoRA-6..7** | Four §5.2 data sources → one `CentralizedReplayBuffer`; `global_state` cannot reach the MCP tool signature (type-boundary leakage guard). |
| **FR-OLoRA-8** | OLoRA ablation (vs full-finetune / frozen-base) — **stretch; DESCOPED** (see `README` §6; the ~8× trainable-param reduction is asserted by `tests/unit/test_olora_linear.py`). |

## Key architect decisions (human-decided, §1.4)

- **QMIX primary; MAPPO prose-only** — ADR-0003 (BRIEF makes value-decomposition the graded outcome).
- **Adversarial boundary = POSG; no cop↔thief mixer** — ADR-0006 (opposed rewards violate IGM monotonicity).
- **Shared net + (learner × mixer) seam** — ADR-0008. **OLoRA = QR on pretrained `W0`** — ADR-0010.
- Honest §7.2 result: at the 50-round budget **QMIX is the LEAST stable arm** (more-expressive ⇒ harder to
  train) — reported faithfully, not idealized (`docs/ANALYSIS.md`, `docs/QUALITY.md`).

## Acceptance & evidence

ACs are the per-`FR-*` AC lines in `docs/PRD.md`. Evidence: `tests/unit/` over `src/marl/` (mixers
monotonicity/IGM, VDN sum, IQL per-agent target, Double-DQN, OLoRA QR rebase + freeze + rank guard,
self-play) and the F1/F5 figures from `src/results/make_figures.py`.
