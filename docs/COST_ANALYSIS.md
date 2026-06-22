# Cost analysis (§11)

Two cost envelopes: the AI-assisted development token spend, and the RL training compute.

## (a) AI-assisted development — token cost

The build was AI-assisted (architect ↔ implementer, CLAUDE.md §1.4) with **Claude Opus 4.8**
as the primary implementer and **Codex** for adversarial per-phase validation. Counts below
are ESTIMATES over the full A6 build (P1–P11); exact figures vary with retries and context.

| Model | Role | ~Input tok | ~Output tok | $/1M in | $/1M out | ~Cost |
|---|---|---|---|---|---|---|
| Claude Opus 4.8 | primary implementer | ~14.0 M | ~0.9 M | 15.00 | 75.00 | ~$278 |
| Codex (validation) | per-phase reviewer | ~0.5 M | ~0.1 M | 1.25 | 10.00 | ~$1.6 |
| **Total** | | **~14.5 M** | **~1.0 M** | | | **~$280** |

Input dominates: each turn re-reads a large context (config, prior modules, the run log).
The single biggest lever is **prompt-cache reuse** — keeping turns within the 5-minute cache
TTL avoids re-billing the conversation prefix uncached.

## (b) RL training-compute envelope (LOCAL only)

Training is LOCAL by mandate (BRIEF §5.2 — no cloud training spend). The budget identity is
config-pinned: `selfplay.rounds (50) × selfplay.episodes_per_round (100) = 5000` episodes per stage.

| Axis | Value |
|---|---|
| Episodes / stage | 5,000 |
| Curriculum stages | 4 (2×2 → 3×3 → 4×4 → 5×5) |
| Algorithm arms | 3 (QMIX, VDN, IQL) |
| Seeds | 5 (7, 17, 37, 71, 107) |
| Figure matrix | 3 × 5 × 4 = 60 runs |
| Measured rate (2×2) | ~1.8 s/round → ~1.5 min per 50-round run |

Wall-clock scales with episode length (larger grids → more steps/episode), so the 5×5 stage
dominates. Runs are SERIAL and thread-capped (`apply_compute_limits`), so a full sweep never
freezes the host; the append-only log makes the sweep resumable.

## (c) Optimization strategies

- **OLoRA fine-tuning** trains only `{A, B, head, mixer}` (the GRU + encoder stay frozen) —
  ~8× fewer trainable parameters than full fine-tuning across curriculum transfer.
- **Seeded reuse + curriculum transfer**: each stage warm-starts from the previous stage's
  nets rather than from scratch.
- **Compute governance**: torch thread caps from config bound every SDK training path.
- **Prompt-cache discipline** (dev cost): batch edits per turn; keep within the cache TTL.
