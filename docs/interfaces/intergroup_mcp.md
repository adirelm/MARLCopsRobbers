# §9 Inter-Group Bonus — Rules Agreement + Wire Protocol (adrl-001 ↔ partner)

This is the pre-game contract for the ex06 §9 bonus match. **§9.3 makes agreement
existential: if the two groups don't agree on the results, BOTH get 0** (and the loser of
an agreed match still gets +7) — so every parameter below is fixed IN WRITING before the
first move, and both groups byte-compare their §9.4 JSON drafts before either sends.

## 1. Rules to agree (fill the partner column together, then freeze)

| # | Parameter | adrl-001 value | Agreed |
|---|---|---|---|
| 1 | Board / length / games | 5×5, ≤25 moves, 6 sub-games (§3.1 — non-negotiable) | ☐ |
| 2 | Scoring | §3.4 Table 1: capture cop 20 / thief 5; timeout thief 10 / cop 5 | ☐ |
| 3 | Role alternation | §9.1: sub-games 1–3 group_1 = cop; 4–6 swapped | ☐ |
| 4 | Move resolution | SIMULTANEOUS (both moves resolved from the same pre-move state) | ☐ |
| 5 | Swap = capture? | YES — a cop↔thief one-tick cell swap counts as capture (blocks the degenerate pass-through exploit); capture checked before timeout on move 25 | ☐ |
| 6 | Start positions | Seeded random with `manhattan(cop, thief) > view radius`; **the SAME 3 seeds are reused for sub-games 1–3 and 4–6**, so both groups play cop and thief from identical layouts (perfect fairness across the swap) | ☐ |
| 7 | Observability | Partial: Manhattan radius 2 at 5×5 (each side sees the opponent only within radius; last-seen memory is each side's own business) | ☐ |
| 8 | Barriers | ALLOWED per §3.3 (cop-only, ≤5 per sub-game, placement consumes the move, barrier cells impassable to BOTH) — §3.3 defines them as part of the game; each group's cop may or may not use them | ☐ |
| 9 | Seeds | Base seed + the 3 spawn seeds chosen jointly and recorded here BEFORE playing | ☐ |
| 10 | Crash handling | §3.7 verbatim: a sub-game interrupted by a technical fault is VOID and replayed with the next agreed seed; per-move RPC timeout = 10 s; persistent no-show → match called off (no forfeit claims) | ☐ |
| 11 | Results agreement | Each group computes totals_by_group independently from the shared per-move log → diff → byte-compare full §9.4 JSON drafts → only then both set `mutual_agreement: true` and send | ☐ |

## 2. How the match runs (proposed topology)

**adrl-001's referee drives both sides** (we already have a §3.7-compliant referee,
timestamps, scoring, and real-HTTP serving — see `scripts/serve_match_http.py`). To stay
implementation-neutral, agents talk a MINIMAL wire format (raw positions — each side does
its own encoding/policy):

- Transport: HTTP POST, JSON, one endpoint per agent, bearer token per §5.3.
- `POST /new_sub_game` `{"session_id": str, "grid": [5,5], "your_role": "cop"|"thief", "your_pos": [r,c], "max_moves": 25}` → `{"ok": true}`
- `POST /request_move` `{"session_id": str, "tick": int, "your_pos": [r,c], "opponent_pos": [r,c] | null, "barriers": [[r,c], ...], "barriers_left": int}` → `{"action": "up"|"down"|"left"|"right"|"place_barrier"}`
  - `opponent_pos` is `null` when outside the agreed view radius; `barriers` lists only barrier cells within the radius (plus own-cell). `place_barrier` is cop-only.
  - Idempotency: a re-POST of the same `(session_id, tick)` returns the same action.
- `GET /health` → `{"status": "ok"}`
- The referee logs every request/response with timestamps; the full log is shared with the partner after each sub-game.
- **Host-advantage neutralizer:** the partner may independently re-verify every transition from the shared log + agreed seeds; on request we also play an unscored mirror sub-game on their runner.

## 3. Report + submission mechanics (§9.3–9.4)

1. Both groups exchange identity blocks privately (group name string, repo URL, students role/full_name/id) → ours goes in `players.local.yaml`, the partner's in `players.partner.local.yaml` (both git-ignored).
2. After the 6 valid sub-games: each group builds its §9.4 JSON (`src/reporting/bonus.py` on our side), byte-compares `sub_games` / `totals_by_group` / `bonus_claim` with the partner's draft, sets `mutual_agreement: true`.
3. Each group sends ONE bonus email to the lecturer — subject `[MARL Bonus Game] <group_1> vs <group_2> – Final Report` (our send is idempotent + human-gated).
4. Each group documents in its README: the opponent's name, the final score, its bonus claim, and screenshots of the bonus match (§9.3).

## 4. Status

- [x] Our side built: `src/reporting/bonus.py` + `bonus_send.py`, `docs/schema/bonus.schema.json`, partner-identity intake, this agreement doc.
- [ ] Agreement filled + frozen with the partner (all 11 rows checked).
- [ ] Match played (6 valid sub-games) + logs shared.
- [ ] JSON drafts byte-compared; both `mutual_agreement: true`.
- [ ] Both emails sent (ours: explicit human "send" only). README §9 documentation + screenshots updated with the real results.
