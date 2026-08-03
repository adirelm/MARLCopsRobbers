# adrl-001 → biu-azri — one correction, then we are ready

## 1. Token — WORKING now, confirmed

Same value, but your deployment accepts it now; something on your side clearly landed
between your last message and this test. Re-ran the full path just now:

| call | result |
|---|---|
| `GET /health` | `200 {"status":"ok","version":"1.00"}` |
| `POST /new_sub_game` (thief, `[1,1]`) | `200 {"ok":true}` |
| `POST /request_move` (tick 0) | `200 {"action":"up"}` |
| identical re-POST | `200 {"action":"up"}` — **idempotent** ✓ |

Nothing further needed here. We hold it only in a git-ignored env file and will delete it
after the match; noted that you will rotate.

## 2. The frozen table is the one we WITHDREW — please re-freeze

This is the one thing that must be settled before we play, and it is our fault for
sending a table we then had to retract.

You wrote: *"every pair satisfies P6 (**minimum distance 4**, all > 2)"*. That value
identifies the table unambiguously:

| list | Manhattan distances | min |
|---|---|---|
| `101, 202, 303, 404, 505, 606` — **withdrawn** | 4, 4, 5, 4, 4, 6 | **4** |
| `911, 822, 733, 644, 555, 466` — current proposal | 5, 3, 5, 4, 5, 3 | 3 |

So you verified and froze the **withdrawn** list. Your verification work was correct —
every row you checked really does hold — it was just applied to a superseded table. A
corrected reply went out after that draft; if it did not reach you, that is on us.

**Why we withdrew it:** those are our local dress-rehearsal seeds. Our own agents have
already played those exact layouts, which would hand us prior exposure you never had. Our
match runner also refuses to start a real match on them — an independent tripwire that is
how we caught it.

**Two ways to close this, your pick:**

- **You send the seeds.** Any 6+ integers; we publish the layouts they resolve to and you
  freeze those. We prefer this: we host the referee, so whoever chooses can preview the
  layouts first, and that surface points at us.
- **Or freeze our current proposal:** `911, 822, 733, 644, 555, 466` → the layouts in the
  second row above. Note two pairs sit at distance 3 rather than 4. Both still satisfy P6
  (`> 2`, so neither side starts inside the other's radius-2 view), but they are tighter
  starts than the withdrawn list — worth knowing before you agree, not after.

Whichever you choose, re-verify it the same way you did the first one, and we are frozen.

## 3. Availability — neither of us has actually sent one

You are right that our line 130 was still a fill-in; that was ours to fill and we missed
it. Ours is at the bottom of this message.

Your item 2 is also still the placeholder text (`<<< FILL IN: your window… >>>`), so we do
not have yours either. Send it and we will pick the overlap.

## 4. C1 — settled, and thank you for measuring it

`stay` stays out. Your numbers make the case better than the argument did: 320 of 749
thief moves select it, but 313 are already exact no-ops under P3, leaving 0.93% residual
distortion. That is well under what a re-freeze and untested policies would have cost
either of us. We would have conceded it — we are glad the data said it was not needed.

## 5. Everything else — agreed as stated

C3 log schema, the bounded re-hello (your `400` with "unknown session" in the body is
covered by our 4xx rule), the `version` key on `/health`, and the unscored conformance
sub-game immediately before the real match in the same session.

---

## Outstanding, both sides

1. **You:** re-freeze the seed list (send yours, or freeze `911, 822, 733, 644, 555, 466`).
2. **You:** your availability window.
3. **Us:** availability, below.

Nothing else is open. Once 1 and 2 land we run the conformance game and then the match.

Our availability: **we are flexible — you pick the slot.** The match is short: our recorded
cloud run took 4m29s for all six sub-games, and most of that was a single free-tier wake-up.
Add the conformance game and the whole session is ~15 minutes. Your service is always-on and
our referee drives everything, so what we actually need is a window where one human on each
side can answer if something breaks, and then do the byte-compare and send the two emails —
call it 30 minutes, not an evening.

**Propose any slot in the next 48 hours (Asia/Jerusalem) and we will take it.** If you would
rather we name one: **Sun–Thu, 20:00–23:00 Asia/Jerusalem** suits us any day this week.
