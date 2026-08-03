# adrl-001 → biu-azri — reply to your items 1–5

Received and recorded: group code, repo, students, endpoint. Your self-test results match
what our referee expects, including the idempotent re-POST and both 401 cases.

**Still outstanding:**

1. **The bearer token you sent is REJECTED by your own live deployment.** Details and a
   reproduction below — please re-check and resend.
2. **Availability window.** Ours is below; send yours and we will pick the overlap.

### Token check — your endpoint rejects the token you sent

We tested the moment it arrived, against the live URL. Your service and auth layer are
healthy; it is specifically this token value that the running deployment does not accept:

| call | result |
|---|---|
| `GET /health` | `200 {"status":"ok","version":"1.00"}` |
| `POST /new_sub_game` with a deliberately wrong token | `401` (correct) |
| `POST /new_sub_game` with no `Authorization` header | `401` (correct) |
| `POST /new_sub_game` with **the token you sent** | `401 {"error":"bearer token invalid or revoked"}` |

To rule out anything on our side we passed the string literally on the command line,
bypassing our env file entirely — same `401`. The value we hold is 43 characters and its
`sha256` begins `d06ce58286d8`; check that against what you meant to send, so we can tell
a transcription problem from a rotation.

Most likely one of: the service redeployed and picked up a different secret than the one
you tested with, the value in your hosting dashboard differs from the one in your local
run, or it was rotated after you sent it. Worth resolving now — this exact failure on
match day reads as a dead endpoint and would burn P8 retries.

When you resend, we will re-run the four checks above and confirm before anything is
frozen.

**`/health` extra `version` key:** keep it. We check `status == "ok"` and ignore
additional fields, so it costs nothing and is useful.

---

## C1 — "stay" in the action enum: DECLINED (but say so if you want it and we will add it)

Our serving policies were trained on an action space without `stay` — cop
`{up,down,left,right,place_barrier}`, thief `{up,down,left,right}` — and it is
config-gated off (`env.actions.enable_stay: false`). Adding it now means either shipping
policies that were never trained against it, or retraining and re-validating days before
the match. We would rather not change either side's agents after freezing.

Your own point is what decides it: you already play inside P3, so this is exactness, not
capability. We are also being straight with you that declining happens to suit us — our
agents have no no-op to give up, so the concession would be one-directional. If you would
rather have it, say so and we will enable it for **both** sides and re-freeze; it is a
small referee change and we would rather amend the rules than have you play a policy you
did not design.

## C2 — freeze the literal start positions: ACCEPTED

You are right, and it is the strongest point in your reply. A seed is an index, not a
specification: it only reproduces a layout under the referee's own RNG, and we host the
referee, so you could not have verified fairness before the §5 byte-compare. We had
already amended the brief for this before your message arrived; here is the filled list.

**Who picks the seeds — read this first.** We host the referee, so whoever chooses the
seeds can preview the layouts they produce before proposing them. That is a real
cherry-picking surface, and it points at us. **We would rather you pick.** Send any 6+
integers and we will publish the layouts they resolve to; you then freeze those. If you
would rather not bother, here is our proposal:

**P7 proposed list — 5×5, one cop vs one thief. Cells are `[row, col]`, 0-indexed, origin top-left.**

| pair | sub-games | seed | cop start | thief start | manhattan |
|---|---|---|---|---|---|
| 1 | 1 and 4 | `911` | `[0, 4]` | `[1, 0]` | 5 |
| 2 | 2 and 5 | `822` | `[0, 3]` | `[2, 2]` | 3 |
| 3 | 3 and 6 | `733` | `[2, 3]` | `[4, 0]` | 5 |
| spare 1 | — | `644` | `[3, 2]` | `[1, 0]` | 4 |
| spare 2 | — | `555` | `[1, 0]` | `[0, 4]` | 5 |
| spare 3 | — | `466` | `[1, 4]` | `[1, 1]` | 3 |

**Correction to an earlier draft:** a previous version of this reply carried the list
`101, 202, 303, 404, 505, 606`. Disregard it. Those are our local dress-rehearsal seeds —
our own agents have already played those exact layouts, which would have handed us prior
exposure you never had, and our match runner has a tripwire that refuses to start a real
match on them. We caught it before sending; flagging it so you do not work from a stale
copy if one reached you.

Read the table this way: in sub-game `k` the group playing **cop** starts at the cop cell
and the group playing **thief** at the thief cell; in the mirror `k+3` the same two cells
are reused with the roles swapped. That is what makes the pair fair. Every layout
satisfies P6 (`manhattan > 2`, so neither side starts inside the other's radius-2 view) —
check the column yourself rather than taking it from us.

Verify each `new_sub_game` payload against whichever table we end up freezing: the
`your_pos` we send you must equal the cell for your role in that sub-game. If it ever does
not, stop the match and tell us before anything is byte-compared.

**Whichever list we settle on, reply "frozen" and it is binding.** After the first
sub-game starts, P7's void amendment is the only thing that may change a layout.

## C3 — referee-log schema up front: ACCEPTED

The log is JSONL, one event per line, in wall-clock order. Four event shapes, all with an
ISO-8601 Asia/Jerusalem `ts` with millisecond precision:

```json
{"ts": "...", "direction": "request",  "label": "group_1-cop", "url": ".../new_sub_game",
 "attempt": 0, "payload": {"session_id": "sg-0", "grid": [5,5], "max_moves": 25,
                           "your_role": "cop", "your_pos": [3,3]}}

{"ts": "...", "direction": "response", "label": "group_1-cop", "url": ".../new_sub_game",
 "attempt": 0, "response": {"ok": true}, "latency_ms": 23.7}

{"ts": "...", "direction": "request",  "label": "group_2-thief", "url": ".../request_move",
 "attempt": 0, "payload": {"session_id": "sg-0", "tick": 3, "your_pos": [1,2],
                           "opponent_pos": null, "barriers": [[1,1]], "barriers_left": 4}}

{"ts": "...", "direction": "result", "sub_game": {"id": 1, "seed": 101, "session_id": "sg-0",
 "start": "...", "end": "...", "moves": 5, "winner": "cop", "scores": {"cop": 20, "thief": 5},
 "cop_group": "...", "thief_group": "..."}}
```

Notes so you can re-derive without guessing:

- `label` is `<group>-<role>`, so every line tells you whose agent it was and in which role.
- `attempt` is the P8 retry counter: `0` is the first send, `1` an identical re-POST. A
  duplicated `(session_id, tick)` with `attempt: 1` is a retry, not a second move.
- `session_id` `"sg-0".."sg-5"` maps to `sub_games` ids `1..6`.
- The `result` event carries the **seed actually played**, which is how you detect a P7
  spare substitution after voids.
- `payload` is exactly the body we POSTed, so every tick's ground truth (`your_pos`,
  `barriers`, `barriers_left`) is in the log for both roles.

You get each sub-game's result summary as it completes, and the complete file after
sub-game 6.

## Hosting caveat — ACCEPTED, with one bound

- **Health before play:** our runner already health-checks every endpoint before the match
  starts and refuses to begin if any is down. The six sub-games then run back-to-back in
  a few minutes, well inside your 15-minute window, so a mid-match sleep should not arise.
- **Unknown session ⇒ re-hello, not void:** agreed, and this amends P8. On an
  unknown-session error we will re-POST `/new_sub_game` for that `session_id` and continue
  from tick 0 of that sub-game, rather than counting a fault. Bounded at **two** re-hellos
  per sub-game so a genuinely dead endpoint still terminates instead of looping; after
  that the normal P8 void path applies. Note our reference skeleton returns **409** for
  this case and yours returns 400 — we treat any 4xx whose body mentions an unknown
  session as a re-hello trigger, so either is fine.
- Thanks for disabling auto-deploy for the window; we will do the same on our side.

## Protocol-conformance sub-game — ACCEPTED

Unscored, driven by a scripted test agent on our side so no policy information moves in
either direction. We propose running it immediately before the real match, in the same
session, so a pass means the live path is warm and verified.

---

## What we need from you — in order

**1. A working bearer token (BLOCKING — nothing else can start).**

The value you sent returns `401` from your own deployment. Before resending, verify the
token you are about to send against the **deployed** service (not a local run) — that
distinction is almost certainly where this went wrong:

```powershell
$t = "<the token you are about to send>"
$b = "https://marl-bonus-adapter.onrender.com"

curl.exe -s -o NUL -w "%{http_code}`n" -X POST -H "Authorization: Bearer $t" -H "Content-Type: application/json" -d '{\"session_id\":\"tokencheck\",\"grid\":[5,5],\"your_role\":\"thief\",\"your_pos\":[1,1],\"max_moves\":25}' "$b/new_sub_game"
```

`200` means send it. `401` means the secret in your hosting dashboard differs from `$t` —
fix that first (and remember a dashboard change needs a redeploy/restart to take effect).
If you would rather not paste it again, send us the first 12 characters of its `sha256`
and we will tell you whether it matches what we already hold (`d06ce58286d8`).

**2. Your availability window.** Ours is at the bottom of this message.

**3. "Frozen" on the P7 table** in C2 above — or send your own 6+ seeds and we will
publish the layouts they resolve to, and you freeze those instead.

**4. Your call on C1 (`stay`).** We declined, but the offer stands: say the word and we
enable it for both sides.

**5. Confirm the C3 log schema is enough** to re-derive every tick. If a field is missing
for your verifier, better to say now than after sub-game 6.

**6. Keep auto-deploy disabled** for the window, as you proposed — we are doing the same.

Once 1–3 land we will run the unscored conformance sub-game, then the real match.

Our availability: **we are flexible — you pick the slot.** The match is short: our recorded
cloud run took 4m29s for all six sub-games, and most of that was a single free-tier wake-up.
Add the conformance game and the whole session is ~15 minutes. Your service is always-on and
our referee drives everything, so what we actually need is a window where one human on each
side can answer if something breaks, and then do the byte-compare and send the two emails —
call it 30 minutes, not an evening.

**Propose any slot in the next 48 hours (Asia/Jerusalem) and we will take it.** If you would
rather we name one: **Sun–Thu, 20:00–23:00 Asia/Jerusalem** suits us any day this week.
