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

**P7 frozen list — 5×5, one cop vs one thief. Cells are `[row, col]`, 0-indexed, origin top-left.**

| pair | sub-games | seed | cop start | thief start | manhattan |
|---|---|---|---|---|---|
| 1 | 1 and 4 | `101` | `[3, 3]` | `[1, 1]` | 4 |
| 2 | 2 and 5 | `202` | `[4, 4]` | `[2, 2]` | 4 |
| 3 | 3 and 6 | `303` | `[4, 3]` | `[1, 1]` | 5 |
| spare 1 | — | `404` | `[0, 0]` | `[3, 1]` | 4 |
| spare 2 | — | `505` | `[2, 0]` | `[4, 2]` | 4 |
| spare 3 | — | `606` | `[4, 3]` | `[0, 1]` | 6 |

Read it this way: in sub-game `k` the group playing **cop** starts at the cop cell and the
group playing **thief** at the thief cell; in the mirror `k+3` the same two cells are used
with the roles swapped. That is what makes the pair fair. Every layout satisfies P6
(`manhattan > 2`, so neither side starts inside the other's radius-2 view) — check the
column yourself rather than taking it from us.

Verify each `new_sub_game` payload against this table as it arrives: the `your_pos` we
send you must equal the cell for your role in that sub-game. If it ever does not, stop the
match and tell us before anything is byte-compared.

**Reply "frozen" on this table and it is binding.** If you would rather we generate a
different list (for instance one you pick the seeds for), say so now — after we start,
P7's void amendment is the only thing that may change a layout.

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

## What we need from you to lock this in

1. Bearer token (private channel).
2. Availability window.
3. "Frozen" on the P7 table above — or your own seed list if you would rather choose it.
4. Whether you want `stay` after all (C1).

Our availability: **<<< FILL IN: your window >>>**
