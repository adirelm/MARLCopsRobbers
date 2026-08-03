# biu-azri — what to do now (adrl-001 §9 bonus match)

Short version: **one thing is blocking** (the bearer token your deployment rejects), then
four replies, then we play. Everything else in your message is settled — we accepted C2
and C3 and your hosting caveat, and declined C1 with the offer left open. Details are in
our full reply; this file is just the actions.

---

## Step 1 — Fix the token (BLOCKING)

The token you sent returns `401` from your own live service. Your service and auth layer
are fine — a wrong token and a missing header both correctly `401`, and `/health` returns
`200 {"status":"ok","version":"1.00"}`. It is **this specific value** the running
deployment does not accept.

Your error text was `{"error":"bearer token invalid or revoked"}`. That wording suggests
your adapter distinguishes two cases, so check both:

| If it is… | Then |
|---|---|
| **invalid** — the deployed secret differs from the string you sent | Read the value from your hosting dashboard (Render → the service → Environment) and compare it to what you pasted. A dashboard edit needs a **redeploy/restart** before it takes effect — that is the most common cause. |
| **revoked** — the value is in a deny-list your adapter keeps | Remove it from the deny-list, or mint a fresh token and send that instead. |

### Verify BEFORE you resend

Run this against the **deployed URL** (not a local run — that difference is very likely
the whole bug). Paste your token into `$t` and run the block as-is:

```powershell
$t = "PASTE_THE_TOKEN_YOU_ARE_ABOUT_TO_SEND"
$b = "https://marl-bonus-adapter.onrender.com"

function Check($name, $want, $got) {
  if ("$got" -eq "$want") { Write-Host "PASS  $name  ($got)" -ForegroundColor Green }
  else                    { Write-Host "FAIL  $name  (got $got, want $want)" -ForegroundColor Red }
}

$hello = '{\"session_id\":\"tokencheck\",\"grid\":[5,5],\"your_role\":\"thief\",\"your_pos\":[1,1],\"max_moves\":25}'
$move  = '{\"session_id\":\"tokencheck\",\"tick\":0,\"your_pos\":[1,1],\"opponent_pos\":null,\"barriers\":[],\"barriers_left\":5}'

Check "health"        200 (curl.exe -s -o NUL -w "%{http_code}" "$b/health")
Check "wrong token"   401 (curl.exe -s -o NUL -w "%{http_code}" -X POST -H "Authorization: Bearer wrong" -H "Content-Type: application/json" -d $hello "$b/new_sub_game")
Check "no auth"       401 (curl.exe -s -o NUL -w "%{http_code}" -X POST -H "Content-Type: application/json" -d $hello "$b/new_sub_game")
Check "new_sub_game"  200 (curl.exe -s -o NUL -w "%{http_code}" -X POST -H "Authorization: Bearer $t" -H "Content-Type: application/json" -d $hello "$b/new_sub_game")

$a1 = curl.exe -s -X POST -H "Authorization: Bearer $t" -H "Content-Type: application/json" -d $move "$b/request_move"
$a2 = curl.exe -s -X POST -H "Authorization: Bearer $t" -H "Content-Type: application/json" -d $move "$b/request_move"
Write-Host "move -> $a1"
Check "idempotent re-POST" "$a1" "$a2"
```

All six `PASS` ⇒ send the token. Any `FAIL` on the last three ⇒ do not send yet, the
match would fail the same way.

**Don't want to paste the secret again?** Send us the first 12 characters of its
`sha256` instead. The value we currently hold hashes to `d06ce58286d8` — if yours matches
that, the token is right and the problem is the deployment; if it differs, it was a
transcription or rotation issue.

---

## Step 2 — Reply with four things

1. **The working token** (private channel).
2. **Your availability window** — we will pick the overlap with ours.
3. **The seed list.** We would rather **you** pick it: we host the referee, so whoever
   chooses the seeds can preview the layouts first, and that surface points at us. Send any
   6+ integers and we will publish the layouts they resolve to. If you would rather not,
   reply "frozen" on the table in our full reply (seeds `911, 822, 733, 644, 555, 466`).
   Note: an earlier draft listed `101…606` — disregard that, those are our dress-rehearsal
   seeds and reusing them would give us layouts we have already played.
4. **Your call on C1 (`stay`)** — we declined, but if you want it we will enable it for
   **both** sides and re-freeze. Your policies, your call.

Optional but useful: confirm the C3 log schema we published has every field your verifier
needs. Better to find a gap now than after sub-game 6.

---

## Step 3 — Conformance sub-game, then the match

Once steps 1–2 land:

1. We run the **unscored protocol-conformance sub-game** you asked for — driven by a
   scripted test agent on our side, so no policy information moves either way.
2. We play the **6 real sub-games** back-to-back. They take a few minutes total, which
   stays well inside your 15-minute sleep window, and we health-check every endpoint
   before starting and refuse to begin if any is down.
3. After each sub-game you get its result summary; after sub-game 6 you get the complete
   per-request JSONL log.
4. You re-derive it, we both build our §9.4 JSON independently, we **byte-compare**
   `sub_games` / `totals_by_group` / `bonus_claim`, and only when they are identical does
   either group set `mutual_agreement: true` and send its one email.

---

## Two things worth knowing before match day

- **Unknown-session errors will not cost you a void.** We accepted your amendment: on an
  unknown-session response we re-POST `/new_sub_game` for that `session_id` and continue,
  rather than counting a P8 fault. Bounded at **two** re-hellos per sub-game so a genuinely
  dead endpoint still terminates. We key off any 4xx whose body mentions an unknown
  session, so your `400` and our reference skeleton's `409` are both fine.
- **Keep auto-deploy disabled** for the window, as you proposed. We are doing the same, and
  neither side should push to the serving branch once the conformance game passes.

The bonus counts only if **both** groups send a valid email with `mutual_agreement: true`
(§9.3) — a disagreement scores 0 for both of us, which is why we would rather over-verify
now than discover a mismatch after the match.
