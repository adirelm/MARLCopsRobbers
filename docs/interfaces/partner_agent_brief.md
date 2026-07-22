# §9 Bonus Match — Partner-Group Agent Brief (adrl-001 ↔ your group)

**Audience:** the partner group's coding agent (Claude Code or equivalent). This file is
the complete, self-contained specification of your side of the match — you need nothing
from our repository, and every rule, payload, and process step you must implement is
defined below. Your job has five parts: (1) confirm the rules agreement, (2) implement
and host a small HTTP adapter that exposes YOUR two policies (cop + thief), (3) send us
your identity + endpoint details, (4) play the match, (5) verify results and send your
group's §9.4 bonus email.

This is the ex06 §9 inter-group bonus of the Bar-Ilan "MARL Cops & Robbers" assignment:
6 sub-games between two groups, roles alternating. Bonus scoring (added to the FINAL
PROJECT grade, not to ex06): winner +10, loser +7, tie +5 each — and it counts **only if
BOTH groups send a valid bonus email AND both set `mutual_agreement: true`; otherwise
both groups get 0** (§9.3, verbatim rule). Each group may send exactly ONE valid bonus
email. The process below is designed so agreement is mechanical, not negotiable after
the fact.

---

## 1a. Rules fixed by the assignment (ex06 §3 / §9 — not negotiable)

| # | Rule | Value |
|---|---|---|
| 1 | Board | 5×5 grid, one cop vs one thief |
| 2 | Sub-game length | ≤ 25 moves; thief wins on timeout, cop wins on capture |
| 3 | Match | 6 sub-games; **sub-games 1–3: group_1 plays cop; 4–6: roles swap** (§9.1) |
| 4 | Scoring per sub-game (§3.4 Table 1) | capture → cop 20 / thief 5; timeout → thief 10 / cop 5 |
| 5 | Barriers (§3.3) | cop-only, ≤ **5** per sub-game; `place_barrier` consumes the move and turns the cop's current cell into a barrier |
| 6 | Faults (§3.7) | a technically-broken sub-game does not count — it is VOID and replayed |
| 7 | Report + emails (§9.3–9.4) | both groups email the lecturer one §9.4 JSON each; README documentation + screenshots required (section 5 below) |

## 1b. Match conventions proposed by adrl-001 (become binding once you reply "agreed")

The assignment fixes the game above but leaves resolution details open. These are our
proposed conventions — confirm them, or counter-propose BEFORE the seeds are frozen:

| # | Convention | Value |
|---|---|---|
| P1 | Move resolution | **SIMULTANEOUS** — both actions are resolved from the same pre-move state |
| P2 | Capture test | any cop on the thief's cell after resolution, **OR a one-tick cop↔thief cell swap** (blocks the pass-through exploit). Capture is checked **before** the move-25 timeout, so a final-move capture wins |
| P3 | Illegal move | moving off-board or into a barrier resolves to **stay** (not a fault). A cop `place_barrier` with `barriers_left: 0`, or while standing on a cell that is already a barrier, likewise resolves to **stay** — not a fault, and no budget is consumed |
| P4 | Barrier semantics | a barrier blocks BOTH sides from *entering*; the agent standing on it may still leave |
| P5 | Observability | partial, **Manhattan radius 2**: you receive the opponent's position only when `manhattan(you, opponent) ≤ 2`, else `null`; barrier cells are reported only within radius 2 of your position |
| P6 | Start positions | seeded random with `manhattan(cop, thief) > 2` (spawned outside view range) |
| P7 | Seed schedule + void replay | we jointly agree an ORDERED list of **6+ seeds** `s1..sN` in writing before the match. Sub-game `k` and its mirror `k+3` both use `s_k` (k = 1..3), so both groups play cop and thief from identical layouts. **Voids:** a technically-voided sub-game (P8) is replayed immediately — SAME sub-game, SAME `session_id`, SAME seed — a transient fault never changes the layout either group prepared for. Only after **3 consecutive voids of the same sub-game** does the next unused spare seed (`s4`, `s5`, …) permanently replace `s_k` **for the whole pair k and k+3**; if the pair's other game already completed on the replaced seed, it is re-queued and replayed first, so the mirror pair always shares one seed. If every spare is exhausted, the match is called off and rescheduled (no forfeit claims) |
| P8 | Fault definition | per-move timeout **10 s**, one identical re-POST, then void (§3.7 replay); malformed response (unknown action string, thief sending `place_barrier`) → one retry, then void. Retries are per fault layer (a malformed reply retried once may itself get one transport re-POST), so a single tick sees at most a handful of identical POSTs — idempotency makes them harmless; persistent no-show → match called off, no forfeit claims |

Coordinate convention (used everywhere below): cells are `[row, col]`, 0-indexed,
origin **top-left**. `up = row−1, down = row+1, left = col−1, right = col+1`.

---

## 2. Topology: our referee, your two endpoints

Our group hosts the referee. To keep this implementation-neutral, agents talk a
**minimal wire format with raw positions** — each side does its own encoding and policy
internally. You expose ONE small HTTP service per role, or one service for both roles —
your choice (there is no role field in `request_move`; your service learns its role from
`new_sub_game.your_role`, keyed by `session_id`).

**Fairness guarantees you get:** the referee logs every request/response with
timestamps; after each sub-game you get its result summary (winner / moves / scores),
and after sub-game 6 you get the COMPLETE per-request log of all six games, so you can
independently re-derive every transition from the agreed seeds before agreeing to
anything. On request, we will also play an unscored **protocol-conformance** sub-game
against your runner before the match (driven by a scripted test agent, so no match
information is revealed in either direction).

### Endpoints you must implement (per agent)

**`POST /new_sub_game`** — called once at each sub-game start:
```json
{"session_id": "sg-0", "grid": [5, 5], "your_role": "cop",
 "your_pos": [2, 0], "max_moves": 25}
```
→ respond `{"ok": true}`. Reset ALL per-session state here (memory, RNN state, and your
idempotency cache for that `session_id`) — a voided sub-game is replayed by calling
`new_sub_game` again with the SAME `session_id` and, per P7, the SAME seed (a spare
seed appears only after 3 consecutive voids of that sub-game).

**`POST /request_move`** — called once per tick:
```json
{"session_id": "sg-0", "tick": 3, "your_pos": [2, 1],
 "opponent_pos": [0, 3],
 "barriers": [[1, 1]], "barriers_left": 4}
```
→ respond `{"action": "up"}` where `action` ∈ `"up" | "down" | "left" | "right" | "place_barrier"`.

- Ticks are 0-indexed: `tick` runs `0..24` within a sub-game.
- `session_id` values are `"sg-0".."sg-5"`, mapping to report `sub_games` ids `1..6`.
- `opponent_pos` is `null` when the opponent is outside your radius-2 view (this always
  holds at tick 0, because spawns satisfy P6).
- `barriers` lists only barrier cells within radius 2 of `your_pos`.
- `barriers_left` is the cop's remaining barrier budget (sent to both roles).
- `place_barrier` is **cop-only** (a thief sending it is a fault per P8).
- **Idempotency:** if we re-POST the same `(session_id, tick)` (network retry), return
  the same action you returned before.
- Respond within **10 seconds** (one retry, then the sub-game is void per P8).

**`GET /health`** → `{"status": "ok"}` — we call it before the match starts. `/health`
is unauthenticated; both POST endpoints require the bearer token.

### Auth

Per §5.3 each POST endpoint is protected by a bearer token. Choose any token(s), send
them to us privately; we send `Authorization: Bearer <token>` on every call.

### Reference skeleton (Python/FastAPI — adapt or reimplement in any stack)

```python
from fastapi import FastAPI, Header, HTTPException

app, SESSIONS, TOKEN = FastAPI(), {}, "choose-a-token"


def _auth(authorization: str) -> None:
    if authorization != f"Bearer {TOKEN}":
        raise HTTPException(401)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/new_sub_game")
def new_sub_game(req: dict, authorization: str = Header("", alias="Authorization")):
    _auth(authorization)
    SESSIONS[req["session_id"]] = {"role": req["your_role"], "memory": None, "answers": {}}
    return {"ok": True}


@app.post("/request_move")
def request_move(req: dict, authorization: str = Header("", alias="Authorization")):
    _auth(authorization)
    session = SESSIONS[req["session_id"]]
    if req["tick"] in session["answers"]:          # idempotent re-POST
        return session["answers"][req["tick"]]
    session["answers"][req["tick"]] = {"action": my_policy(session, req)}
    return session["answers"][req["tick"]]
```

(`my_policy(session, req) -> str` is yours: read the payload, update `session["memory"]`,
return an action string.) Host anywhere reachable over HTTPS/HTTP: a cloud service, or
local + a tunnel (e.g. `ngrok http 8080`), or we meet on one network. Uptime is only
needed for the match window.

### Self-test before you send us the URLs

```bash
curl -s https://<your-host>/health
# -> {"status":"ok"}
curl -s -X POST -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
  -d '{"session_id":"selftest","grid":[5,5],"your_role":"thief","your_pos":[4,4],"max_moves":25}' \
  https://<your-host>/new_sub_game
# -> {"ok":true}
curl -s -X POST -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
  -d '{"session_id":"selftest","tick":0,"your_pos":[4,4],"opponent_pos":null,"barriers":[],"barriers_left":5}' \
  https://<your-host>/request_move
# -> {"action":"up"}  (any legal move string; repeat the call — the SAME answer must repeat)
# and a wrong token must 401:
curl -s -o /dev/null -w "%{http_code}" -X POST -H "Authorization: Bearer wrong" \
  -H "Content-Type: application/json" -d '{}' https://<your-host>/new_sub_game
# -> 401
```

---

## 3. What to send back to adrl-001 (5 items)

1. **Group code** — the exact 8-char string (ours: `adrl-001`).
2. **Repo URL** — needed only for the final §9.4 JSON; fine to share at report time.
3. **Students block** — `role / full_name / id` per member (`id` is a digit STRING, as
   in the assignment's example). Privately; PII goes only in the emails + each group's
   git-ignored local files, never in tracked content.
4. **Written "agreed"** on section 1b (or specific counter-proposals).
5. **Two endpoint base URLs + bearer token(s)**, after the curl self-tests pass —
   plus your availability window for the match.

---

## 4. Match day flow

1. Joint ordered seed list (6+ integers, in writing) → frozen (P7).
2. `GET /health` both sides → referee plays sub-games 1–3 (**group_1 = adrl-001 as cop**
   unless we agree otherwise), then 4–6 with roles swapped, same seeds per P7.
3. After each sub-game: result summary (winner / moves / scores). After sub-game 6: the
   complete per-request log of all six games.
4. A faulted sub-game is void → replayed per P7/P8.

## 5. After the match — the §9.4 report (both groups must do this)

Each group independently builds its own JSON report with (exact key names, matching the
assignment's §9.4 example):

- `report_type: "bonus_game"`
- `groups: {group_1, group_2}` (the two group-name strings)
- `github_repo_group_1`, `github_repo_group_2`
- `students_group_1`, `students_group_2` — arrays of `{role, full_name, id}` (id = string)
- `timezone: "Asia/Jerusalem"`
- `sub_games[6]`: `{id: 1..6, start, end}` (ISO-8601 with offset), `moves`,
  `winner: "cop"|"thief"`, `scores: {cop, thief}` (Table 1), and `cop_group` /
  `thief_group` (the group-name strings — group_1 is cop in 1–3, group_2 in 4–6)
- `totals_by_group: {"<group_1 name>": n, "<group_2 name>": n}` — each group's points
  summed across BOTH its roles (in the assignment's own example: 45+15=60 vs 20+60=80)
- `bonus_claim: {"<group_1 name>": 10|7|5, "<group_2 name>": 10|7|5}` — winner 10 /
  loser 7 / tie 5+5 (§9.2)
- `mutual_agreement: true`

**Canonical values for the byte-compare:** take `start` / `end` / `moves` for each
sub-game VERBATIM from the shared referee log — timestamps are ISO-8601 Asia/Jerusalem
with millisecond precision (e.g. `2026-07-22T18:00:05.123+03:00`). Two independently
recorded clocks can never byte-match, so the referee log is the single source of truth
for those strings.

Then: **byte-compare** the two drafts' `sub_games` / `totals_by_group` / `bonus_claim`
with us → only when identical do both groups set `mutual_agreement: true` → **each group
sends ONE email** to the course recipient (`rmisegal+marl@gmail.com`) with subject:

```
[MARL Bonus Game] <group_1> vs <group_2> – Final Report
```

and the JSON as the body. Finally, per §9.3, each group documents in its own README: the
opponent group's name, the final score, its bonus claim, and screenshots of the match.

---

## 6. Contact

Reply to the human who sent you this file with items 1–5 from section 3. Questions about
any rule or payload field are welcome BEFORE the match — after the seeds are frozen, the
spec above is the contract.
