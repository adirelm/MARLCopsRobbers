# adrl-001 → biu-azri — match played, here is the log

Played 2026-08-04 00:21 Asia/Jerusalem against your live adapter, on your frozen seed list.
Attached: **`wire_log_20260804T002124.jsonl`** — the complete per-request log of all six
sub-games, exactly as promised in C3.

Conformance first, as agreed: one unscored sub-game on seed **424242** — deliberately
outside the frozen list so no real layout was exposed to either side. 52 requests, 52
responses, zero retries, zero errors.

## Result

**adrl-001 60 — biu-azri 40.**

| sub-game | cop | thief | moves | winner | scores (cop/thief) |
|---|---|---|---|---|---|
| 1 | adrl-001 | biu-azri | 6 | cop | 20 / 5 |
| 2 | adrl-001 | biu-azri | 25 | thief | 5 / 10 |
| 3 | adrl-001 | biu-azri | 25 | thief | 5 / 10 |
| 4 | biu-azri | adrl-001 | 25 | thief | 5 / 10 |
| 5 | biu-azri | adrl-001 | 25 | thief | 5 / 10 |
| 6 | biu-azri | adrl-001 | 25 | thief | 5 / 10 |

`bonus_claim`: adrl-001 → 10, biu-azri → 7.

## Run health, so you can judge the log's integrity

- **274 requests, 0 P8 retries, 0 error responses, 0 voids, 0 re-hellos.** Nothing was
  replayed or substituted; every sub-game is its first and only attempt.
- Your latency: min 93 ms, median 107 ms, max 215 ms — against the 10 s per-move budget.
- Every sub-game's seed AND both start cells match the frozen table. Mirror pairs 1&4,
  2&5, 3&6 each share their seed, so both groups played cop from the identical start.

We re-derived every tick from this log on our side before sending it. Please do the same
independently rather than trusting the table above.

## The byte-compare — how we propose to do it

Build your §9.4 JSON from the log **without looking at ours**; that independence is the
whole point. Then compare only the three sections §5 names, canonicalised so formatting
cannot cause a false mismatch:

```python
import hashlib, json
subject = {
    "sub_games": report["sub_games"],
    "totals_by_group": report["totals_by_group"],
    "bonus_claim": report["bonus_claim"],
}
canon = json.dumps(subject, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
print(len(canon), hashlib.sha256(canon.encode()).hexdigest())
```

Ours:

```
length : 1261
sha256 : b15848a23bb0c37d2333d0c695194bdaa8f7d169daa7453af9abca311e81dd58
```

Send yours. Matching hashes ⇒ agreed, and neither of us had to see the other's file first.
If they differ, send the canonical string itself and we will diff it — no student blocks or
repo URLs are in these three sections (the `id` fields are sub-game indices 1–6), so it is
safe to exchange in full.

## Only after the hashes match

Both groups set `mutual_agreement: true` and each sends its **one** email to
`rmisegal+marl@gmail.com` with subject:

```
[MARL Bonus Game] adrl-001 vs biu-azri – Final Report
```

Ours currently reads `mutual_agreement: false` and **nothing has been sent**. We will not
send until you confirm the hash matches. Per §9.3 a disagreement scores 0 for both of us,
so there is no hurry that beats getting this right.

Also still to do on both sides, per §9.3: document the match in each group's README —
opponent name, final score, your bonus claim — and attach match screenshots.

Good match. Your thief took three of three as evader, and sub-game 1 was the only capture
either cop managed.
