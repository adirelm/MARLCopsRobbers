# biu-azri — final step: sending your §9.4 bonus email

Ours is sent. `adrl-001 60 — biu-azri 40`, claims 10 / 7, `mutual_agreement: true`,
subject `[MARL Bonus Game] adrl-001 vs biu-azri – Final Report`, to `rmisegal+marl@gmail.com`.

Per §9.3 the bonus is calculated **only if BOTH groups send** and both agree. One of us
having sent is not enough — yours still has to go out.

---

## ⚠️ First: you need our student block, and we never sent it

`§9.4` requires **both** `students_group_1` (ours) and `students_group_2` (yours) in
**each** group's email. We asked for yours in the brief and you sent it — but we never
sent you ours, and the byte-compare would not have caught it: the compared digest covers
only `sub_games` / `totals_by_group` / `bonus_claim`, deliberately, because those three
carry no identity data.

So if you build `students_group_1` from what you have, it is empty or a placeholder, and
your report is incomplete. **Our block is in the private message alongside this file** —
copy it in verbatim before you send. Our mistake; better caught now than by the grader.

While you are there, also check `github_repo_group_1` is our repo URL:
`https://github.com/adirelm/MARLCopsRobbers`.

## What we sent, as a reference

This is our actual sent body with **both** groups' identity fields masked — the shape,
key names and every agreed value are exactly what went out. Replace the four masked
fields with the real values (yours you already have; ours are in the private message).

```json
{
  "report_type": "bonus_game",
  "groups": {
    "group_1": "adrl-001",
    "group_2": "biu-azri"
  },
  "github_repo_group_1": "<redacted — real URL only in the git-ignored copy + the email>",
  "github_repo_group_2": "<redacted — real URL only in the git-ignored copy + the email>",
  "timezone": "Asia/Jerusalem",
  "students_group_1": [
    {
      "role": "A"
    }
  ],
  "students_group_2": [
    {
      "role": "A"
    },
    {
      "role": "B"
    }
  ],
  "sub_games": [
    {
      "id": 1,
      "start": "2026-08-04T00:21:24.647+03:00",
      "end": "2026-08-04T00:21:25.574+03:00",
      "moves": 6,
      "winner": "cop",
      "scores": {
        "cop": 20,
        "thief": 5
      },
      "cop_group": "adrl-001",
      "thief_group": "biu-azri"
    },
    {
      "id": 2,
      "start": "2026-08-04T00:21:25.574+03:00",
      "end": "2026-08-04T00:21:28.822+03:00",
      "moves": 25,
      "winner": "thief",
      "scores": {
        "cop": 5,
        "thief": 10
      },
      "cop_group": "adrl-001",
      "thief_group": "biu-azri"
    },
    {
      "id": 3,
      "start": "2026-08-04T00:21:28.822+03:00",
      "end": "2026-08-04T00:21:31.874+03:00",
      "moves": 25,
      "winner": "thief",
      "scores": {
        "cop": 5,
        "thief": 10
      },
      "cop_group": "adrl-001",
      "thief_group": "biu-azri"
    },
    {
      "id": 4,
      "start": "2026-08-04T00:21:31.874+03:00",
      "end": "2026-08-04T00:21:35.065+03:00",
      "moves": 25,
      "winner": "thief",
      "scores": {
        "cop": 5,
        "thief": 10
      },
      "cop_group": "biu-azri",
      "thief_group": "adrl-001"
    },
    {
      "id": 5,
      "start": "2026-08-04T00:21:35.066+03:00",
      "end": "2026-08-04T00:21:38.120+03:00",
      "moves": 25,
      "winner": "thief",
      "scores": {
        "cop": 5,
        "thief": 10
      },
      "cop_group": "biu-azri",
      "thief_group": "adrl-001"
    },
    {
      "id": 6,
      "start": "2026-08-04T00:21:38.121+03:00",
      "end": "2026-08-04T00:21:41.176+03:00",
      "moves": 25,
      "winner": "thief",
      "scores": {
        "cop": 5,
        "thief": 10
      },
      "cop_group": "biu-azri",
      "thief_group": "adrl-001"
    }
  ],
  "totals_by_group": {
    "adrl-001": 60,
    "biu-azri": 40
  },
  "bonus_claim": {
    "adrl-001": 10,
    "biu-azri": 7
  },
  "mutual_agreement": true
}
```

Notes on the shape, since a mismatch here is what §9.3 punishes:

- `id` is a **string**, not a number.
- Every `sub_games` entry keeps the §3.5 keys (`id`, `start`, `end`, `moves`, `winner`,
  `scores`) **and** adds `cop_group` / `thief_group`.
- `start` / `end` are ISO-8601 Asia/Jerusalem with millisecond precision, taken
  **verbatim from the referee log** — do not re-stamp them from your own clock, or the
  two reports will disagree on fields the lecturer can compare.
- `totals_by_group` and `bonus_claim` are keyed by the **group-name strings**.
- `mutual_agreement` must be exactly `true` (boolean, not the string `"true"`).

## Your checklist

1. Paste our `students_group_1` block from the private message.
2. Confirm `github_repo_group_1` and `github_repo_group_2` are both present and correct.
3. Confirm `mutual_agreement: true`.
4. Subject exactly: `[MARL Bonus Game] adrl-001 vs biu-azri – Final Report`
5. Send **one** email to `rmisegal+marl@gmail.com`. §9.3 allows exactly one valid bonus
   email per group.
6. **Document the match in your README** — §9.3 requires it explicitly: our group name
   (`adrl-001`), the final score (`adrl-001 60 — biu-azri 40`), your bonus claim (**7**),
   and screenshots of the match. Ours is in README §9a if you want a model.
7. Tell us when it is out, so we both know the §9.3 condition is met.

## After that

Nothing further. You may rotate the adapter token whenever you like — we hold it only in
a git-ignored env file and will delete it once you confirm your email is sent.

Good match, and genuinely well played on the process: the C2 objection about seeds was
the single most valuable thing either side raised, and your independent 274-move replay is
what makes the agreement mean something.
