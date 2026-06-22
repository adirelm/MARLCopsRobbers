# PRD — §3.5 match report, scoring & Gmail egress

> **Per-area PRD (V3 §2 "dedicated PRD per mechanism").** Decomposes the area-relevant requirements of
> the human-approved master `docs/PRD.md` (`FR-RPT-*`, `FR-API-*`) — no new requirements/scope/KPIs.
> Per `CLAUDE.md` §1.4 requirements are human-decided; sign-off is **inherited** from the master PRD.
> Canonical text + ACs: `docs/PRD.md`.

## Scope

The end-of-match **§3.5 report**: a structured JSON body assembled by the referee/Cop after exactly 6
valid sub-games, validated against `docs/schema/report.schema.json`, scored by the **§3.4 scoreboard**
(20/10/5/5 — separate from the RL reward), redacted of PII, rate-limited through the `ApiGatekeeper`,
and emailed **once** by the Cop to `gmail.to`. PII (real names/ids) lives only in the git-ignored repo-root
`players.local.yaml`, injected at send time; only a role-only redacted copy is committed.

## Functional requirements (canonical text in `docs/PRD.md`)

| ID | Requirement (essence) |
|---|---|
| **FR-RPT-1** | After the 6th valid sub-game the **Cop sends exactly ONE** email (idempotent; sha256 sentinel inside the egress thunk). |
| **FR-RPT-2** | Body conforms to `{group_name, students[]{role,full_name,id}, github_repo, timezone, sub_games[6]{id,start,end,moves,winner,scores}, totals}` (schema `minItems:1` for SOLO); `validate()` rejects bad winner, wrong game count, `end<start`, `totals≠Σ`. |
| **FR-RPT-3** | Scores **derived** from `winner` + `game.scoring` (the §3.4 REPORT scoreboard) — never the RL `reward.*`; callers cannot supply scores. |
| **FR-API-1** | All Gmail/peer-MCP/Prefect egress routes through `ApiGatekeeper` (per-channel token-bucket from `config/rate_limits.json`; FIFO overflow; calls logged). |

## Key architect decisions (human-decided, §1.4)

- **Gmail = smtplib + STARTTLS + App-Password** behind an `EmailSender` Protocol (OAuth drop-in fallback);
  **PII only in git-ignored `players.local.yaml`**, role-only redacted JSON committed — ADR-0013.
- **§5 governance flips N/A→REQUIRED** for A6 (real runtime egress) — ADR-0009; the gatekeeper is the
  single egress seam.
- The schema (`docs/schema/report.schema.json`) is a **repo constant** loaded by `src/reporting/schema.py`.

## Hard gate (operational, user-decided)

The actual lecturer email send is a **hard no-send-until-explicit-go** gate: the report is assembled,
validated, and dry-run/DRY-RUN exercised in tests, but is **never** auto-sent — it awaits an explicit
human "send" + `.env` Gmail credentials. This is an operational control, not a code defect.

## Acceptance & evidence

ACs are the per-`FR-RPT-*` AC lines in `docs/PRD.md`. Evidence: `tests/unit/test_reporting.py`
(schema + totals invariants), `tests/unit/test_report_send.py` (single/idempotent send),
`tests/integration/test_match.py` (one dry-run send in a full match), `src/api/gatekeeper.py` tests.
