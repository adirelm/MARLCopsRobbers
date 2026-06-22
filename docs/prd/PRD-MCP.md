# PRD — Dual-MCP topology, cloud & GUI spectator

> **Per-area PRD (V3 §2 "dedicated PRD per mechanism").** Decomposes the area-relevant requirements of
> the human-approved master `docs/PRD.md` (`FR-MCP-*`, `FR-CLOUD-*`, `FR-GUI-*`) — no new requirements.
> Per `CLAUDE.md` §1.4 requirements are human-decided; sign-off is **inherited** from the master PRD.
> Canonical text + ACs: `docs/PRD.md`.

## Scope

The runtime topology: **two independent FastMCP agent servers** (Cop, Thief) + a **neutral referee** =
the environment (CTDE global-state holder, not a third player); ONE canonical five-tool contract for
localhost **and** cloud; revocable RS256 bearer-JWT auth; genuine server-to-server location comms
(evidence-only, decoupled from the policy); the Prefect-Horizon/FastMCP-Cloud deploy (built + tested,
not live-run); and a strict read-only Pygame **god-view spectator**.

## Functional requirements (canonical text in `docs/PRD.md`)

| ID | Requirement (essence) |
|---|---|
| **FR-MCP-1/2** | Two independent servers (ports `mcp.cop_port`/`thief_port`, path `/mcp`) + a neutral referee = sole ground-truth holder. |
| **FR-MCP-3 / 3b** | Local-obs-only `request_move`; ONE canonical tool contract (`request_move`/`reveal_location`/`query_opponent`/`new_sub_game`/`send_final_report`); `session_id`-keyed server-side GRU state, reset on `new_sub_game`. |
| **FR-MCP-4 / 5** | No tool leaks weights/Q/logits/hidden/full board; `reveal_location` radius-gated; real cop↔thief HTTP queries each tick (audit-only). |
| **FR-MCP-6 / 9** | Revocable bearer token, invalid/revoked → 401; technical-loss replay to reach exactly 6 valid sub-games. |
| **FR-MCP-7 / 8** | Full localhost run (two ports) producing 6 valid sub-games; the **Cop only** sends ONE report. |
| **FR-MCP-10** | Zero hardcoded ports/URLs/tokens (config/`.env`; local vs cloud by env). |
| **FR-CLOUD-1..3** | Public HTTPS deploy (Horizon primary, Render fallback); cloud-to-cloud comms with one shared `trace_id`; inference-only (actor weights only). |
| **FR-CLOUD-4 / 5** | Revocable (jti deny-list) + idempotent `(session_id,tick)` + resilient client (timeout/retry/backoff/prewarm); cloud match emits the §3.5 JSON but does NOT send. |
| **FR-GUI-1..7** | Mandatory real-time render; **spectator purity** (zero effect on agent obs; reads referee truth only); deterministic 2×2–5×5 screenshots; HUD/scoreboard; dynamic grid; Stage-2 SSE spectator path; rendering literals local in `palette.py`. |

## Key architect decisions (human-decided, §1.4)

- **Referee-mediated 3-process topology**, server-side per-session GRU state (single worker) — ADR-0011.
- **Cloud = Prefect Horizon / FastMCP Cloud**, app-level **revocable RS256 JWT** (not Horizon OAuth),
  jti deny-list + key rotation — ADR-0012.
- **GUI = Pygame god-view spectator**, dual read path, render literals local — ADR-0014.
- **§5 ApiGatekeeper** (peer_mcp/gmail/prefect channels) governs egress — ADR-0006/0009.

## Acceptance & evidence

ACs are the per-`FR-*` AC lines in `docs/PRD.md`. Evidence: `tests/unit/test_mcp_*`, `test_mcp_clients`,
`tests/integration/test_match.py` (full local match, 401, decoupling), `tests/architecture/`
(no global_state in MCP schemas; GUI purity), and the F4 comms proof (`scripts/capture_comms.py`).
