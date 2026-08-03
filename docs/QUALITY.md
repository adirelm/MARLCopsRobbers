# Quality model — ISO/IEC 25010 (§13)

How MARL Cops & Robbers addresses each of the eight ISO/IEC 25010 product-quality
characteristics, with a concrete A6 artifact for each (no prose-only claims).

## Functional Suitability
The system trains CTDE QMIX/VDN/IQL pursuit policies and plays a 6-sub-game match over a
dual-MCP contract, emailing the §3.5 report. Evidence: 981 tests (happy + error paths)
at ≥85% coverage; the §3.5 report passes `report.schema.json` + the derived-totals
invariants (`src/reporting/schema.py::validate`); F5 reports the IQL/VDN/QMIX comparison —
VDN the highest mean, IQL the tightest cross-seed spread, QMIX least stable at the 50-round
budget (reported faithfully, not idealized).

## Performance Efficiency
Compute is governed at the single SDK entry: `apply_compute_limits` caps the torch thread
pools from config, so a full training run never grabs all cores (the host stays
responsive). Recurrent replay samples whole episodes; OLoRA trains ~8× fewer parameters
than full fine-tuning. Evidence: `src/utils/compute.py`; `docs/COST_ANALYSIS.md` envelope.

## Compatibility
One canonical MCP tool contract serves localhost AND cloud (no divergent fork, ADR-D5-01);
the GUI spectator reads referee ground truth without ever calling an agent server. Evidence:
`src/mcp/server_builder.py` (shared builder); `test_gui_purity.py` (GUI imports only sdk/gui/pygame).

## Usability
A Pygame god-view spectator with pause / next-sub-game / speed / reset / radius-overlay keys; the
10 Nielsen heuristics are mapped in `docs/UX.md`; §7.3c screenshots at 2×2…5×5. Evidence:
`docs/UX.md`; `results/screenshots/grid_*.png`; `tests/integration/test_gui_render.py`.

## Reliability
The report send is guarded four ways (sha256 sentinel): an identical resend is a no-op, an
`intent` line written BEFORE dialing SMTP blocks retries after a mid-send crash until the
operator verifies the inbox and clears it, a CHANGED report is refused unless
`RESEND_APPROVED=1`, and the check→intent window runs under a per-sentinel
`<sentinel>.lock` file (`O_CREAT|O_EXCL`) plus the thread lock — so two concurrent
PROCESSES cannot both pass the gate (a stale lockfile blocks fail-closed until the
operator removes it). The precise guarantee is "at most one email per sentinel scope
without explicit operator action", not an unconditional never; the API gatekeeper degrades to a FIFO
queue instead of crashing on overflow; training runs are resumable (`done_runs`). Evidence:
`src/reporting/send.py`; `src/reporting/send_lock.py`;
`tests/integration/test_sentinel_process_lock.py`; `src/api/gatekeeper.py`; `src/results/run_log.py`.

## Security
RS256 bearer-JWT cloud auth with a `jti` revoke deny-list (`RevocableJWTVerifier`) and
per-role audience enforcement; every accepted token MUST carry a numeric `exp` and a
non-empty string `jti` (the verifier rejects otherwise — no permanently-valid or
un-revocable credential slips through the base layer's optional-claims default), and
`request_move` rejects any `global_state` at the protocol edge. Outbound egress is admitted
through the single `ApiGatekeeper` on every channel where its queue semantics fit — the Gmail
report send and every peer-MCP tool call (`peer_mcp` channel, asserted by ROUTING in
`test_egress_via_gatekeeper.py`, not merely by import location); a DEFERRED admission
(backpressure) hard-faults with a `RuntimeError` rather than letting the real call fire
ungoverned. The one documented exception: the §9 wire-match client (`src/mcp/wire_client.py`)
bypasses the gatekeeper BY DESIGN — its DEFERRED/queue semantics are incompatible with the
match's synchronous 10-second move deadline (a deferred move IS a technical fault). Logs are
redacted; PII lives only in git-ignored files.

**Threat model — what the auth layer does NOT defend.** This is a *cooperative, graded* match
between two KNOWN groups over authenticated tokens, every tool call logged with its session
trace and §9.3-verified post-hoc. The auth layer proves token authenticity, audience binding,
expiry and revocation — it does NOT enforce application-session integrity: a valid token may
pick any caller-chosen `session_id` (so `new_sub_game` can reset a session) and, per the
brief's OWN mutual-position protocol, the requester ASSERTS its own `requester_pos` to
`reveal_location` (so a token holder could probe cells); input sizes are likewise unbounded
(a large `session_id`, `NaN` scalars, or nested `image` payloads are accepted). These are
protocol-inherent for this threat model, not auth bugs — session rebinding and reveal-probe
prevention would require redesigning the shared session model the brief specifies. The backstop
is the §9.3 post-hoc log audit: every peer call (including each `reveal_location`) is emitted by
`AgentClient._call` with its `session_id` trace, so out-of-turn resets or reveal-probing are
detectable after the fact. Evidence: `src/mcp/jwt_auth.py`; `src/mcp/schemas.py`;
`src/mcp/clients.py`; `.env-example`.

## Maintainability
Every `.py` ≤150 LOC; all business logic behind the single `MarlSDK` seam; OOP with no
duplication; ruff(+D) clean; ADRs record every decision. Evidence:
`scripts/check_file_sizes.py`; the ADR index in `docs/PLAN.md` §6; the per-section commit trail.
**Extension points (V3 §12):** the `Mixer` ABC (QMIX/VDN swap; a new mixer is one subclass),
the learner branch seam (IQL vs CTDE, ADR-0008), the `EmailSender` Protocol (App-Password ↔ OAuth
drop-in), the `StateClient` trio (in-proc / replay / HTTP spectator feeds), and the
`build_verifier` auth seam (static bearer ↔ RS256 JWT) — each is one interface, already exercised
by two implementations or a test double.

## Portability
`uv` + `pyproject.toml` (no `pip`/`requirements.txt`); pure-Python + torch; headless via
`SDL_VIDEODRIVER=dummy`; a Render deploy blueprint (`deploy/render.yaml`).
Evidence: `pyproject.toml` + `uv.lock`; `deploy/render.yaml`; `deploy/runbook.md`.
