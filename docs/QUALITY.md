# Quality model — ISO/IEC 25010 (§13)

How MARL Cops & Robbers addresses each of the eight ISO/IEC 25010 product-quality
characteristics, with a concrete A6 artifact for each (no prose-only claims).

## Functional Suitability
The system trains CTDE QMIX/VDN/IQL pursuit policies and plays a 6-sub-game match over a
dual-MCP contract, emailing the §3.5 report. Evidence: 535 tests (happy + error paths)
at ≥85% coverage; the §3.5 report passes `report.schema.json` + the derived-totals
invariants (`src/reporting/schema.py::validate`); F5 reports the IQL/VDN/QMIX comparison —
VDN most consistent, QMIX least stable at the 50-round budget (reported faithfully, not idealized).

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
The report send is idempotent (sha256 sentinel written inside the egress thunk, so a
deferred-then-drained send can never double-email); the API gatekeeper degrades to a FIFO
queue instead of crashing on overflow; training runs are resumable (`done_runs`). Evidence:
`src/reporting/send.py`; `src/api/gatekeeper.py`; `src/results/run_log.py`.

## Security
RS256 bearer-JWT cloud auth with a `jti` revoke deny-list (`RevocableJWTVerifier`),
per-role audience enforcement, and `request_move` rejecting any `global_state` at the
protocol edge; all egress is gatekept and logs are redacted; PII lives only in git-ignored
files. Evidence: `src/mcp/jwt_auth.py` (codex-validated); `src/mcp/schemas.py`; `.env-example`.

## Maintainability
Every `.py` ≤150 LOC; all business logic behind the single `MarlSDK` seam; OOP with no
duplication; ruff(+D) clean; ADRs record every decision. Evidence:
`scripts/check_file_sizes.py`; the ADR index in `docs/PLAN.md` §6; the per-section commit trail.

## Portability
`uv` + `pyproject.toml` (no `pip`/`requirements.txt`); pure-Python + torch; headless via
`SDL_VIDEODRIVER=dummy`; a Prefect Horizon deploy with a Render fallback blueprint.
Evidence: `pyproject.toml` + `uv.lock`; `deploy/render.yaml`; `deploy/runbook.md`.
