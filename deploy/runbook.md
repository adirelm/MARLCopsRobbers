# Cloud deploy runbook (§8 four-stage) — MARL Cops & Robbers

The copy-pasteable, ordered version of the BRIEF §8 appendix. Both cloud servers expose
the **same** canonical tool contract + `/mcp` path as localhost (ADR-D5-01); the only
delta is the auth verifier (static → RS256 JWT).

> **STATUS: DEPLOYED AND LIVE (2026-07-22)** on **Render** —
> `https://adrl-001-cop.onrender.com/mcp` + `https://adrl-001-thief.onrender.com/mcp`
> (recorded in `config.cloud.cop_url/thief_url`). A full 6-sub-game match ran between the
> two cloud servers over the public internet, plus the §5.3 mutual position verification
> and the auth matrix — see README §7.3d (F4c + cloud_auth) and the "What actually
> happened" section at the end of this file.
>
> **Why Render, not FastMCP Cloud / Prefect Horizon:** Horizon's
> FREE tier forces its own platform OAuth *in front of* the app. That breaks §5.3's
> public-access requirement AND collides with our in-app RS256 JWT (a single
> `Authorization` header cannot satisfy two bearer layers); disabling it is a paid
> Developer-plan feature. Render serves the endpoint publicly with OUR token auth as the
> only gate — exactly the §5.3 architecture. See `deploy/render.yaml`.

## Stage 1 — local uv deps (T8.0)

```bash
uv sync --group mcp                         # fastmcp + httpx + pyjwt (base project deps)
uv run python -c 'import fastmcp; print(fastmcp.__version__)'   # expect 3.x
uv run pytest tests/unit/test_jwt_auth.py -q                    # auth proven offline
```

There is **no** `deploy/requirements.txt` (V3 forbids it). If a target demands a pinned
file, generate it at deploy time and never track it: `uv export --no-dev > /tmp/req.txt`.

## Stage 2 — mint an RS256 keypair + set the server env (T8.1)

```bash
# Generate a keypair (private mints client tokens; public verifies on the servers).
# NOTE: fastmcp 3.x wraps private_key in a pydantic SecretStr — unwrap when writing.
# The private key is written 0600 (owner-only) and lives OUTSIDE the synced repo (/tmp,
# never tracked — *.pem is git-ignored); rotate it if it is ever the live cloud key.
uv run python -c "import os; from fastmcp.server.auth.providers.jwt import RSAKeyPair; \
kp=RSAKeyPair.generate(); \
fd=os.open('/tmp/mcp_priv.pem', os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0o600); \
os.write(fd, kp.private_key.get_secret_value().encode()); os.close(fd); \
open('/tmp/mcp_pub.pem','w').write(kp.public_key); print('wrote /tmp/mcp_{priv,pub}.pem (priv 0600)')"
```

Each server (cop / thief) needs, in its cloud env (NEVER tracked — see `.env-example`):

| Var | Meaning |
|---|---|
| `MCP_AUTH_MODE=jwt` | selects the RS256 verifier via the `build_verifier` seam |
| `MCP_PUBLIC_KEY` | the RS256 public-key PEM (verifies bearer tokens) |
| `MODEL_PATH` | path to the actor `.pt` (cop=`n_agents 2`, thief=`1`). The deployed
`deploy/model/bonus_*.pt` are plain QMIX self-play actors; an OLoRA bundle from
`finetune_ctde.py` is drop-in compatible |
| `REVOKED_TOKEN_JTIS` | comma-separated jti deny-list (the revoke demo) |
| ~~`PEER_MCP_URL` / `PEER_MCP_TOKEN`~~ | NOT used by the cloud build — `query_opponent`'s direct peer seam is wired only in the localhost match; the cloud's §5.3 mutual verification goes through `reveal_location` (see `src/mcp/cloud.py`) |

Deploy entrypoints (module-level `mcp`): `src/mcp/cloud_cop.py:mcp` and
`src/mcp/cloud_thief.py:mcp`.

## Stage 3 — deploy to Render + publish URLs (T8.2/T8.3)

Render (chosen over FastMCP Cloud / Prefect Horizon — see "Why Render" above) deploys
from GitHub via a Blueprint; there is no deploy-platform API key.

```bash
# 1. In the Render dashboard: New -> Blueprint -> connect the repo -> path deploy/render.yaml.
#    It provisions BOTH web services (adrl-001-cop, adrl-001-thief). The only dashboard
#    secret is MCP_PUBLIC_KEY (the one-line \n-escaped PEM; the verifier normalizes it);
#    MODEL_PATH + MCP_AUTH_MODE=jwt come from render.yaml.
# 2. Render publishes:
#    cop   -> https://adrl-001-cop.onrender.com/mcp
#    thief -> https://adrl-001-thief.onrender.com/mcp   (already in config.cloud.*)
# 3. Verify health() over the public internet (from a machine OFF the dev host):
uv run python -c "import asyncio; from fastmcp import Client; from fastmcp.client.auth import BearerAuth; \
print(asyncio.run(Client('https://adrl-001-cop.onrender.com/mcp', auth=BearerAuth('<token>')).__aenter__()))"
```

Mint a client token from the private key (the minter). The verifier REQUIRES both a numeric
`exp` and a string `jti` on every accepted token (`RevocableJWTVerifier` — no un-expiring or
un-revocable credential), so mint with both: `expires_in_seconds` sets `exp`, `additional_claims`
carries `jti`. The token is printed to stdout for the demo only — it is a short-lived bearer
secret, so do not paste it into shared logs or tracked files.

```bash
# (private_key is a pydantic SecretStr in fastmcp 3.x — wrap the PEM on load)
uv run python -c "from pydantic import SecretStr; \
from fastmcp.server.auth.providers.jwt import RSAKeyPair; \
kp=RSAKeyPair(private_key=SecretStr(open('/tmp/mcp_priv.pem').read()), public_key=open('/tmp/mcp_pub.pem').read()); \
print(kp.create_token(subject='marl-cop', issuer='adrl-001-mcp-auth', audience='marl-cop', \
scopes=['game:write'], expires_in_seconds=3600, additional_claims={'jti':'cop-001'}))"
```

> **Pre-flight (verified locally, 2026-07-22):** the full cloud stack was smoke-proven on this
> machine BEFORE any account existed — `fastmcp run src/mcp/cloud_cop.py:mcp --transport http`
> with `MCP_PUBLIC_KEY` + `MODEL_PATH=deploy/model/bonus_cop.pt`: valid RS256 token → 200
> (+protocol version), bad token → 401, revoked `jti` (via `REVOKED_TOKEN_JTIS`) → 401,
> missing token → 401.

## Stage 4 — Gmail report (Phase 9)

After the 6th cloud sub-game, the cop emails the §3.5 report exactly once:

```bash
uv run python scripts/run_match.py --send    # needs GMAIL_SENDER / GMAIL_APP_PASSWORD
```

## What actually happened (the LIVE run, 2026-07-22)

Deployed via `deploy/render.yaml` (Blueprint → repo → path `deploy/render.yaml`); the only
dashboard secret is `MCP_PUBLIC_KEY` (the one-line `\n`-escaped PEM; the verifier
normalizes it). Captured evidence, all against the two public endpoints:

1. **Full match over the cloud** → `results/figures/mcp_comms_cloud.png` (F4c): **6 valid
   sub-games** (traces `sg-0`…`sg-5`), every `request_move` alternating cop↔thief, final
   cop 30 – thief 60. Its §3.5 body: `results/subgames/cloud_match_5x5.redacted.json`.
2. **§5.3 mutual position verification** (same capture): `reveal_location` answered the
   *other* agent's HTTP query radius-gated — adjacent requester → `{visible: true,
   position: [1,0]}`, distant requester → `{visible: false}`, in both directions.
3. **Auth matrix** → `results/figures/cloud_auth.png`: valid RS256 → 200; bad token,
   missing token, and **wrong-audience** token → 401; revoked `jti` → 401.

Deployment gotchas found the hard way (all fixed in-repo):

- The builder installs base deps only (`uv sync --no-dev`) → `fastmcp`/`httpx`/`pyjwt`
  must live in `[project.dependencies]`, not a dependency-group.
- Linux needs CPU-only torch (`[tool.uv.sources]` → pytorch-cpu index) or the image
  drags ~2 GB of CUDA wheels and blows the free tier.
- The build-time `uv` binary is **not** preserved at runtime → start via
  `.venv/bin/fastmcp` (deps are still uv-resolved, `--frozen`).
- fastmcp 3.x wraps `RSAKeyPair.private_key` in a pydantic `SecretStr`.
- Free tier sleeps after 15 min → the first request takes **~90 s** (measured 2026-07-30).
  The per-move budget (`mcp.client.timeout_s` 10 × `max_retries` 3 ≈ 31 s) is deliberately far
  too tight for that, so the warm-up does NOT rely on it: `prewarm_ping` polls health to its own
  `mcp.client.prewarm_deadline_s` (180 s) budget. Mention the ~90 s wake to anyone testing the URLs.
