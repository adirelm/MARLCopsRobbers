# Cloud deploy runbook (§8 four-stage) — MARL Cops & Robbers

The copy-pasteable, ordered version of the BRIEF §8 appendix. Cloud is **upside, not an
A6 grading gate** (ADR-D10-E) — the localhost match (F4, Phase 5/6) is canonical. Both
cloud servers expose the **same** canonical tool contract + `/mcp` path as localhost
(ADR-D5-01); the only delta is the auth verifier (static → RS256 JWT).

> Requires accounts you own: a Prefect Horizon (`horizon.prefect.io`) login and, for the
> fallback, a Render account. These are the only steps Claude cannot run for you.

## Stage 1 — local uv deps (T8.0)

```bash
uv sync --group mcp --group cloud          # fastmcp + httpx + pyjwt + prefect
uv run python -c 'import fastmcp; print(fastmcp.__version__)'   # expect 3.x
uv run pytest tests/unit/test_jwt_auth.py -q                    # auth proven offline
```

There is **no** `deploy/requirements.txt` (V3 forbids it). If a target demands a pinned
file, generate it at deploy time and never track it: `uv export --no-dev > /tmp/req.txt`.

## Stage 2 — mint an RS256 keypair + set the server env (T8.1)

```bash
# Generate a keypair (private mints client tokens; public verifies on the servers):
uv run python -c "from fastmcp.server.auth.providers.jwt import RSAKeyPair; \
kp=RSAKeyPair.generate(); open('/tmp/mcp_priv.pem','w').write(kp.private_key); \
open('/tmp/mcp_pub.pem','w').write(kp.public_key); print('wrote /tmp/mcp_{priv,pub}.pem')"
```

Each server (cop / thief) needs, in its cloud env (NEVER tracked — see `.env-example`):

| Var | Meaning |
|---|---|
| `MCP_AUTH_MODE=jwt` | selects the RS256 verifier via the `build_verifier` seam |
| `MCP_PUBLIC_KEY` | the RS256 public-key PEM (verifies bearer tokens) |
| `MODEL_PATH` | path to the OLoRA-tuned actor `.pt` (cop=`n_agents 2`, thief=`1`) |
| `REVOKED_TOKEN_JTIS` | comma-separated jti deny-list (the revoke demo) |
| `PEER_MCP_URL` / `PEER_MCP_TOKEN` | the other server's URL + a bearer for `query_opponent` |

Deploy entrypoints (module-level `mcp`): `src/mcp/cloud_cop.py:mcp` and
`src/mcp/cloud_thief.py:mcp`.

## Stage 3 — deploy to Prefect Horizon + publish URLs (T8.2/T8.3)

```bash
# 1. Sign in to horizon.prefect.io with GitHub; create a deploy key:
export PREFECT_API_KEY=...          # from the Horizon UI (egress is gatekept: prefect_deploy 30/min)
# 2. Deploy each server (entrypoint, name, env per Stage 2); disable org-OAuth, rely on app JWT:
#    cop   -> name adrl-001-cop   -> https://adrl-001-cop.fastmcp.app/mcp
#    thief -> name adrl-001-thief -> https://adrl-001-thief.fastmcp.app/mcp
# 3. Record both URLs in config/config.yaml -> cloud.cop_url / cloud.thief_url.
# 4. Verify health() over the public internet (from a machine OFF the dev host):
uv run python -c "import asyncio; from fastmcp import Client; from fastmcp.client.auth import BearerAuth; \
print(asyncio.run(Client('https://adrl-001-cop.fastmcp.app/mcp', auth=BearerAuth('<token>')).__aenter__()))"
```

Mint a client token from the private key (the minter):

```bash
uv run python -c "from fastmcp.server.auth.providers.jwt import RSAKeyPair; \
kp=RSAKeyPair(private_key=open('/tmp/mcp_priv.pem').read(), public_key=open('/tmp/mcp_pub.pem').read()); \
print(kp.create_token(subject='marl-cop', issuer='adrl-001-mcp-auth', audience='marl-cop', \
scopes=['game:write'], additional_claims={'jti':'cop-001'}))"
```

## Stage 4 — Gmail report (Phase 9)

After the 6th cloud sub-game, the cop emails the §3.5 report exactly once:

```bash
uv run python scripts/run_match.py --send    # needs GMAIL_SENDER / GMAIL_APP_PASSWORD
```

## §7.3d proof captures (run `scripts/redact_logs.py` BEFORE every screenshot)

1. **cloud comms** — a 3-turn cross-server run; the SAME `trace_id` in BOTH logs → `cloud_comms.png`.
2. **401 without token** — call a tool with no / a bad bearer → `cloud_auth_401.png`.
3. **revoke** — add the token's jti to `REVOKED_TOKEN_JTIS`, redeploy, re-run → old token 401,
   a freshly-minted token 200 → `cloud_revoke.png`.

## Render fallback (T8.5)

`deploy/render.yaml` is a Render blueprint for the same two entrypoints. After a Render
deploy, warm it with a `health()` ping (cold start) before the match.
