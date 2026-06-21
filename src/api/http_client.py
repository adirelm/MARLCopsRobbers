"""The single bearer-authed httpx wrapper — the ONLY raw HTTP egress (§5, T5.9).

Every NON-MCP outbound HTTP call goes through here; the egress-boundary test fails
if any module outside ``src/api`` (and later ``src/reporting/mailer.py``) imports
``httpx``. Callers route these through :meth:`ApiGatekeeper.execute` so the per-
channel rate limits + FIFO overflow queue apply to all egress.
"""

from __future__ import annotations

import httpx


def bearer_get(url: str, token: str, timeout: float = 10.0) -> httpx.Response:
    """GET ``url`` with a bearer token (wrap the call in ApiGatekeeper.execute)."""
    return httpx.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=timeout)


def bearer_post(url: str, token: str, payload: dict, timeout: float = 10.0) -> httpx.Response:
    """POST ``payload`` JSON to ``url`` with a bearer token (route via the gatekeeper)."""
    return httpx.post(url, headers={"Authorization": f"Bearer {token}"}, json=payload, timeout=timeout)
