"""Egress boundary: httpx is imported ONLY inside src/api (§5, T5.9 DoD).

ALL raw HTTP egress must go through src/api/http_client.py (routed via the
ApiGatekeeper). This test fails if any module outside src/api (and, once it exists,
src/reporting/mailer.py) imports httpx — preventing an ungoverned egress path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.api import http_client
from src.api.gatekeeper import ApiGatekeeper

_SRC = Path(__file__).resolve().parents[2] / "src"
_ALLOWED_PARTS = {"api"}  # + reporting/mailer.py when the Gmail sender lands (P9)


def test_httpx_imported_only_inside_api():
    """No src module outside src/api imports httpx (the single sanctioned wrapper)."""
    offenders = []
    for py in _SRC.rglob("*.py"):
        if set(py.parts) & _ALLOWED_PARTS:
            continue
        if "import httpx" in py.read_text(encoding="utf-8"):
            offenders.append(str(py.relative_to(_SRC)))
    assert offenders == [], f"httpx imported outside src/api: {offenders}"


def test_bearer_get_sets_authorization_header(monkeypatch):
    """bearer_get attaches the bearer token (the only httpx GET wrapper)."""
    captured = {}

    def fake_get(url, headers, timeout):
        captured.update(url=url, headers=headers, timeout=timeout)
        return "resp"

    monkeypatch.setattr(http_client.httpx, "get", fake_get)
    assert http_client.bearer_get("http://x/mcp", "tok") == "resp"
    assert captured["headers"]["Authorization"] == "Bearer tok"


def test_bearer_post_sends_json_with_bearer(monkeypatch):
    """bearer_post attaches the bearer token + JSON body (the only httpx POST wrapper)."""
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured.update(headers=headers, json=json)
        return "resp"

    monkeypatch.setattr(http_client.httpx, "post", fake_post)
    assert http_client.bearer_post("http://x", "tok", {"a": 1}) == "resp"
    assert captured["headers"]["Authorization"] == "Bearer tok"
    assert captured["json"] == {"a": 1}


def test_gatekeeper_is_importable_as_single_entry():
    """The §5 gatekeeper exposes execute + get_queue_status (the single egress entry)."""
    assert hasattr(ApiGatekeeper, "execute")
    assert hasattr(ApiGatekeeper, "get_queue_status")


@pytest.mark.parametrize("symbol", ["bearer_get", "bearer_post"])
def test_http_client_exposes_bearer_wrappers(symbol):
    """The http_client exposes both bearer wrappers."""
    assert hasattr(http_client, symbol)
