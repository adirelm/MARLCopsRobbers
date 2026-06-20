"""Tests for the MCP wire schemas (T5.2) — the inter-agent protocol contract.

Pins: every request requires session_id; request_move rejects a global_state leak;
MoveResponse exposes ONLY the action (no value/logit/hidden); reveal is radius-gated.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.mcp.schemas import MoveRequest, MoveResponse, NewSubGameRequest, RevealRequest, RevealResponse

_REQUESTS = [MoveRequest, NewSubGameRequest, RevealRequest]


def _minimal(model) -> dict:
    """Return a minimal valid payload for a request model."""
    base = {"session_id": "s1"}
    if model is MoveRequest:
        return {**base, "tick": 0, "image": [[[0.0]]], "scalars": [0.0], "legal_mask": [True]}
    if model is NewSubGameRequest:
        return {**base, "grid": [2, 2]}
    return {**base, "requester": "thief"}


@pytest.mark.parametrize("model", _REQUESTS)
def test_every_request_requires_session_id(model):
    """Each request model rejects a payload missing session_id."""
    payload = _minimal(model)
    del payload["session_id"]
    with pytest.raises(ValidationError):
        model(**payload)


def test_move_request_rejects_global_state_leak():
    """request_move forbids unknown fields — a global_state leak is rejected."""
    with pytest.raises(ValidationError):
        MoveRequest(**{**_minimal(MoveRequest), "global_state": [1, 2, 3]})


def test_move_response_exposes_only_action():
    """MoveResponse carries ONLY action — no value/logit/hidden field exists or is accepted."""
    assert set(MoveResponse.model_fields) == {"action"}
    with pytest.raises(ValidationError):
        MoveResponse(action=1, value=0.5)


def test_reveal_response_is_radius_gated():
    """RevealResponse position is None when not visible (beyond view radius)."""
    assert RevealResponse(visible=False).position is None
    assert RevealResponse(visible=True, position=(1, 2)).position == (1, 2)
