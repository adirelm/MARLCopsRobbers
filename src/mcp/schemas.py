"""MCP wire schemas — the inter-agent request/response contract (T5.2).

Pydantic models pinning the dual-server protocol. Every per-tick request REQUIRES
``session_id`` (it keys the server-side GRU hidden state ``z_t`` AND gives
``(session_id, tick)`` idempotency so a retried tick never double-advances state).
``extra="forbid"`` on the requests REJECTS any unknown field — most importantly a
``global_state`` leak into ``request_move`` (local-obs-only, partial observability).
A ``MoveResponse`` exposes ONLY the chosen ``action`` — never a value / logit /
hidden (policy internals are unreachable from any tool return). ``reveal_location``
is radius-gated: the position is ``None`` when the requester is beyond view radius.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

_STRICT = ConfigDict(extra="forbid")


class NewSubGameRequest(BaseModel):
    """Start a sub-game session: resets the server's per-session hidden ``z_t``."""

    model_config = _STRICT
    session_id: str
    grid: tuple[int, int]
    num_cops: int = 1
    position: tuple[int, int] | None = None  # the agent's OWN start cell (its own info)


class MoveRequest(BaseModel):
    """A per-tick local-obs move request (rejects any ``global_state`` field)."""

    model_config = _STRICT
    session_id: str
    tick: int
    image: list
    scalars: list[float]
    legal_mask: list[bool]


class MoveResponse(BaseModel):
    """The agent's chosen action ONLY — no value / logit / hidden ever leaves."""

    model_config = _STRICT
    action: int


class RevealRequest(BaseModel):
    """A radius-gated location query from a peer (evidence-only).

    The requester provides its OWN cell so the server can decide visibility; the
    reveal is mutual — you only learn a peer's position if you are within its view.
    """

    model_config = _STRICT
    session_id: str
    requester: str
    requester_pos: tuple[int, int]


class RevealResponse(BaseModel):
    """The radius-gated reveal: ``position`` is ``None`` beyond the view radius."""

    model_config = _STRICT
    visible: bool
    position: tuple[int, int] | None = None


class HealthResponse(BaseModel):
    """Liveness + the cross-server protocol-version handshake string."""

    model_config = _STRICT
    status: str
    protocol_version: str
