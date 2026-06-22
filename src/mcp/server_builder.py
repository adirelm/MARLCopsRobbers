"""Shared FastMCP server builder — ONE canonical tool contract for both roles (T5.6).

Builds a role's FastMCP server registering the canonical tools wired to its
:class:`~src.mcp.agent_runtime.AgentController`. The SAME builder serves the cop and
thief servers (DRY; one contract for localhost AND cloud — no divergent fork). Each
handler documents Input/Output/Setup and calls ``_require`` (raises ``TypeError`` on
a malformed payload, V3 §16). ``request_move`` carries LOCAL obs only (pydantic
``extra="forbid"`` rejects a ``global_state`` leak at the protocol edge), and a
``MoveResponse`` returns only the action — no value/logit/hidden ever escapes.
"""

from __future__ import annotations

from fastmcp import FastMCP

from src.mcp.agent_runtime import AgentController
from src.mcp.schemas import (
    HealthResponse,
    MoveRequest,
    MoveResponse,
    NewSubGameRequest,
    QueryOpponentRequest,
    QueryOpponentResponse,
    ReportAck,
    RevealRequest,
    RevealResponse,
)
from src.reporting.schema import validate as validate_report


def _require(condition: object, message: str) -> None:
    """§16 input guard: raise ``TypeError`` on a malformed/empty payload field."""
    if not condition:
        raise TypeError(message)


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Return the L1 (Manhattan) distance between two board cells."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def build_server(
    cfg: dict,
    role: str,
    controller: AgentController,
    verifier: object,
    peer_query: object = None,
) -> FastMCP:
    """Build a role's FastMCP server with the canonical tools (auth-gated).

    Args:
        cfg: Loaded config (reads ``mcp.protocol_version`` + ``mcp.observation``).
        role: ``"cop"`` / ``"thief"`` (the server name).
        controller: The role's per-session :class:`AgentController`.
        verifier: The FastMCP auth verifier (Stage-1 static / Stage-2 JWT).
        peer_query: Optional ``(session_id, requester_role, own_pos) -> {"visible", "position"}``
            seam backing ``query_opponent`` (the orchestrator wires it to the peer's
            radius-gated reveal); ``None`` yields no evidence.

    Returns:
        A configured :class:`FastMCP` server (call ``.run`` or test in-memory).
    """
    proto = cfg["mcp"]["protocol_version"]
    view_radius = int(cfg["mcp"]["observation"]["view_radius"])
    positions: dict[str, tuple[int, int] | None] = {}
    mcp = FastMCP(name=f"marl-{role}", auth=verifier)

    @mcp.tool
    def health() -> dict:
        """In: none. Out: HealthResponse. Setup: none — liveness + protocol handshake."""
        return HealthResponse(status="ok", protocol_version=proto).model_dump()

    @mcp.tool
    def ping() -> str:
        """In: none. Out: 'pong'. Setup: none — cheap reachability probe."""
        return "pong"

    @mcp.tool
    def new_sub_game(req: NewSubGameRequest) -> dict:
        """In: NewSubGameRequest. Out: ack. Setup: none — start a session (fresh z_0)."""
        _require(req.session_id, "session_id must be non-empty")
        controller.new_session(req.session_id)
        positions[req.session_id] = tuple(req.position) if req.position is not None else None
        return {"session_id": req.session_id, "ok": True}

    @mcp.tool
    def request_move(req: MoveRequest) -> dict:
        """In: MoveRequest (LOCAL obs only). Out: MoveResponse. Setup: new_sub_game first.

        Idempotent on ``(session_id, tick)`` — a retried tick returns the cached
        action without re-advancing the GRU state.
        """
        _require(req.session_id, "session_id must be non-empty")
        action = controller.act(req.session_id, req.tick, req.image, req.scalars, req.legal_mask)
        return MoveResponse(action=action).model_dump()

    @mcp.tool
    def reveal_location(req: RevealRequest) -> dict:
        """In: RevealRequest. Out: RevealResponse (radius-gated). Setup: new_sub_game w/ position.

        Reveals the position the referee PROVISIONED at new_sub_game (the only
        absolute cell a decentralized server legitimately holds) — radius-gated.
        """
        _require(req.session_id, "session_id must be non-empty")
        own = positions.get(req.session_id)
        visible = own is not None and _manhattan(own, req.requester_pos) <= view_radius
        return RevealResponse(visible=visible, position=own if visible else None).model_dump()

    @mcp.tool
    def query_opponent(req: QueryOpponentRequest) -> dict:
        """In: QueryOpponentRequest. Out: QueryOpponentResponse (evidence-only). Setup: peer reachable.

        Passes THIS server's own provisioned cell to the peer so the peer can
        radius-gate its reveal; the evidence never re-enters request_move.
        """
        _require(req.session_id, "session_id must be non-empty")
        own = positions.get(req.session_id)
        if peer_query is None or own is None:
            return QueryOpponentResponse(visible=False, position=None).model_dump()
        return QueryOpponentResponse(**peer_query(req.session_id, req.requester_role, own)).model_dump()

    if role == "cop":  # the Cop (and ONLY the Cop) holds the report tool

        @mcp.tool
        def send_final_report(report: dict) -> dict:
            """In: §3.5 report body. Out: ReportAck. Setup: cop-only, after the 6th sub-game (dry-run).

            Validates the report (schema + derived-totals) at the boundary before ack.
            """
            _require(report, "report must be non-empty")
            try:
                validate_report(report)
            except ValueError as exc:
                raise TypeError(f"invalid report body: {exc}") from exc
            return ReportAck(sent=False, dry_run=True).model_dump()

    return mcp
