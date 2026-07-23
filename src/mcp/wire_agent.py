"""Our-side §9 wire agent — the partner-brief HTTP adapter serving OUR policies.

Serves docs/interfaces/partner_agent_brief.md §2 verbatim over a STDLIB
:class:`ThreadingHTTPServer` (no new web-framework dependency): unauthenticated
``GET /health``; bearer-authed ``POST /new_sub_game`` (FULL per-session reset —
policy hidden state, visibility counter, idempotency cache — because a voided
sub-game is replayed under the SAME ``session_id``) and ``POST /request_move``
(rebuilds the exact env Observation + legality mask via :mod:`src.mcp.wire_obs`,
advances the plugged-in ``act()``+``reset()`` policy — ``MarlSDK.build_policy``
output, e.g. a :class:`RecurrentPolicy` whose hidden state moves one tick per
move — and answers the brief's action string). A re-POSTed IDENTICAL
``(session_id, tick)`` body returns the cached answer WITHOUT re-advancing the
recurrent state; a DIFFERING body for a cached tick — a stale pre-void request a
server thread finishes AFTER the void re-hello reset the session — recomputes and
overwrites, so it can never poison the new run. A lock serializes acting so a
referee timeout-retry can never double-advance z_t; the acting rng is reseeded on
every hello so a void replay redraws the SAME stream. Acting is greedy (ε=0,
decentralized execution — the ``AgentController`` convention). One server serves
ONE role; ``make_wire_agent`` returns a started handle.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from random import Random

from src.marl.env.actions import Action
from src.mcp import wire_obs

# Wire action strings (brief §2) keyed by the frozen env action ints.
_WIRE_ACTIONS = {
    Action.UP: "up",
    Action.DOWN: "down",
    Action.LEFT: "left",
    Action.RIGHT: "right",
    Action.PLACE_BARRIER: "place_barrier",
}
# Error -> HTTP status: bad values 400, unknown session/field 404, stale session 409.
_STATUS = {ValueError: 400, TypeError: 400, KeyError: 404, RuntimeError: 409}


class _WireHandler(BaseHTTPRequestHandler):
    """Route the brief's three endpoints; auth + typed errors -> JSON responses."""

    server: WireAgentServer

    def log_message(self, format: str, *args: object) -> None:
        """Silence the default per-request stderr access log."""

    def _reply(self, code: int, body: dict) -> None:
        """Write ``body`` as the JSON response with HTTP status ``code``."""
        data = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        """Serve the UNAUTHENTICATED ``/health`` probe (brief §2)."""
        if self.path == "/health":
            self._reply(200, {"status": "ok"})
        else:
            self._reply(404, {"error": f"unknown path {self.path}"})

    def do_POST(self) -> None:
        """Serve the two bearer-authed POST endpoints (401 before any parsing)."""
        srv = self.server
        if self.headers.get("Authorization") != f"Bearer {srv.token}":
            self._reply(401, {"error": "unauthorized"})
            return
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            if self.path == "/new_sub_game":
                self._reply(200, srv.start_sub_game(payload))
            elif self.path == "/request_move":
                self._reply(200, srv.serve_move(payload))
            else:
                self._reply(404, {"error": f"unknown path {self.path}"})
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError) as err:
            self._reply(_STATUS.get(type(err), 400), {"error": f"{type(err).__name__}: {err}"})


class WireAgentServer(ThreadingHTTPServer):
    """One role's brief-§2 server: session registry, idempotency cache, greedy policy."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], cfg: dict, role: str, policy: object, token: str) -> None:
        """Bind the role's acting policy + bearer token; start with no sessions."""
        super().__init__(address, _WireHandler)
        self.cfg, self.role, self.policy, self.token = cfg, role, policy, token
        self._sessions: dict[str, dict] = {}
        self._active: str | None = None
        self._lock = threading.Lock()  # a timeout re-POST must never double-advance z_t
        self._rng = Random(0)  # reseeded per hello: stochastic policies (flee pre-contact,
        #   conformance uniform-random) DO draw, and a void replay must redraw the same stream

    def start_sub_game(self, payload: dict) -> dict:
        """Reset ALL state for ``session_id`` — a void replay re-POSTs the SAME id."""
        with self._lock:
            if payload["your_role"] != self.role:
                raise ValueError(f"this server plays {self.role!r}, not {payload['your_role']!r}")
            wire = wire_obs.new_session(self.role, payload["grid"], payload["max_moves"], self.cfg)
            self._sessions[payload["session_id"]] = {"wire": wire, "cache": {}, "last_tick": -1}
            self._active = payload["session_id"]
            self._rng = Random(0)  # FRESH deterministic stream: a void replay reproduces attempt 1
            self.policy.reset()
        return {"ok": True}

    def serve_move(self, payload: dict) -> dict:
        """Answer one ``request_move`` tick, idempotently per ``(session_id, tick, exact body)``."""
        with self._lock:
            sid, tick = payload["session_id"], int(payload["tick"])
            session = self._sessions.get(sid)
            if session is None:
                raise KeyError(f"unknown session {sid!r}: POST /new_sub_game first")
            body_key = json.dumps(payload, sort_keys=True)  # a genuine P8 retry re-POSTs THIS byte-body
            cached = session["cache"].get(tick)
            if cached is not None and cached["key"] == body_key:
                return cached["body"]  # idempotent re-POST: do NOT re-advance z_t
            if sid != self._active:
                raise RuntimeError(f"session {sid!r} is no longer active; only cached ticks replay")
            if cached is None and tick != session["last_tick"] + 1:
                raise ValueError(f"tick {tick} breaks the 0-indexed sequence (last={session['last_tick']})")
            obs = wire_obs.build_observation(session["wire"], payload, self.cfg)
            mask = [bool(b) for b in wire_obs.build_mask(session["wire"], payload, self.cfg)]
            action = self.policy.act([obs], [mask], 0.0, self._rng)[0]
            body = {"action": _WIRE_ACTIONS[Action(int(action))]}
            session["cache"][tick] = {"key": body_key, "body": body}
            session["last_tick"] = max(session["last_tick"], tick)
            return body


class WireAgent:
    """A STARTED wire-agent handle: daemon serve thread + ``port`` + ``close()``."""

    def __init__(self, server: WireAgentServer) -> None:
        """Begin serving ``server`` on a daemon thread immediately."""
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        """The bound TCP port (resolves an ephemeral ``port=0`` request)."""
        return int(self._server.server_address[1])

    def close(self) -> None:
        """Stop serving, release the socket, and join the serve thread."""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=10)


def make_wire_agent(cfg: dict, role: str, policy: object, token: str, port: int = 0) -> WireAgent:
    """Start one role's wire agent; return the running :class:`WireAgent` handle.

    Args:
        cfg: The loaded config. The bind host is ``wire_agent.host`` when present
            (key owned by the referee-side config), else ``mcp.host``.
        role: ``"cop"`` or ``"thief"`` — the only ``your_role`` this server accepts.
        policy: Any ``act()``+``reset()`` acting object (``MarlSDK.build_policy``).
        token: The bearer token every POST must carry (§5.3; value never tracked).
        port: TCP port to bind; ``0`` picks an ephemeral free port (tests).

    Returns:
        A started :class:`WireAgent`; call ``close()`` to stop it.
    """
    host = cfg.get("wire_agent", {}).get("host", cfg["mcp"]["host"])
    return WireAgent(WireAgentServer((host, int(port)), cfg, role, policy, token))
