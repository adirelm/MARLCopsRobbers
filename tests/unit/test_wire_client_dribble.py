"""A dribbling peer must NOT hold the referee past the P8 wall-clock budget (was CRITICAL).

httpx's scalar timeout is only per-phase: a peer feeding one byte inside every read
window keeps the blocking POST alive indefinitely, so the post-hoc elapsed check alone
ran only AFTER the peer chose to finish. The client now runs each attempt on a daemon
worker joined at the budget and abandons it on expiry (the orphan's late result lands in
a discarded local box). Pinned against a REAL socket server that dribbles the body one
byte per 30 ms — the fault must land within budget + epsilon, not after the body.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

from src.mcp.wire_client import VoidSubGame, WireClient

_GAP_S = 0.03  # one body byte per gap — each read lands well inside any per-phase timeout
_BODY_BYTES = 60  # bounded so the abandoned worker's connection winds down after the test


def _serve_dribble(conn: socket.socket) -> None:
    """Answer one request with valid headers, then dribble the body one byte at a time."""
    with conn:
        conn.recv(65536)
        head = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {_BODY_BYTES}\r\n\r\n"
        try:
            conn.sendall(head.encode())
            for _ in range(_BODY_BYTES):
                conn.sendall(b" ")
                time.sleep(_GAP_S)
        except OSError:
            pass  # client side gone — done


def _accept_loop(sock: socket.socket) -> None:
    """Serve every connection on its own daemon thread until the listener closes."""
    while True:
        try:
            conn, _ = sock.accept()
        except OSError:
            return
        threading.Thread(target=_serve_dribble, args=(conn,), daemon=True).start()


def test_dribbling_peer_faults_within_the_wall_clock_budget():
    sock = socket.create_server(("127.0.0.1", 0))
    threading.Thread(target=_accept_loop, args=(sock,), daemon=True).start()
    budget = 0.2  # dribbling the full body would take ~1.8s — 9x the budget
    client = WireClient(f"http://127.0.0.1:{sock.getsockname()[1]}", "tok", timeout_s=budget, retries=0)
    started = time.monotonic()
    try:
        with pytest.raises(VoidSubGame, match="wall clock"):
            client.request_move({"session_id": "sg-0", "tick": 0})
        elapsed = time.monotonic() - started
    finally:
        sock.close()
    assert elapsed < budget + 0.15  # bounded by join(budget), NOT by when the peer stops dribbling
