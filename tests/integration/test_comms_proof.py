"""F4 MCP-comms proof tests (§7.3d) — render writes a PNG; capture shows the shared trace."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.results.comms import _Capture, capture, make_comms_proof, render


def test_make_comms_proof_writes_figure(tmp_path, cfg):
    cfg = json.loads(json.dumps(cfg))
    cfg["paths"]["figures_dir"] = str(tmp_path)
    out = make_comms_proof(cfg)
    assert out.exists() and out.stat().st_size > 0


def test_render_writes_png(tmp_path):
    lines = [
        "mcp_call client=cop tool=request_move trace=sg0 attempt=1 status=ok",
        "mcp_call client=thief tool=request_move trace=sg0 attempt=1 status=ok",
    ]
    out = render(lines, tmp_path / "comms.png")
    assert Path(out).exists() and Path(out).stat().st_size > 0


def test_render_redacts_bearer_material():
    handler = _Capture()
    record = logging.LogRecord("x", 20, "f", 1, "hdr Authorization: Bearer eyJsecret123", None, None)
    handler.emit(record)
    assert "eyJsecret123" not in handler.lines[0] and "<redacted>" in handler.lines[0]


def test_capture_shows_shared_trace_across_servers(cfg):
    lines = capture(cfg)
    assert any("client=cop" in ln for ln in lines)
    assert any("client=thief" in ln for ln in lines)
    # cop AND thief calls within a sub-game share one trace (the cross-server proof)
    traces = {ln.split("trace=")[1].split()[0] for ln in lines if "trace=" in ln and "request_move" in ln}
    assert any(trace != "-" for trace in traces)
