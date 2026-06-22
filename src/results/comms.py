"""Localhost MCP-comms proof (F4, §7.3d) — capture + redact + render the cop<->thief log.

Runs a short in-memory match, captures the structured ``marl.mcp.client`` log lines (each
``client=cop/thief tool=... trace=<session_id>``), REDACTS any bearer/token material, and
renders the transcript to a PNG. The SAME trace appears on BOTH servers' calls within a
sub-game — the cross-server-comms proof. Pure results layer (routes through the SDK).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend before pyplot

import matplotlib.pyplot as plt

from src.reporting.players import load_players
from src.sdk.sdk import MarlSDK

# Catch `[Authorization:] Bearer <token>` (incl. the token after Bearer) AND `key=value`.
_REDACT = re.compile(
    r"(?:authorization\s*[=:]\s*)?bearer\s+\S+|(?:token|api[_-]?key|password|secret)\s*[=:]\s*\S+",
    re.IGNORECASE,
)


class _Capture(logging.Handler):
    """A logging handler that collects redacted ``marl.mcp.client`` messages."""

    def __init__(self) -> None:
        """Start with an empty transcript."""
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Append the record's message with any bearer/token material redacted."""
        self.lines.append(_REDACT.sub("<redacted>", record.getMessage()))


def capture(cfg: dict) -> list[str]:
    """Run a 1-sub-game in-memory match; return the redacted cop<->thief comms transcript."""
    handler = _Capture()
    logger = logging.getLogger("marl.mcp.client")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        sdk = MarlSDK(cfg)
        seed = int(cfg["training"]["seeds"][0])
        sdk.run_local_match(sdk.fresh_net("cop"), sdk.fresh_net("thief"), load_players(), seed, num_games=1)
    finally:
        logger.removeHandler(handler)
    return handler.lines


def render(lines: list[str], out_path: str | Path) -> Path:
    """Render the comms transcript (first 40 lines) to a monospace PNG; return the path."""
    shown = lines[:40]
    fig, ax = plt.subplots(figsize=(9.0, 0.28 * len(shown) + 1.0))
    ax.axis("off")
    ax.set_title("localhost cop<->thief MCP comms (redacted) — shared trace per sub-game")
    ax.text(0.01, 0.99, "\n".join(shown), va="top", family="monospace", fontsize=8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_comms_proof(cfg: dict) -> Path:
    """Capture + render the F4 localhost MCP-comms proof; return the figure path."""
    return render(capture(cfg), Path(cfg["paths"]["figures_dir"]) / "mcp_comms_local.png")
