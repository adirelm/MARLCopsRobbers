"""Capture the localhost cop<->thief MCP-comms proof -> results/figures/mcp_comms_local.png (F4).

Thin launcher over ``src.results.comms.make_comms_proof`` (all logic lives in the SDK /
results layer). Renders the redacted cop<->thief comms transcript — the SAME trace on BOTH
servers per sub-game (§7.3d evidence). Headless. Run: ``uv run python scripts/capture_comms.py``.
"""

from __future__ import annotations

from src.results.comms import make_comms_proof
from src.utils.config_loader import load_config


def main(cfg: dict | None = None) -> None:
    """Render the F4 localhost MCP-comms proof from a short in-memory match."""
    out = make_comms_proof(cfg or load_config())
    print(f"[comms] rendered the redacted localhost cop<->thief comms proof -> {out}")


if __name__ == "__main__":
    main()
