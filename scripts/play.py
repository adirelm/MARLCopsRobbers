"""Spectator launcher — opens the god-view window (T7.6; REQUIRES pygame).

Routes through the SDK (``spectator_session``) + the source-agnostic StateClient;
the window loop lives in ``src.gui.render`` (guarded). pygame can't build on
py3.14/this host, so run on a pygame-capable machine:
``uv run python scripts/play.py --seed 7``.
"""

from __future__ import annotations

import argparse

from src.gui.render import run_app
from src.gui.state_client import InProcStateClient
from src.sdk.sdk import MarlSDK
from src.utils.config_loader import load_config


def main() -> None:
    """Parse flags, build a spectator session via the SDK, and run the window loop."""
    parser = argparse.ArgumentParser(description="MARL Cops & Robbers god-view spectator")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid", type=int, default=5)
    args = parser.parse_args()
    cfg = load_config()
    session = MarlSDK(cfg).spectator_session(args.grid, args.grid, num_cops=1, seed=args.seed)
    run_app(InProcStateClient(session))  # pragma: no cover - requires pygame


if __name__ == "__main__":
    main()
