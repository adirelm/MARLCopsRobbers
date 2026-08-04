"""``uv run python -m src.gui`` — launch the Pygame god-view spectator (T7; routed through the SDK).

Thin module entrypoint mirroring ``scripts/play.py``: builds a 5x5 spectator session via
the SDK and runs the window loop (in ``src.gui.render``). REQUIRES pygame (the ``gui``
extra). Coverage-omitted — it opens an interactive window and cannot run headless in CI.
"""

from __future__ import annotations

import argparse

from src.gui.render import run_app
from src.gui.state_client import InProcStateClient
from src.sdk.sdk import MarlSDK
from src.utils.config_loader import load_config


def main(argv: list[str] | None = None) -> None:
    """Build a spectator session (final board, config seed) via the SDK; run the window loop.

    Parses argv even though both flags default from config. Without a parser this module
    IGNORED argv entirely and fell straight into the window loop, so ``-m src.gui --help``
    hung forever with no output on a headless host instead of printing usage — and a typo'd
    flag was silently swallowed. Its sibling ``scripts/play.py`` always had a parser; two
    documented spectator entrypoints should not have different CLI contracts.
    """
    cfg = load_config()
    board, seeds = cfg["env"]["curriculum"]["stages"][-1], cfg["training"]["seeds"]
    parser = argparse.ArgumentParser(description="MARL Cops & Robbers god-view spectator")
    parser.add_argument("--seed", type=int, default=int(seeds[0]))
    parser.add_argument("--grid", type=int, default=int(board[0]), help="square board size")
    args = parser.parse_args(argv)
    cops = int(cfg["env"]["num_cops"])  # V3 no-hardcode, and keeps this launcher == scripts/play.py
    session = MarlSDK(cfg).spectator_session(args.grid, args.grid, num_cops=cops, seed=args.seed)
    run_app(InProcStateClient(session))  # pragma: no cover - requires pygame


if __name__ == "__main__":
    main()
