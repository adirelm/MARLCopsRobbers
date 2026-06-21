"""``python -m src.gui`` — launch the Pygame god-view spectator (T7; routed through the SDK).

Thin module entrypoint mirroring ``scripts/play.py``: builds a 5x5 spectator session via
the SDK and runs the window loop (in ``src.gui.render``). REQUIRES pygame (the ``gui``
extra). Coverage-omitted — it opens an interactive window and cannot run headless in CI.
"""

from __future__ import annotations

from src.gui.render import run_app
from src.gui.state_client import InProcStateClient
from src.sdk.sdk import MarlSDK
from src.utils.config_loader import load_config


def main() -> None:
    """Build a 5x5 spectator session via the SDK and run the god-view window loop."""
    cfg = load_config()
    session = MarlSDK(cfg).spectator_session(5, 5, num_cops=1, seed=0)
    run_app(InProcStateClient(session))


if __name__ == "__main__":
    main()
