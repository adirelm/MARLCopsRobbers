"""Headless screenshot matrix (T7.6; REQUIRES pygame) — the §7.3c F3 evidence.

Renders one frame per board size (``gui.screenshot_sizes`` = 2x2/3x3/4x4/5x5) with
cop + thief + >=1 barrier visible and saves a PNG under ``gui.screenshot_dir``.
Headless (``SDL_VIDEODRIVER=dummy``). pygame can't build on py3.14/this host, so
run on a pygame-capable machine: ``uv run python scripts/capture_screens.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.sdk.sdk import MarlSDK
from src.utils.config_loader import load_config


def main(cfg: dict | None = None) -> list[str]:  # pragma: no cover - requires pygame
    """Render + save one screenshot per configured board size; return the paths."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame  # noqa: PLC0415 - lazy: pygame is the optional gui extra

    from src.gui.render import render_frame  # noqa: PLC0415 - lazy with pygame

    cfg = cfg or load_config()
    out_dir = Path(cfg["gui"]["screenshot_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pygame.init()
    sdk = MarlSDK(cfg)
    font = pygame.font.SysFont(None, 24)
    saved = []
    for size in cfg["gui"]["screenshot_sizes"]:
        session = sdk.spectator_session(size, size, num_cops=1, seed=7)
        frame = session.reset()
        for _ in range(3):
            frame = session.step()
        surface = pygame.Surface((720, 560))
        render_frame(surface, font, frame)
        path = out_dir / f"grid_{size}x{size}.png"
        pygame.image.save(surface, str(path))
        saved.append(str(path))
    pygame.quit()
    return saved


if __name__ == "__main__":
    main()
