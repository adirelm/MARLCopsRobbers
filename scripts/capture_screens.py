"""Headless screenshot matrix (T7.6; REQUIRES pygame-ce) — the §7.3c F3 + §10.2 state evidence.

Renders the running board at every size (``gui.screenshot_sizes`` = 2x2/3x3/4x4/5x5) with cop +
thief + >=1 barrier visible (the mandatory §7.3c grid-size matrix), PLUS the two distinct GUI
STATES §10.2 (Nielsen) needs beyond "running" — the **view-radius overlay** (the ``v`` toggle) and
the **terminal winner-banner** — and saves the PNGs under ``gui.screenshot_dir``. Headless via
pygame-ce (``SDL_VIDEODRIVER=dummy``). Run: ``uv run --extra gui python scripts/capture_screens.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.sdk.sdk import MarlSDK
from src.utils.config_loader import load_config


def main(cfg: dict | None = None) -> list[str]:  # pragma: no cover - requires pygame
    """Render + save the running grid-size matrix + the two distinct GUI states; return the paths."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame  # noqa: PLC0415 - lazy: pygame is the optional gui extra

    from src.gui.render import render_frame  # noqa: PLC0415 - lazy with pygame

    cfg = cfg or load_config()
    out_dir = Path(cfg["gui"]["screenshot_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pygame.init()
    sdk = MarlSDK(cfg)
    font = pygame.font.SysFont(None, 24)
    saved: list[str] = []

    def _shot(frame: object, name: str, show_radius: bool = False) -> None:
        surface = pygame.Surface((720, 560))
        render_frame(surface, font, frame, show_radius)
        path = out_dir / name
        pygame.image.save(surface, str(path))
        saved.append(str(path))

    def _running(size: int) -> object:  # a mid-run frame (3 heuristic moves in)
        session = sdk.spectator_session(size, size, num_cops=1, seed=7)
        frame = session.reset()
        for _ in range(3):
            frame = session.step()
        return session, frame

    for size in cfg["gui"]["screenshot_sizes"]:  # §7.3c: the running board at each grid size
        _shot(_running(size)[1], f"grid_{size}x{size}.png")

    # §10.2: the two distinct STATES beyond "running" (captured at the graded 5x5 stage).
    session, frame = _running(5)
    _shot(frame, "state_view_radius.png", show_radius=True)  # the 'v' overlay state
    while frame.winner is None and frame.move < frame.max_moves:
        frame = session.step()
    _shot(frame, "state_terminal.png")  # the terminal winner-banner state
    pygame.quit()
    return saved


if __name__ == "__main__":
    main()
