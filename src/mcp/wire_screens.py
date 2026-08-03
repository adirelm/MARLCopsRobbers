"""§9.3 replay rendering — pick evidence frames and save the god-view PNGs.

Split out of :mod:`src.mcp.wire_replay` for the 150-LOC cap (the replay module still
re-exports both helpers, so existing imports keep working). Rendering calls INTO the
god-view GUI path headlessly (``src.gui.render.render_frame``; src/gui gains no imports).
"""

from __future__ import annotations

import os
from pathlib import Path


def mid_frame_index(frames: list, radius: int) -> int:
    """Pick the most informative mid frame: first barrier, else first mutual visibility, else middle."""
    inner = range(1, max(len(frames) - 1, 1))
    for i in inner:
        if frames[i].barriers:
            return i
    for i in inner:
        (cr, cc), (tr, tc) = frames[i].cop_positions[0], frames[i].thief_position
        if abs(cr - tr) + abs(cc - tc) <= radius:
            return i
    return len(frames) // 2


def save_screens(cfg: dict, replays: list[dict], out_dir: str | Path | None = None) -> list[Path]:
    """Render t00/mid/final PNGs per sub-game via the EXISTING GUI path (headless pygame)."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame  # noqa: PLC0415 - lazy: pygame is the optional gui extra

    from src.gui import palette  # noqa: PLC0415 - lazy with pygame
    from src.gui.render import render_frame  # noqa: PLC0415 - lazy with pygame

    out = Path(cfg["gui"]["bonus_screenshot_dir"] if out_dir is None else out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pygame.init()
    font, saved = pygame.font.Font(None, 24), []  # bundled font -> machine-independent PNGs
    for game in replays:
        frames = game["frames"]
        mid = mid_frame_index(frames, int(cfg["mcp"]["observation"]["view_radius"]))
        picks = (("t00", 0), ("mid", mid), ("final", len(frames) - 1))
        for tag, idx in picks:
            surface = pygame.Surface((palette.WINDOW_W, palette.WINDOW_H))
            render_frame(surface, font, frames[idx])
            path = out / f"bonus_sg{game['gid']}_{tag}.png"
            pygame.image.save(surface, str(path))
            saved.append(path)
    pygame.quit()
    return saved
