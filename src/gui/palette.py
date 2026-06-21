"""GUI palette + local styling constants (T7.3).

LOCAL visual-design literals ONLY — colours, cell/token geometry, frame rate, and
animation timing. These are part of the rendering design, NOT tunable algorithm
parameters, so they live here and NOT in config (CLAUDE.md §4 local-styling rule);
a test asserts this module imports nothing from config. RGB triples are 0-255.
"""

from __future__ import annotations

# Colours (R, G, B), 0-255.
BG = (18, 18, 24)
GRID_LINE = (60, 60, 72)
CHECKER = (26, 26, 34)
COP = (80, 160, 255)
THIEF = (255, 90, 90)
BARRIER = (140, 140, 150)
CAPTURE_FLASH = (255, 230, 120)
VIEW_RADIUS = (70, 120, 90)
TEXT = (230, 230, 235)

# Geometry / timing (pixels / fps / ms).
GRID_W = 1
TOKEN_INSET = 6
CELL_PX_CAP = 96
FPS = 30
MOVE_ANIM_MS = 180
FONT_PX = 18
