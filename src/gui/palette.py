"""GUI palette + local styling constants (T7.3).

LOCAL visual-design literals ONLY — colours, cell/token geometry, frame rate, and
animation timing. These are part of the rendering design, NOT tunable algorithm
parameters, so they live here and NOT in config (CLAUDE.md §4 local-styling rule);
a test asserts this module imports nothing from config. RGB triples are 0-255.
"""

from __future__ import annotations

BG = (12, 12, 18)
GRID_LINE = (46, 50, 74)
CHECKER = (22, 22, 30)
# Cop/thief stay recognisably BLUE and RED — the HUD legend and the README/UX captions
# name those hues, so a shift to e.g. cyan/magenta would make committed prose false.
# They are only pushed brighter, for the neon read against the darker background.
COP = (90, 180, 255)
THIEF = (255, 80, 110)
BARRIER = (150, 150, 168)
CAPTURE_FLASH = (255, 230, 120)
VIEW_RADIUS = (70, 190, 235)
TEXT = (232, 232, 240)
EYE_WHITE = (240, 244, 255)
EYE_PUPIL = (30, 34, 60)
HUD_PANEL = (16, 16, 24)
HUD_RULE = (52, 56, 82)

# Geometry / timing (pixels / fps / ms).
GRID_W = 1
TOKEN_INSET = 6
CELL_PX_CAP = 96
FPS = 30
MOVE_ANIM_MS = 180
FONT_PX = 18

# Window size — the SINGLE source. It used to be spelled 720/560 in both
# ``render.run_app``'s defaults and ``scripts/capture_screens.py``, so a resize would have
# silently desynced the live app from the committed §7.3 screenshots.
WINDOW_W = 720
WINDOW_H = 560

# Breathing room under the board. Without it the letterbox consumed every pixel below the
# HUD and the board's bottom edge sat flush against the window frame.
BOARD_MARGIN_PX = 14

# Alpha levels (0-255) for the radar/neon layers. Kept low: every one of these is CONTEXT
# behind the tokens, and anything that competes with them hurts the thing being watched.
HALO_ALPHA = 46  # the cops' Manhattan knowledge disk
GHOST_ALPHA = 105  # the thief while OUTSIDE that disk (drawn, but visibly unknown)
TRAIL_ALPHA = 110  # trail CEILING: each dot gets a fraction of this, so none reaches it
SHOCKWAVE_ALPHA = 150  # capture rings

TRAIL_LEN = 4  # cells retained per agent (including the current one)
TRAIL_SCALE = 0.62  # trail size CEILING as a fraction of a token; each dot scales below it
SHOCKWAVE_RINGS = 3  # concentric rings drawn on the capturing cop
