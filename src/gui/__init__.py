"""GUI package — the Pygame god-view spectator and its local palette."""

# V3 §14 public surface. Submodule names (not eagerly imported symbols): the package is a
# module container, so `from ... import *` binds the modules and heavy/optional deps stay
# off the import path. Private `_*` helpers and CLI/deploy entrypoints are deliberately out.
__all__ = [
    "draw_board",
    "draw_plan",
    "effects",
    "input_map",
    "palette",
    "render",
    "spectator",
    "sprites",
    "state_client",
    "transform",
]
