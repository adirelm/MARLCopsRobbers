"""GUI package — the Pygame god-view spectator and its local palette."""

# V3 §14 public surface. Submodule names (not eagerly imported symbols): the package is a
# module container, so `from ... import *` binds the modules and heavy/optional deps stay
# off the import path. Private `_*` helpers and CLI/deploy entrypoints are deliberately out.
__all__ = [
    "draw_plan",
    "input_map",
    "palette",
    "render",
    "spectator",
    "state_client",
    "transform",
]
