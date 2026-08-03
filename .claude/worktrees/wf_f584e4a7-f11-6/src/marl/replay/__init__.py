"""Replay package — the centralized episodic replay buffer for CTDE (BPTT)."""

# V3 §14 public surface. Submodule names (not eagerly imported symbols): the package is a
# module container, so `from ... import *` binds the modules and heavy/optional deps stay
# off the import path. Private `_*` helpers and CLI/deploy entrypoints are deliberately out.
__all__ = [
    "episode_buffer",
]
