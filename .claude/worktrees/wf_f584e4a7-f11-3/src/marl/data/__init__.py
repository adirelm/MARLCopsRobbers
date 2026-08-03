"""Data package — behavior-cloning rollouts and supervised (o, a*) pair sources."""

# V3 §14 public surface. Submodule names (not eagerly imported symbols): the package is a
# module container, so `from ... import *` binds the modules and heavy/optional deps stay
# off the import path. Private `_*` helpers and CLI/deploy entrypoints are deliberately out.
__all__ = [
    "bc_dataset",
    "bc_npz",
    "bc_split",
    "heuristics",
    "obs_encoder",
    "schemas",
]
