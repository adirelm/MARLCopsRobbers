"""Tabular game-theory baselines (L11 §5 bonus): Minimax-Q + its maximin LP."""

# V3 §14 public surface. Submodule names (not eagerly imported symbols): the package is a
# module container, so `from ... import *` binds the modules and heavy/optional deps stay
# off the import path. Private `_*` helpers and CLI/deploy entrypoints are deliberately out.
__all__ = [
    "minimax_lp",
    "minimax_q",
    "minimax_runner",
    "tabular_pursuit",
]
