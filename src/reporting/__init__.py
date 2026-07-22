"""Reporting package — the end-of-game Gmail report builder and sender."""

# V3 §14 public surface. Submodule names (not eagerly imported symbols): the package is a
# module container, so `from ... import *` binds the modules and heavy/optional deps stay
# off the import path. Private `_*` helpers and CLI/deploy entrypoints are deliberately out.
__all__ = [
    "bonus",
    "bonus_send",
    "mailer",
    "players",
    "schema",
    "send",
]
