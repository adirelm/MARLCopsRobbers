"""MARL package — CTDE QMIX/VDN/IQL agents, env, nets, mixers, and replay."""

# V3 §14 public surface. Submodule names (not eagerly imported symbols): the package is a
# module container, so `from ... import *` binds the modules and heavy/optional deps stay
# off the import path. Private `_*` helpers and CLI/deploy entrypoints are deliberately out.
__all__ = [
    "baselines",
    "data",
    "env",
    "learner",
    "mixers",
    "nets",
    "olora_bundle",
    "replay",
]
