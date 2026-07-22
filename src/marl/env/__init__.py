"""Environment package — the custom CopsRobbersEnv Dec-POMDP/POSG and actions."""

# V3 §14 public surface. Submodule names (not eagerly imported symbols): the package is a
# module container, so `from ... import *` binds the modules and heavy/optional deps stay
# off the import path. Private `_*` helpers and CLI/deploy entrypoints are deliberately out.
__all__ = [
    "actions",
    "cops_robbers_env",
    "curriculum",
    "grid",
    "observation",
    "observation_encoder",
    "render_state",
    "reward",
    "scorer",
    "transition",
    "types",
]
