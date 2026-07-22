"""Services package — referee/orchestration services and the API gatekeeper."""

# V3 §14 public surface. Submodule names (not eagerly imported symbols): the package is a
# module container, so `from ... import *` binds the modules and heavy/optional deps stay
# off the import path. Private `_*` helpers and CLI/deploy entrypoints are deliberately out.
__all__ = [
    "bonus_policies",
    "checkpoints",
    "episode_pad",
    "finetune",
    "heuristic_policy",
    "pipelines",
    "policy",
    "rollout",
    "selfplay",
    "spectator",
    "sweep",
    "trainer",
]
