"""CTDE learner package (T4.2, P4b).

Holds the centralized BPTT learners (VDN/QMIX cop team, IQL baseline, thief
adversary) and their shared pure-tensor helpers. The learner owns all masking
(``active``/``filled``) so the mixers stay mask-unaware (the P4a pure-net
contract). Global state lives only here and in replay — it never crosses the
MCP boundary.
"""

# V3 §14 public surface. Submodule names (not eagerly imported symbols): the package is a
# module container, so `from ... import *` binds the modules and heavy/optional deps stay
# off the import path. Private `_*` helpers and CLI/deploy entrypoints are deliberately out.
__all__ = [
    "learner_base",
    "learners",
]
