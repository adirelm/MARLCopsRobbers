"""CTDE learner package (T4.2, P4b).

Holds the centralized BPTT learners (VDN/QMIX cop team, IQL baseline, thief
adversary) and their shared pure-tensor helpers. The learner owns all masking
(``active``/``filled``) so the mixers stay mask-unaware (the P4a pure-net
contract). Global state lives only here and in replay — it never crosses the
MCP boundary.
"""

from __future__ import annotations
