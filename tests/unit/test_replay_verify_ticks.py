"""Per-tick position verification — the half of the §9.3 integrity claim nothing pinned.

The module advertises that divergence is "caught at the tick it happens — not only when it
survives to the terminal summary". The MASKING half had tests; the POSITION half did not.
If the `your_pos` / `barriers_left` comparison were wrong, every replay would still print
OK for all six sub-games and no test would notice — which is exactly the kind of guard that
looks like evidence while proving nothing.
"""

from __future__ import annotations

import pytest

from src.mcp._replay_log import ReplayMismatchError
from src.mcp._replay_verify import verify_tick
from src.mcp.wire_referee import mask_payload
from src.utils.config_loader import load_config


class _State:
    """Minimal stand-in for the replayed env state verify_tick reads."""

    def __init__(self, cop=(1, 1), thief=(3, 3), barriers=(), used=0):
        self.cop_pos, self.thief_pos = [list(cop)], list(thief)
        self.barriers, self.barriers_used = list(barriers), used
        self.h = self.w = 5  # the graded board; the tests never need another size


def _session(cfg, state, tick=0, **override):
    """A logged tick that HONESTLY matches ``state`` (override to forge a divergence)."""
    left = int(cfg["game"]["max_barriers"]) - state.barriers_used
    radius = int(cfg["env"]["view_radius_by_grid"][min(state.h, state.w)])
    pos = {"cop": tuple(state.cop_pos[0]), "thief": tuple(state.thief_pos)}
    states = {}
    for role in ("cop", "thief"):
        other = "thief" if role == "cop" else "cop"
        payload = mask_payload("sg-0", tick, pos[role], pos[other], state.barriers, left, radius)
        states[role] = {
            "your_pos": pos[role],
            "barriers_left": payload["barriers_left"],
            "opponent_pos": payload["opponent_pos"],
            "barriers": payload["barriers"],
            **override.get(role, {}),
        }
    return {"states": {tick: states}}


def test_an_honest_tick_verifies() -> None:
    """Baseline — otherwise the negative cases below could pass for the wrong reason."""
    cfg, state = load_config(), _State()
    verify_tick(cfg, "sg-0", 0, _session(cfg, state), state)


def test_a_shifted_position_is_caught_at_that_tick() -> None:
    """Forge one cell of drift: the raise must name the tick, not surface at the summary."""
    cfg, state = load_config(), _State()
    sess = _session(cfg, state, cop={"your_pos": (4, 4)})
    with pytest.raises(ReplayMismatchError, match="tick 0 cop"):
        verify_tick(cfg, "sg-0", 0, sess, state)


def test_a_wrong_barrier_budget_is_caught() -> None:
    """The other half of the position check — a forged budget must not slip through."""
    cfg, state = load_config(), _State()
    sess = _session(cfg, state, thief={"barriers_left": 99})
    with pytest.raises(ReplayMismatchError, match="tick 0 thief"):
        verify_tick(cfg, "sg-0", 0, sess, state)


def test_a_missing_role_payload_is_caught() -> None:
    """A log that simply omits a role's request for a tick is unverifiable, not 'fine'."""
    cfg, state = load_config(), _State()
    sess = _session(cfg, state)
    del sess["states"][0]["thief"]
    with pytest.raises(ReplayMismatchError, match="no logged request payload for thief"):
        verify_tick(cfg, "sg-0", 0, sess, state)
