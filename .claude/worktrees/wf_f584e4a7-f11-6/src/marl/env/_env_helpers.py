"""Per-agent dict builders + visibility-memory updates for CopsRobbersEnv (T2.3).

These pure helpers keep ``cops_robbers_env.py`` <=150 LOC. They build the
per-agent (keyed ``cop_0..`` + ``thief``) LOCAL observation, action-mask, and
reward dicts and update the env-owned per-AGENT :class:`VisibilityMemory`
(``steps_since_seen`` reset to 0 when ANY of that agent's opponents is in view,
else +1). The thief tracks the WHOLE cop team; each cop tracks the thief. The
agent-key convention here MUST match RewardModel.compute / action_mask so every
downstream stage integrates without translation.
"""

from __future__ import annotations

from random import Random

from src.marl.env.actions import action_mask
from src.marl.env.grid import sample_spawn
from src.marl.env.observation import (
    VisibilityMemory,
    build_observation,
    opponent_in_view,
    view_radius,
)
from src.marl.env.types import GlobalState, Observation, Pos


def agent_keys(num_cops: int) -> list[str]:
    """Return the ordered agent keys ``[cop_0.., thief]`` for ``num_cops`` cops."""
    return [f"cop_{i}" for i in range(num_cops)] + ["thief"]


def sample_positions(rng: Random, h: int, w: int, radius: int, num_cops: int) -> tuple[tuple[Pos, ...], Pos]:
    """Sample ``num_cops`` cop cells + a thief cell with spawn dist > radius.

    The lead cop/thief pair satisfies ``manhattan > radius`` via
    ``grid.sample_spawn``; extra cops (the 4x4 stage) are sampled distinct from
    the thief — inter-cop double-occupancy is allowed (cooperative team).
    """
    lead_cop, thief = sample_spawn(rng, h, w, radius)
    cops: list[Pos] = [lead_cop]
    cells = [(r, c) for r in range(h) for c in range(w) if (r, c) != thief]
    while len(cops) < num_cops:
        cops.append(rng.choice(cells))
    return tuple(cops), thief


def fresh_memory(num_cops: int) -> dict[str, VisibilityMemory]:
    """Return a reset per-AGENT visibility memory (``steps_since_seen = 0``).

    Keyed by agent key (``cop_0..`` + ``thief``) so each agent owns its own
    ``steps_since_seen`` — the thief no longer shares the cop's counter.
    """
    return {key: VisibilityMemory(steps_since_seen=0) for key in agent_keys(num_cops)}


def _agent_sees_opponent(state: GlobalState, agent_key: str, cfg: dict) -> bool:
    """Return whether ``agent_key`` sees ANY opponent from its OWN center.

    A ``cop_{i}`` checks the thief against ``cop_pos[i]``; the ``thief`` checks
    EVERY cop against ``thief_pos`` (it sees the whole team).
    """
    radius = view_radius(state.h, state.w, cfg)
    if agent_key == "thief":
        return any(opponent_in_view(state.thief_pos, cop, radius) for cop in state.cop_pos)
    idx = int(agent_key.split("_", 1)[1])
    return opponent_in_view(state.cop_pos[idx], state.thief_pos, radius)


def update_memory(memory: dict[str, VisibilityMemory], state: GlobalState, cfg: dict) -> None:
    """Update each agent's ``steps_since_seen`` from the post-move ``state``.

    For EACH agent key, recompute "any opponent in view" from that agent's OWN
    center (cop ``i`` uses ``cop_pos[i]``; the thief uses every cop): reset its
    own counter to 0 when an opponent is in view this tick, else increment by one.
    """
    for agent_key, mem in memory.items():
        if _agent_sees_opponent(state, agent_key, cfg):
            mem.steps_since_seen = 0
        else:
            mem.steps_since_seen += 1


def build_obs_dict(
    state: GlobalState, memory: dict[str, VisibilityMemory], num_cops: int, cfg: dict
) -> dict[str, Observation]:
    """Build the per-agent LOCAL observation dict keyed ``cop_0.. + thief``.

    Each ``cop_{i}`` obs uses its own per-cop memory and sees only the thief; the
    ``thief`` obs uses its own memory and sees the WHOLE cop team.
    """
    return {key: build_observation(state, key, memory, cfg) for key in agent_keys(num_cops)}


def build_mask_dict(state: GlobalState, num_cops: int, cfg: dict) -> dict[str, object]:
    """Build the per-agent action-mask dict keyed ``cop_0.. + thief``."""
    masks: dict[str, object] = {}
    for i in range(num_cops):
        masks[f"cop_{i}"] = action_mask(state, "cop", cfg, idx=i)
    masks["thief"] = action_mask(state, "thief", cfg, idx=0)
    return masks
