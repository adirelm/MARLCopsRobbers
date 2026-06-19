"""P1 train/exec split exit-gate test (architect decision #3).

This is the TYPE-LEVEL seed of the full P2 runtime leak test (T2.3): at P1 we
prove, at the type boundary, that the EXEC-TIME ``Observation`` exposes ONLY
local fields ({image, scalars}) while the TRAIN-TIME ``GlobalState`` is a
distinct, richer type carrying the global/train-only fields (barriers, cop_pos,
thief_pos, h/w, step) that must NEVER cross to an executor or the MCP boundary.
The P2 test promotes this static guarantee into a full-episode runtime check.
"""

from __future__ import annotations

import dataclasses

from src.marl.env.types import GlobalState, Observation

LOCAL_OBS_FIELDS = {"image", "scalars"}
# Derive the forbidden set from the dataclass itself (every GlobalState field
# that is NOT one of the two local exec fields) so a newly-added train-only
# field (e.g. barriers_used, terminal) is guarded automatically — never a
# hand-maintained list that can silently drift out of date.
GLOBAL_ONLY_FIELDS = {f.name for f in dataclasses.fields(GlobalState)} - LOCAL_OBS_FIELDS


def test_observation_annotations_exactly_local_fields():
    """Observation exposes EXACTLY {image, scalars} (decision #3: ONLY those)."""
    keys = set(Observation.__annotations__.keys())
    assert keys == LOCAL_OBS_FIELDS


def test_observation_carries_no_global_fields():
    """Observation must expose NO global/train-only field (the exec leak guard)."""
    keys = set(Observation.__annotations__.keys())
    for forbidden in GLOBAL_ONLY_FIELDS:
        assert forbidden not in keys


def test_global_state_has_barriers_cop_thief():
    """GlobalState HAS barriers/cop_pos/thief_pos — the richer train-only type."""
    fields = {f.name for f in dataclasses.fields(GlobalState)}
    for required in ("barriers", "cop_pos", "thief_pos"):
        assert required in fields


def test_global_state_and_observation_are_distinct_types():
    """The train-time and exec-time types do not share their field sets."""
    state_fields = {f.name for f in dataclasses.fields(GlobalState)}
    obs_keys = set(Observation.__annotations__.keys())
    assert state_fields.isdisjoint(obs_keys)
