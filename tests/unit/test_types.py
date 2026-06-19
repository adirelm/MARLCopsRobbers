"""Unit tests for src.marl.env.types (T1.3 + CTDE split seed)."""

from __future__ import annotations

import dataclasses

import pytest

from src.marl.env.types import GlobalState, Observation


def test_global_state_has_h_w():
    """GlobalState carries explicit board dims h and w (architect decision #1)."""
    fields = {f.name for f in dataclasses.fields(GlobalState)}
    assert "h" in fields
    assert "w" in fields


def test_global_state_has_global_fields():
    """GlobalState exposes the train-only global fields."""
    fields = {f.name for f in dataclasses.fields(GlobalState)}
    for name in ("cop_pos", "thief_pos", "barriers", "barriers_used", "step"):
        assert name in fields


def test_global_state_is_frozen(make_state):
    """GlobalState is frozen (immutable) and hashable."""
    state = make_state()
    assert dataclasses.is_dataclass(state)
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.step = 99  # type: ignore[misc]
    hash(state)  # must not raise


def test_cop_pos_is_tuple_of_pos(make_state):
    """cop_pos is a tuple of positions (hashable for frozen dataclass)."""
    single = make_state(cop_pos=(1, 1))
    assert single.cop_pos == ((1, 1),)
    two = make_state(cop_pos=[(0, 0), (4, 4)])
    assert two.cop_pos == ((0, 0), (4, 4))


def test_observation_annotations_are_local_only():
    """Observation exposes ONLY local fields {image, scalars} (decision #3)."""
    assert set(Observation.__annotations__.keys()) == {"image", "scalars"}


def test_observation_excludes_global_fields():
    """Observation must NOT leak any global/train-only field."""
    keys = set(Observation.__annotations__.keys())
    for forbidden in ("barriers", "thief_pos", "cop_pos", "h", "w", "step"):
        assert forbidden not in keys
