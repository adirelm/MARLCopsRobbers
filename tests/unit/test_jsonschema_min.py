"""Edge-case tests for the minimal JSON-schema validator (T6.3).

Covers the array length bounds, required-on-non-dict, type/number, and clean
pass-through paths that the report/sub-game schemas don't all exercise.
"""

from __future__ import annotations

import pytest

from src.utils.jsonschema_min import validate

_LIST_SCHEMA = {"type": "array", "minItems": 1, "maxItems": 2, "items": {"type": "integer"}}


def test_accepts_a_valid_instance():
    """A valid number + a bounded list pass without raising."""
    validate(3, {"type": "number"})
    validate([1, 2], _LIST_SCHEMA)


def test_rejects_too_few_items():
    """An array below minItems is rejected."""
    with pytest.raises(ValueError, match="too few"):
        validate([], _LIST_SCHEMA)


def test_rejects_too_many_items():
    """An array above maxItems is rejected."""
    with pytest.raises(ValueError, match="too many"):
        validate([1, 2, 3], _LIST_SCHEMA)


def test_required_on_non_dict_is_rejected():
    """A required key on a non-object instance is rejected."""
    with pytest.raises(ValueError, match="required"):
        validate("not-an-object", {"required": ["x"]})


def test_item_type_violation_is_rejected():
    """A wrongly-typed list element is rejected (recursion into items)."""
    with pytest.raises(ValueError, match="type mismatch"):
        validate([1, "two"], _LIST_SCHEMA)
