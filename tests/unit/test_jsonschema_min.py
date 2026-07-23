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


def test_number_bounds_and_bool_exemption():
    """minimum/maximum bound numbers; bools are NOT numbers for the bounds check."""
    validate(3, {"type": "integer", "minimum": 1, "maximum": 6})
    with pytest.raises(ValueError, match="minimum"):
        validate(0, {"minimum": 1})
    with pytest.raises(ValueError, match="maximum"):
        validate(26, {"maximum": 25})
    validate(True, {"minimum": 5})  # a bool never trips the numeric bounds


def test_string_pattern_and_min_length():
    """pattern/minLength constrain strings (the F5 schema-tightening keywords)."""
    iso = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?(Z|[+-]\d{2}:\d{2})$"
    validate("2026-07-23T02:20:07.170+03:00", {"type": "string", "pattern": iso})
    validate("2026-07-04T18:00:00+03:00", {"type": "string", "pattern": iso})
    with pytest.raises(ValueError, match="pattern"):
        validate("not-a-timestamp", {"pattern": iso})
    with pytest.raises(ValueError, match="minLength"):
        validate("", {"minLength": 1})
