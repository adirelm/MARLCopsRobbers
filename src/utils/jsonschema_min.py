"""Minimal draft-2020-12 JSON-schema validator — no external dependency (T6.3).

Supports the subset our committed schemas use: ``type`` / ``required`` / ``enum`` /
``additionalProperties`` / ``properties`` / ``items`` / ``minItems`` / ``maxItems``.
Raises :class:`ValueError` on the first violation. It is the SINGLE validator used
by the §3.5 report build AND the sub-game schema tests (no pydantic runtime dep for
the report, per the §3.5 contract).
"""

from __future__ import annotations

_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "boolean": bool,
    "number": (int, float),
}


def _check_type(value: object, type_name: str) -> bool:
    """Return whether ``value`` matches a draft-2020-12 primitive type name."""
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    return isinstance(value, _TYPES[type_name])


def _validate_dict(instance: dict, schema: dict) -> None:
    """Check ``additionalProperties:false`` + recurse into declared properties."""
    props = schema.get("properties", {})
    if schema.get("additionalProperties") is False:
        extra = set(instance) - set(props)
        if extra:
            raise ValueError(f"unexpected keys {sorted(extra)}")
    for key, sub in props.items():
        if key in instance:
            validate(instance[key], sub)


def _validate_list(instance: list, schema: dict) -> None:
    """Check ``minItems`` / ``maxItems`` + recurse into each item."""
    if len(instance) < schema.get("minItems", 0):
        raise ValueError(f"too few items: {len(instance)} < {schema['minItems']}")
    if "maxItems" in schema and len(instance) > schema["maxItems"]:
        raise ValueError(f"too many items: {len(instance)} > {schema['maxItems']}")
    for item in instance:
        validate(item, schema["items"])


def validate(instance: object, schema: dict) -> None:
    """Validate ``instance`` against ``schema``; raise ``ValueError`` on any violation.

    Args:
        instance: The parsed JSON value to check.
        schema: A draft-2020-12 schema (the supported keyword subset).

    Raises:
        ValueError: On the first type/required/enum/extra-key/length violation.
    """
    if "type" in schema and not _check_type(instance, schema["type"]):
        raise ValueError(f"type mismatch: expected {schema['type']}, got {type(instance).__name__}")
    if "enum" in schema and instance not in schema["enum"]:
        raise ValueError(f"{instance!r} not in enum {schema['enum']}")
    for key in schema.get("required", []):
        if not isinstance(instance, dict) or key not in instance:
            raise ValueError(f"missing required key {key!r}")
    if isinstance(instance, dict):
        _validate_dict(instance, schema)
    if isinstance(instance, list) and "items" in schema:
        _validate_list(instance, schema)
