"""P3 end-to-end pipeline integration (T3.3, via the single SDK entry).

Drives the whole P3 smoke THROUGH ``MarlSDK``: collect (cop=tabular learner vs
thief=heuristic) -> tabular-train on the 2x2 stage -> export. Asserts the run is
NaN-free and reaches 2x2 optimal capture for two seeds; that a minimal sub-game
JSON is written + validates against ``docs/schema/subgame.schema.json``; and that
a headless god-view ``render_state()`` dict is produced carrying NO GlobalState
(the sanctioned spectator seam, never the train-only type across the boundary).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.marl.env.types import GlobalState
from src.sdk.sdk import MarlSDK

_SCHEMA = Path(__file__).resolve().parents[2] / "docs" / "schema" / "subgame.schema.json"

_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "boolean": bool,
}


def _check_type(value, type_name: str) -> bool:
    """Return whether ``value`` matches a draft-2020-12 primitive type name."""
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    return isinstance(value, _TYPES[type_name])


def _validate(instance, schema) -> None:
    """Minimal draft-2020-12 validator: required/type/enum/additionalProperties."""
    if "type" in schema:
        assert _check_type(instance, schema["type"]), f"type mismatch: {schema['type']}"
    for key in schema.get("required", []):
        assert key in instance, f"missing required key {key!r}"
    if "enum" in schema:
        assert instance in schema["enum"], f"{instance!r} not in {schema['enum']}"
    props = schema.get("properties", {})
    if isinstance(instance, dict):
        if schema.get("additionalProperties") is False:
            extra = set(instance) - set(props)
            assert not extra, f"unexpected keys {extra}"
        for key, sub in props.items():
            if key in instance:
                _validate(instance[key], sub)
    if isinstance(instance, list) and "items" in schema:
        if "minItems" in schema:
            assert len(instance) >= schema["minItems"]
        if "maxItems" in schema:
            assert len(instance) <= schema["maxItems"]
        for item in instance:
            _validate(item, schema["items"])


def _load_schema() -> dict:
    """Load the draft-2020-12 sub-game schema (single source of truth)."""
    return json.loads(_SCHEMA.read_text(encoding="utf-8"))


def _contains_globalstate(value) -> bool:
    """Recursively report whether a GlobalState instance hides anywhere inside."""
    if isinstance(value, GlobalState):
        return True
    if isinstance(value, dict):
        return any(_contains_globalstate(v) for v in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_globalstate(v) for v in value)
    return False


def test_run_p3_smoke_collect_train_export_nan_free(cfg):
    """run_p3_smoke over two seeds is NaN-free and reaches 2x2 optimal capture."""
    seeds = cfg["training"]["seeds"][:2]
    result = MarlSDK(cfg).run_p3_smoke(seeds)
    assert result["optimal"] is True
    assert result["nan_free"] is True
    assert result["all_captured"] is True
    assert result["seeds"] == list(seeds)


def test_run_p3_smoke_writes_validating_subgame_json(cfg, tmp_path):
    """run_p3_smoke exports a sub-game JSON that validates vs the schema."""
    sdk = MarlSDK(cfg)
    result = sdk.run_p3_smoke(cfg["training"]["seeds"][:2], out_dir=tmp_path)
    sub_path = Path(result["subgame_path"])
    assert sub_path.exists()
    record = json.loads(sub_path.read_text(encoding="utf-8"))
    _validate(record, _load_schema())  # raises AssertionError on any violation
    assert record["grid"] == [2, 2]
    assert record["winner"] == "cop"
    assert record["capture"] is True


_GOOD_RECORD = {
    "game_id": "p3-smoke-7",
    "grid": [2, 2],
    "winner": "cop",
    "capture": True,
    "steps": 1,
    "scores": {"cop": 20, "thief": 5},
    "seed": 7,
}


@pytest.mark.parametrize(
    ("mutate", "why"),
    [
        (lambda r: {**r, "winner": "burglar"}, "bad enum value"),
        (lambda r: {k: v for k, v in r.items() if k != "seed"}, "missing required key"),
        (lambda r: {**r, "extra": 1}, "extra top-level key (additionalProperties:false)"),
        (lambda r: {**r, "scores": {**r["scores"], "ref": 0}}, "extra nested key"),
        (lambda r: {**r, "capture": "yes"}, "wrong primitive type"),
    ],
)
def test_validator_rejects_malformed_subgame(mutate, why):
    """The self-contained validator REJECTS bad enum / missing / extra-key records.

    Proves the validator is not vacuous: a known-good record passes, and every
    single-field corruption (including additionalProperties:false at both the top
    level and a nested object) raises ``AssertionError`` (``why`` labels each case).
    """
    assert why  # the human-readable rejection reason (also the parametrize id)
    schema = _load_schema()
    _validate(_GOOD_RECORD, schema)  # sanity: the good record still passes
    with pytest.raises(AssertionError):
        _validate(mutate(copy.deepcopy(_GOOD_RECORD)), schema)


def test_run_p3_smoke_emits_godview_render_state(cfg, tmp_path):
    """run_p3_smoke produces a headless god-view dict with NO GlobalState anywhere."""
    result = MarlSDK(cfg).run_p3_smoke(cfg["training"]["seeds"][:2], out_dir=tmp_path)
    god = result["render_state"]
    assert isinstance(god, dict)
    # The spectator dict carries plain JSON-serializable board fields only.
    assert {"h", "w", "cop_positions", "thief_position", "barriers"} <= set(god)
    # RECURSIVE no-leak scan: no GlobalState field/instance nested anywhere.
    assert not _contains_globalstate(god)
    json.dumps(god)  # must be JSON-serializable (no GlobalState / numpy leaks)
    # No raw GlobalState dataclass field name is exposed as a god-view key.
    assert "cop_pos" not in god and "thief_pos" not in god
