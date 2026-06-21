"""Sub-game artifact integration: schema validation + god-view no-leak.

The durable P3 deliverables (kept after the throwaway tabular smoke was removed in
P4): a minimal sub-game record written via ``MarlSDK.write_subgame_json`` validates
against ``docs/schema/subgame.schema.json``; a self-contained validator REJECTS
malformed records (non-vacuous); and the headless god-view ``render_state()`` dict
carries NO :class:`GlobalState` anywhere (the sanctioned spectator seam). The real
sub-game producer is the CTDE referee built in the MCP phase (P5).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.marl.env.render_state import render_state
from src.marl.env.types import GlobalState
from src.sdk.sdk import MarlSDK
from src.utils.jsonschema_min import validate as _validate

_SCHEMA = Path(__file__).resolve().parents[2] / "docs" / "schema" / "subgame.schema.json"

_GOOD_RECORD = {
    "game_id": "subgame-7",
    "grid": [2, 2],
    "winner": "cop",
    "capture": True,
    "steps": 1,
    "scores": {"cop": 20, "thief": 5},
    "seed": 7,
}


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


def test_sdk_write_subgame_json_validates_against_schema(cfg, tmp_path):
    """A record written via the SDK reads back and validates against the schema."""
    sdk = MarlSDK(cfg)
    path = tmp_path / "subgames" / "g.json"
    sdk.write_subgame_json(_GOOD_RECORD, path)
    record = json.loads(path.read_text(encoding="utf-8"))
    _validate(record, _load_schema())
    assert record == _GOOD_RECORD


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
    """The validator is non-vacuous: a good record passes; each corruption raises."""
    assert why
    schema = _load_schema()
    _validate(_GOOD_RECORD, schema)
    with pytest.raises(ValueError):
        _validate(mutate(copy.deepcopy(_GOOD_RECORD)), schema)


def test_render_state_godview_has_no_globalstate(cfg):
    """The headless god-view dict is JSON-serializable with NO GlobalState leak."""
    env = MarlSDK(cfg).build_env(h=2, w=2, num_cops=1)
    env.reset(seed=1)
    god = render_state(env.state(), cfg)
    assert {"h", "w", "cop_positions", "thief_position", "barriers"} <= set(god)
    assert not _contains_globalstate(god)
    json.dumps(god)  # must serialize (no GlobalState / numpy leaks)
    assert "cop_pos" not in god and "thief_pos" not in god
