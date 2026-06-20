"""Unit tests for src.marl.data.schemas (T3.2).

Covers the SourceTag provenance enum members + values and the frozen
TransitionDTO / DatasetManifest interchange records.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.marl.data.schemas import DatasetManifest, SourceTag, TransitionDTO


def test_source_tag_members_and_values():
    """SourceTag exposes the four pinned provenance members as string values."""
    assert SourceTag.EXPERT == "expert"
    assert SourceTag.SELF_PLAY == "self_play"
    assert SourceTag.RANDOM == "random"
    assert SourceTag.LIVE_CTDE == "live_ctde"
    assert {t.value for t in SourceTag} == {"expert", "self_play", "random", "live_ctde"}


def test_source_tag_is_str():
    """SourceTag is a StrEnum: a member IS its string value (npz/json friendly)."""
    assert isinstance(SourceTag.EXPERT, str)
    assert SourceTag("self_play") is SourceTag.SELF_PLAY


def _make_dto() -> TransitionDTO:
    return TransitionDTO(
        obs=np.zeros((5, 5, 5), dtype=np.float32),
        scalars=np.zeros((6,), dtype=np.float32),
        global_state=np.zeros((77,), dtype=np.float32),
        action=2,
        reward=1.0,
        done=False,
        next_legal_mask=np.ones((5,), dtype=np.bool_),
    )


def test_transition_dto_fields_and_frozen():
    """TransitionDTO carries the pinned fields and is immutable (frozen)."""
    dto = _make_dto()
    assert dto.action == 2
    assert dto.reward == 1.0
    assert dto.done is False
    assert dto.obs.shape == (5, 5, 5)
    assert dto.scalars.shape == (6,)
    assert dto.global_state.shape == (77,)
    assert dto.next_legal_mask.dtype == np.bool_
    with pytest.raises(dataclasses.FrozenInstanceError):
        dto.action = 3  # type: ignore[misc]


def test_dataset_manifest_fields_and_frozen():
    """DatasetManifest carries the provenance fields and is immutable (frozen)."""
    manifest = DatasetManifest(
        grid=(5, 5),
        n_pairs=40000,
        seed=7,
        source=SourceTag.EXPERT.value,
        obs_channels=5,
        obs_scalars=6,
        w_v=5,
        schema_version="adrl-001-bc-v1",
    )
    assert manifest.grid == (5, 5)
    assert manifest.n_pairs == 40000
    assert manifest.source == "expert"
    assert manifest.obs_channels == 5
    assert manifest.w_v == 5
    with pytest.raises(dataclasses.FrozenInstanceError):
        manifest.seed = 9  # type: ignore[misc]
