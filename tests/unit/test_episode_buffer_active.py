"""Active-mask unit tests for src.marl.replay.episode_buffer (T4.2 / P4b).

Covers the EPISODE-CONSTANT per-slot ``active`` mask added for the N=2 cop
buffer: a WIDENED N=2 episode (one real cop + one zero-filled phantom) round-
trips its ``active=[True, False]`` mask, and a missing OR mis-shaped ``active``
fails loud (the producer must widen to N and stamp the mask explicitly).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.marl.data.schemas import SourceTag
from tests.unit._buffer_fixtures import _T_MAX, make_buffer, make_episode

# Per-agent fields whose agent axis (shape[1], or shape[0] for ``active``) must
# equal n_agents — an N=1 entry must NOT silently numpy-broadcast into N=2.
_AGENT_AXIS_FIELDS = ("scalars", "actions", "reward", "next_legal_mask")


def test_widened_n2_episode_active_mask_round_trips():
    """A WIDENED N=2 episode with one phantom cop (active=[True,False]) round-trips."""
    buf = make_buffer(n_agents=2)
    ep = make_episode(_T_MAX, n_agents=2, fill=1.0)
    ep["active"] = np.array([True, False], dtype=bool)
    buf.add_episode(ep, SourceTag.LIVE_CTDE)
    batch = buf.sample(1)
    assert batch["active"].shape == (1, 2)
    assert batch["active"].dtype == bool
    np.testing.assert_array_equal(batch["active"][0], np.array([True, False]))


def test_missing_or_misshaped_active_raises():
    """A missing OR mis-shaped (wrong agent-axis) active mask FAILS LOUD."""
    buf = make_buffer(n_agents=2)
    missing = make_episode(_T_MAX, n_agents=2, fill=1.0)
    del missing["active"]
    with pytest.raises((ValueError, KeyError)):
        buf.add_episode(missing, SourceTag.RANDOM)
    misshaped = make_episode(_T_MAX, n_agents=2, fill=1.0)
    misshaped["active"] = np.ones(1, dtype=bool)  # wrong agent-axis width
    with pytest.raises(ValueError, match="active"):
        buf.add_episode(misshaped, SourceTag.RANDOM)


@pytest.mark.parametrize("field", _AGENT_AXIS_FIELDS)
def test_n1_per_agent_field_in_n2_episode_raises(field):
    """An N=1 per-agent field in an N=2 episode FAILS LOUD (no silent broadcast).

    obs is N=2 but ONE other per-agent field is N=1: without a full-coverage
    guard the size-1 axis numpy-broadcasts into the N=2 slot, silently
    duplicating cop 0. Every per-agent field must be validated, not just obs.
    """
    buf = make_buffer(n_agents=2)
    ep = make_episode(_T_MAX, n_agents=2, fill=1.0)
    # Re-build only the offending field with a width-1 agent axis.
    ep[field] = ep[field][:, :1]
    with pytest.raises(ValueError, match=field):
        buf.add_episode(ep, SourceTag.LIVE_CTDE)
