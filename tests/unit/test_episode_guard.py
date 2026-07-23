"""Episode step-semantics guard tests (codex W2 M5) — replay ingestion fails loud.

The buffer already validates agent-axis WIDTHS; these tests pin the STEP semantics:
``filled`` must be a NON-EMPTY PREFIX (a hole would make the GRU carry hidden state
across fake zero steps) and ``done`` may be True only at the LAST real step.
"""

from __future__ import annotations

import pytest

from src.marl.data.schemas import SourceTag
from tests.unit._buffer_fixtures import _T_MAX, make_buffer, make_episode


def test_valid_prefix_episode_still_accepted():
    buf = make_buffer()
    buf.add_episode(make_episode(3), SourceTag.SELF_PLAY)  # 3 real steps, done at t=2
    assert len(buf) == 1


def test_non_prefix_filled_rejected():
    buf = make_buffer()
    episode = make_episode(_T_MAX)
    episode["filled"][:] = False
    episode["filled"][[0, 2]] = True  # a hole at t=1
    episode["done"][:] = False
    with pytest.raises(ValueError, match="PREFIX"):
        buf.add_episode(episode, SourceTag.RANDOM)


def test_all_empty_episode_rejected():
    buf = make_buffer()
    episode = make_episode(_T_MAX)
    episode["filled"][:] = False
    episode["done"][:] = False
    with pytest.raises(ValueError, match="NON-EMPTY"):
        buf.add_episode(episode, SourceTag.RANDOM)


def test_done_before_last_real_step_rejected():
    buf = make_buffer()
    episode = make_episode(3)  # real steps t=0..2
    episode["done"][:] = False
    episode["done"][1] = True  # terminal flagged with a real step still after it
    with pytest.raises(ValueError, match="LAST real"):
        buf.add_episode(episode, SourceTag.RANDOM)


def test_done_inside_the_pad_rejected():
    buf = make_buffer()
    episode = make_episode(3)
    episode["done"][:] = False
    episode["done"][2] = True  # honest terminal at the last real step...
    episode["filled"][2] = False  # ...then shrink the prefix so it lands in the pad
    with pytest.raises(ValueError, match="LAST real"):
        buf.add_episode(episode, SourceTag.RANDOM)
