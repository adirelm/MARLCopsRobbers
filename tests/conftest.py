"""Shared pytest fixtures for the MARL Cops & Robbers test suite.

Exposes the real `cfg` (loaded from config/config.yaml) and a `make_state`
factory that builds a `GlobalState` with sensible 5x5 defaults.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pytest

from src.marl.env.types import GlobalState, Pos
from src.utils.config_loader import load_config


@pytest.fixture
def cfg() -> dict:
    """Return the real project config (config/config.yaml), validated."""
    return load_config()


@pytest.fixture
def make_state() -> Callable[..., GlobalState]:
    """Return a factory building a GlobalState with 5x5 defaults.

    Keyword overrides accepted: cop_pos, thief_pos, barriers, barriers_used,
    step, h, w, terminal. `cop_pos` accepts a single Pos or an iterable of Pos.
    """

    def _factory(  # noqa: PLR0913 — one kwarg per GlobalState field is intentional
        cop_pos: Pos | Iterable[Pos] = (0, 0),
        thief_pos: Pos = (4, 4),
        barriers: Iterable[Pos] = (),
        barriers_used: int = 0,
        step: int = 0,
        h: int = 5,
        w: int = 5,
        terminal: bool = False,
    ) -> GlobalState:
        if isinstance(cop_pos, tuple) and len(cop_pos) == 2 and isinstance(cop_pos[0], int):
            cops: tuple[Pos, ...] = (cop_pos,)  # type: ignore[assignment]
        else:
            cops = tuple(cop_pos)  # type: ignore[arg-type]
        return GlobalState(
            cop_pos=cops,
            thief_pos=thief_pos,
            barriers=frozenset(barriers),
            barriers_used=barriers_used,
            step=step,
            h=h,
            w=w,
            terminal=terminal,
        )

    return _factory
