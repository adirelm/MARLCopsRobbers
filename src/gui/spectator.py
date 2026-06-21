"""SpectatorFrame — the frozen god-view snapshot the GUI renders (T7.1).

A plain, FROZEN dataclass carrying the FULL board (the spectator reads the
referee's ground truth, NOT agent obs) plus the match HUD (sub-game, move,
scores, totals, winner, last action). It is a pure leaf — it imports NOTHING from
``src.marl`` / ``src.mcp`` / ``src.sdk`` (the GUI-purity boundary): the
:class:`~src.services.spectator.SpectatorSession` (obtained via
``SDK.spectator_session``) BUILDS frames; the GUI only consumes them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpectatorFrame:
    """A frozen god-view board snapshot + the match HUD (the GUI's only input)."""

    grid: tuple[int, int]
    cop_positions: tuple[tuple[int, int], ...]
    thief_position: tuple[int, int]
    barriers: tuple[tuple[int, int], ...]
    view_radius: int
    move: int
    max_moves: int
    sub_game: int
    num_games: int
    scores: dict[str, int]
    totals: dict[str, int]
    winner: str | None
    last_action: dict[str, str] | None
    trace_id: str | None = None  # optional cloud field (None in-proc)
