"""StateClient — source-agnostic SpectatorFrame source (T7.5).

The GUI app reads frames through ONE ``reset()/step() -> SpectatorFrame`` interface
regardless of source: ``InProcStateClient`` drives an in-process SpectatorSession
(Stage-1 local), ``ReplayStateClient`` replays a recorded frame list, and (Stage-2)
an HTTP SSE/poll client deserializes frames from a cloud match — all yielding the
SAME :class:`SpectatorFrame`, so the renderer is decoupled from transport. Pure (no
pygame); the HTTP transport is injected so it stays testable headless.
"""

from __future__ import annotations

from collections.abc import Callable

from src.gui.spectator import SpectatorFrame


class InProcStateClient:
    """Drive frames from an in-process SpectatorSession (Stage-1 local)."""

    def __init__(self, session: object) -> None:
        """Wrap a ``SpectatorSession`` (from ``SDK.spectator_session``)."""
        self._session = session

    def reset(self) -> SpectatorFrame:
        """Restart and return the opening frame."""
        return self._session.reset()

    def step(self) -> SpectatorFrame:
        """Advance one move and return the new frame."""
        return self._session.step()


class ReplayStateClient:
    """Replay a recorded sequence of frames (same interface; clamps at the end)."""

    def __init__(self, frames: list[SpectatorFrame]) -> None:
        """Bind a non-empty recorded frame list."""
        self._frames = list(frames)
        self._index = 0

    def reset(self) -> SpectatorFrame:
        """Rewind to the first recorded frame."""
        self._index = 0
        return self._frames[0]

    def step(self) -> SpectatorFrame:
        """Advance to the next recorded frame (clamped at the last)."""
        self._index = min(self._index + 1, len(self._frames) - 1)
        return self._frames[self._index]


class HttpStateClient:
    """Stage-2 cloud frame source — an injected transport fetches the next frame.

    The ``fetch`` callable (SSE/poll wrapper) returns the current frame; injecting
    it keeps this client transport-agnostic and testable without a live server.
    """

    def __init__(self, fetch: Callable[[bool], SpectatorFrame]) -> None:
        """Bind a ``fetch(is_reset) -> SpectatorFrame`` transport."""
        self._fetch = fetch

    def reset(self) -> SpectatorFrame:
        """Fetch the opening frame from the remote match."""
        return self._fetch(True)

    def step(self) -> SpectatorFrame:
        """Fetch the next frame from the remote match."""
        return self._fetch(False)
