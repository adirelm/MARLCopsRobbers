"""run_app loop-ordering pins (wave-2 G4) — headless under SDL_VIDEODRIVER=dummy.

The loop must render the CURRENT frame BEFORE consuming a step: startup shows
move 0 (not move 1), a reset frame is rendered before any further step, and a
quit event exits without stepping. A counting client records reset/step calls
and posts scripted pygame events from inside step(); a recorder patched over
``render_frame`` records exactly which frames reach the screen.
"""

from __future__ import annotations

import pygame

from src.gui import render
from src.gui.spectator import SpectatorFrame


def _frame(move: int) -> SpectatorFrame:
    return SpectatorFrame(
        grid=(3, 3),
        cop_positions=((0, 0),),
        thief_position=(2, 2),
        barriers=(),
        view_radius=1,
        move=move,
        max_moves=25,
        sub_game=1,
        num_games=6,
        scores={"cop": 0, "thief": 0},
        totals={"cop": 0, "thief": 0},
        winner=None,
        last_action=None,
    )


class _CountingClient:
    """Counts reset()/step(); posts a scripted pygame event on given step numbers."""

    def __init__(self, post_on_step: dict) -> None:
        self.resets = 0
        self.steps = 0
        self._post = post_on_step

    def reset(self) -> SpectatorFrame:
        self.resets += 1
        return _frame(0)

    def step(self) -> SpectatorFrame:
        self.steps += 1
        if self.steps in self._post:
            pygame.event.post(self._post[self.steps])
        return _frame(self.steps)


def _run(client, monkeypatch) -> list[tuple[int, int]]:
    """Run the real run_app loop; return the (move, resets-so-far) of every rendered frame."""
    rendered: list[tuple[int, int]] = []

    def recorder(surface, font, frame, show_radius=False, supported_commands=None):
        rendered.append((frame.move, client.resets))

    monkeypatch.setattr(render, "render_frame", recorder)
    render.run_app(client, fps=240)
    return rendered


def test_startup_renders_move_zero_and_quit_consumes_no_step(monkeypatch):
    """Playback renders the opening frame FIRST; each step is rendered; quit steps 0 times."""
    client = _CountingClient({2: pygame.event.Event(pygame.QUIT)})
    rendered = _run(client, monkeypatch)
    assert [move for move, _ in rendered] == [0, 1, 2]  # move 0 IS shown; nothing skipped
    assert client.steps == 2  # the quit iteration renders but does NOT step
    assert client.resets == 1


def test_reset_frame_is_rendered_before_any_further_step(monkeypatch):
    """An 'r' reset frame reaches the screen (move 0 again) before the loop steps on."""
    client = _CountingClient(
        {
            1: pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r),
            2: pygame.event.Event(pygame.QUIT),
        }
    )
    rendered = _run(client, monkeypatch)
    assert rendered[0] == (0, 1)  # opening frame first
    assert (0, 2) in rendered  # the reset frame (move 0, after the 2nd reset) IS rendered
    assert client.steps == 2 and client.resets == 2
