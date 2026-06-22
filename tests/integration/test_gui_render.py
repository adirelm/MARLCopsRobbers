"""Headless GUI render tests (T7.4) — execute_plan / render_frame / input.

Runs the REAL pygame-ce renderer under SDL_VIDEODRIVER=dummy (set in conftest):
golden pixel asserts that the cop + thief tokens land on their cells, the
background fills, and the key handler maps space/esc/r/v. No window opens.
"""

from __future__ import annotations

import pygame
import pytest

from src.gui import palette, render
from src.gui.spectator import SpectatorFrame
from src.gui.transform import GridView


@pytest.fixture(scope="module", autouse=True)
def _pygame():
    """Init/quit pygame once for the module (headless via conftest SDL dummy)."""
    pygame.init()
    yield
    pygame.quit()


def _frame(**over) -> SpectatorFrame:
    base = {
        "grid": (5, 5),
        "cop_positions": ((4, 4),),
        "thief_position": (2, 2),
        "barriers": ((0, 4),),
        "view_radius": 2,
        "move": 3,
        "max_moves": 25,
        "sub_game": 1,
        "num_games": 6,
        "scores": {"cop": 0, "thief": 0},
        "totals": {"cop": 0, "thief": 0},
        "winner": None,
        "last_action": {"cop_0": "UP", "thief": "DOWN"},
    }
    return SpectatorFrame(**{**base, **over})


def _center(view: GridView, row: int, col: int) -> tuple[int, int]:
    x, y, w, h = view.cell_rect(col, row)
    return (x + w // 2, y + h // 2)


def test_execute_plan_handles_all_op_kinds():
    """execute_plan paints background, fill, rect (border), circle, and text ops."""
    surface = pygame.Surface((120, 120))
    font = pygame.font.SysFont(None, 18)
    render.execute_plan(
        surface,
        font,
        [
            {"kind": "background", "color": palette.BG},
            {"kind": "fill", "rect": (10, 10, 30, 30), "color": palette.BARRIER},
            {"kind": "rect", "rect": (50, 50, 30, 30), "color": palette.CAPTURE_FLASH},
            {"kind": "circle", "rect": (60, 60, 24, 24), "color": palette.COP},
            {"kind": "text", "pos": (4, 4), "text": "hi", "color": palette.TEXT},
        ],
    )
    assert surface.get_at((0, 0))[:3] == palette.BG
    assert surface.get_at((20, 20))[:3] == palette.BARRIER
    assert surface.get_at((72, 72))[:3] == palette.COP


def test_render_frame_draws_cop_and_thief_on_their_cells():
    """The cop + thief tokens render in their (HUD-free) cell centres."""
    surface = pygame.Surface((400, 400))
    font = pygame.font.SysFont(None, 18)
    render.render_frame(surface, font, _frame())
    view = GridView(400, 400, 5, 5)
    assert surface.get_at(_center(view, 4, 4))[:3] == palette.COP
    assert surface.get_at(_center(view, 2, 2))[:3] == palette.THIEF


def test_handle_key_maps_all_spectator_commands():
    """space->pause, esc->quit, r->reset, v->radius, n->next_sub_game, =/- -> speed; fps threaded."""

    class _Client:
        def reset(self):
            return _frame(move=0)

        def next_sub_game(self):
            return _frame(sub_game=2)

    client, frame, fps = _Client(), _frame(), 12
    _, paused, _, _, _ = render._handle_key(_key(pygame.K_SPACE), client, True, False, False, frame, fps)
    assert paused is True
    running, *_ = render._handle_key(_key(pygame.K_ESCAPE), client, True, False, False, frame, fps)
    assert running is False
    *_, reset_frame, _ = render._handle_key(_key(pygame.K_r), client, True, False, False, frame, fps)
    assert reset_frame.move == 0
    _, _, show_radius, _, _ = render._handle_key(_key(pygame.K_v), client, True, False, False, frame, fps)
    assert show_radius is True
    *_, ng_frame, _ = render._handle_key(_key(pygame.K_n), client, True, False, False, frame, fps)
    assert ng_frame.sub_game == 2  # the 'n' key now actually advances the sub-game
    *_, faster = render._handle_key(_key(pygame.K_EQUALS), client, True, False, False, frame, fps)
    assert faster == fps + 2  # '=' speeds up
    *_, slower = render._handle_key(_key(pygame.K_MINUS), client, True, False, False, frame, fps)
    assert slower == fps - 2  # '-' slows down
    # an unbound key is a no-op (state incl. fps passes through unchanged)
    assert render._handle_key(_key(pygame.K_a), client, True, False, False, frame, fps) == (
        True,
        False,
        False,
        frame,
        fps,
    )


def _key(code: int) -> pygame.event.Event:
    return pygame.event.Event(pygame.KEYDOWN, key=code)
