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


def test_handle_key_maps_pause_quit_and_reset():
    """space->pause, esc->quit, r->reset (calls client.reset)."""

    class _Client:
        def reset(self):
            return _frame(move=0)

    frame = _frame()
    _, paused, _, _ = render._handle_key(_key(pygame.K_SPACE), _Client(), True, False, False, frame)
    assert paused is True
    running, *_ = render._handle_key(_key(pygame.K_ESCAPE), _Client(), True, False, False, frame)
    assert running is False
    *_, reset_frame = render._handle_key(_key(pygame.K_r), _Client(), True, False, False, frame)
    assert reset_frame.move == 0
    _, _, show_radius, _ = render._handle_key(_key(pygame.K_v), _Client(), True, False, False, frame)
    assert show_radius is True
    # an unbound key is a no-op (state passes through unchanged)
    state = render._handle_key(_key(pygame.K_a), _Client(), True, False, False, frame)
    assert state == (True, False, False, frame)


def _key(code: int) -> pygame.event.Event:
    return pygame.event.Event(pygame.KEYDOWN, key=code)
