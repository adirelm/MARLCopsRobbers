"""Tests for the pure (pygame-free) GUI logic — transform/palette/input_map/state_client.

GridView geometry (T7.3), the palette local-styling rule (no config import, §4), the
input-map bindings (T7.5), and the source-agnostic state clients all yielding the
SAME SpectatorFrame (T7.5). Verified headless (no pygame needed).
"""

from __future__ import annotations

import inspect

import pytest

from src.gui import palette, render
from src.gui.input_map import bindings, command_for
from src.gui.spectator import SpectatorFrame
from src.gui.state_client import HttpStateClient, InProcStateClient, ReplayStateClient
from src.gui.transform import GridView


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_gridview_cells_are_square_and_within_window(n):
    """Cells are square and the whole board fits inside the window for 2x2..5x5."""
    view = GridView(640, 480, n, n)
    assert view.cell_px >= 1
    x, y, w, h = view.cell_rect(n - 1, n - 1)
    assert x >= 0 and x + w <= 640
    assert y >= 0 and y + h <= 480


def test_gridview_rejects_off_board_cell():
    """An off-board cell raises (bounds-checked)."""
    with pytest.raises(ValueError, match="out of"):
        GridView(640, 480, 5, 5).cell_rect(5, 0)


def test_gridview_caps_cell_size():
    """A small board in a large window caps the cell at CELL_PX_CAP."""
    assert GridView(2000, 2000, 2, 2).cell_px == palette.CELL_PX_CAP


def test_palette_is_valid_rgb_and_imports_no_config():
    """Colours are 0-255 RGB and the palette imports NOTHING from config (§4)."""
    for name in ("BG", "COP", "THIEF", "BARRIER", "CAPTURE_FLASH"):
        rgb = getattr(palette, name)
        assert len(rgb) == 3
        assert all(0 <= channel <= 255 for channel in rgb)
    source = inspect.getsource(palette)
    assert "load_config" not in source
    assert "config_loader" not in source


def test_render_module_exposes_executor_api():
    """src/gui/render.py (guarded import) exposes the executor + loop API."""
    assert hasattr(render, "execute_plan")
    assert hasattr(render, "render_frame")
    assert hasattr(render, "run_app")


def test_input_map_bindings():
    """Keys (pygame.key.name values) map to the spectator commands; unbound -> None."""
    assert command_for("space") == "toggle_pause"
    assert command_for("escape") == "quit"
    assert command_for("unbound") is None
    assert set(bindings()) >= {"space", "escape", "n", "r", "v", "-"}


def _frame(move: int, sub_game: int = 1) -> SpectatorFrame:
    """A minimal SpectatorFrame at a given move."""
    return SpectatorFrame(
        grid=(5, 5),
        cop_positions=((0, 0),),
        thief_position=(4, 4),
        barriers=(),
        view_radius=2,
        move=move,
        max_moves=25,
        sub_game=sub_game,
        num_games=6,
        scores={"cop": 0, "thief": 0},
        totals={"cop": 0, "thief": 0},
        winner=None,
        last_action=None,
    )


def test_state_clients_are_source_agnostic():
    """In-proc, replay, and http (injected transport) yield IDENTICAL frames."""
    frames = [_frame(0), _frame(1)]

    class _Session:
        def reset(self):
            return frames[0]

        def step(self):
            return frames[1]

    inproc = InProcStateClient(_Session())
    replay = ReplayStateClient(frames)
    http = HttpStateClient(lambda is_reset: frames[0] if is_reset else frames[1])
    assert inproc.reset() == replay.reset() == http.reset() == frames[0]
    assert inproc.step() == replay.step() == http.step() == frames[1]


def test_replay_next_sub_game_seeks_forward_and_clamps():
    """'n' on a replay JUMPS to the next sub-game's first frame — never rewinds (wave-2 G5)."""
    frames = [_frame(0, sub_game=1), _frame(1, sub_game=1), _frame(0, sub_game=2), _frame(1, sub_game=2)]
    replay = ReplayStateClient(frames)
    replay.reset()
    replay.step()  # inside sub-game 1
    advanced = replay.next_sub_game()
    assert advanced == frames[2]  # first frame OF THE NEXT sub-game
    assert replay.next_sub_game() == frames[-1]  # no later sub-game -> clamps at the end
    assert replay.step() == frames[-1]  # still clamped, no rewind ever happened


def test_handle_key_next_is_a_noop_for_clients_without_next_sub_game():
    """'n' on a capability-less client leaves the frame UNCHANGED — no silent reset (wave-2 G5)."""
    pygame = pytest.importorskip("pygame")

    class _NoNextClient:
        def __init__(self) -> None:
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1
            return _frame(0)

        def step(self):
            return _frame(1)

    client, frame = _NoNextClient(), _frame(7)
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n)
    *_, out_frame, _ = render._handle_key(event, client, True, False, False, frame, 8)
    assert out_frame is frame  # unchanged — the HUD no longer advertises 'n' for this client
    assert client.reset_calls == 0  # and reset was NOT silently substituted


def test_supported_commands_reflect_client_capability():
    """_supported_commands drops next_sub_game iff the client cannot honestly do it."""
    replay = ReplayStateClient([_frame(0)])
    http = HttpStateClient(lambda is_reset: _frame(0))
    assert "next_sub_game" in render._supported_commands(replay)
    assert "next_sub_game" not in render._supported_commands(http)
    assert "toggle_pause" in render._supported_commands(http)  # app-level keys always stay
