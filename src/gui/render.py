"""Thin pygame executor + spectator window loop (T7.4/T7.5).

Runs the PURE draw-plan ops (:mod:`src.gui.draw_plan`) against a pygame surface and
hosts the window loop (render the CURRENT frame -> input -> step). Tested HEADLESS via
pygame-ce under ``SDL_VIDEODRIVER=dummy`` (conftest); the import is GUARDED so the repo
still imports where pygame is absent. The render DECISIONS are tested in ``test_draw_plan``;
``execute_plan`` / ``render_frame`` / ``_handle_key`` are tested in ``test_gui_render``;
the ``run_app`` while-loop ordering is pinned headless in ``test_gui_app_loop``.
"""

from __future__ import annotations

try:  # pygame-ce provides `pygame`; guarded so the repo imports where it's absent.
    import pygame
except ImportError:  # pragma: no cover - pygame optional
    pygame = None

from src.gui import palette
from src.gui.draw_plan import build_board_plan, build_hud_plan, hud_height
from src.gui.input_map import bindings, command_for
from src.gui.transform import GridView

_FPS_STEP = 2  # speed_up / slow_down increment (local UI behaviour, not a tuned param)
_FPS_MIN = 1
_FPS_MAX = 60


def execute_plan(surface, font, plan) -> None:
    """Execute draw ops against a pygame surface (fill / rect / ellipse / text)."""
    for op in plan:
        kind = op["kind"]
        if kind == "background":
            surface.fill(op["color"])
        elif kind == "fill":
            pygame.draw.rect(surface, op["color"], op["rect"])
        elif kind == "rect":
            pygame.draw.rect(surface, op["color"], op["rect"], width=palette.GRID_W + 2)
        elif kind == "circle":
            pygame.draw.ellipse(surface, op["color"], op["rect"])
        elif kind == "text":
            surface.blit(font.render(op["text"], True, op["color"]), op["pos"])


def render_frame(surface, font, frame, show_radius=False, supported_commands=None) -> None:
    """Render one SpectatorFrame to ``surface`` — the board letterboxed BELOW the HUD strip.

    ``supported_commands`` (optional) filters the HUD help line to what the frame
    source can honestly do (see :func:`_supported_commands`); ``None`` shows all keys.
    """
    rows, cols = frame.grid
    view = GridView(surface.get_width(), surface.get_height(), cols, rows, top_reserved=hud_height(frame))
    execute_plan(surface, font, build_board_plan(frame, view, show_radius))
    execute_plan(surface, font, build_hud_plan(frame, supported_commands))


def _supported_commands(client) -> frozenset:
    """The spectator commands honestly available for ``client``.

    Every command is app-level except ``next_sub_game``, which needs the client to
    expose ``next_sub_game()`` (the HTTP cloud spectator cannot command the remote
    match) — the HUD help line drops the 'n' hint instead of advertising a lie.
    """
    commands = set(bindings().values())
    if not callable(getattr(client, "next_sub_game", None)):
        commands.discard("next_sub_game")
    return frozenset(commands)


def run_app(client, width=720, height=560, fps=palette.FPS) -> None:
    """Run the spectator window loop over a state ``client`` (reset/step -> frame).

    The CURRENT frame is rendered BEFORE the next step is consumed, so startup /
    reset / next-sub-game all show move 0 first, and a quit event exits without
    consuming a step (wave-2 finding G4). Pause skips the step but keeps rendering.
    """
    pygame.init()
    surface = pygame.display.set_mode((width, height))
    font = pygame.font.SysFont(None, palette.FONT_PX + 6)
    clock = pygame.time.Clock()
    supported = _supported_commands(client)
    frame, paused, show_radius, running = client.reset(), False, False, True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                running, paused, show_radius, frame, fps = _handle_key(
                    event, client, running, paused, show_radius, frame, fps
                )
        render_frame(surface, font, frame, show_radius, supported)
        pygame.display.flip()
        clock.tick(fps)
        if running and not paused:
            frame = client.step()
    pygame.quit()


def _handle_key(event, client, running, paused, show_radius, frame, fps):  # noqa: PLR0913
    """Map a KEYDOWN to a spectator command; return the updated loop state (incl. fps)."""
    command = command_for(pygame.key.name(event.key))
    if command == "quit":
        running = False
    elif command == "toggle_pause":
        paused = not paused
    elif command == "reset":
        frame = client.reset()
    elif command == "toggle_view_radius":
        show_radius = not show_radius
    elif command == "next_sub_game":
        advance = getattr(client, "next_sub_game", None)
        frame = advance() if callable(advance) else frame  # unsupported client -> honest no-op
    elif command == "speed_up":
        fps = min(fps + _FPS_STEP, _FPS_MAX)
    elif command == "slow_down":
        fps = max(fps - _FPS_STEP, _FPS_MIN)
    return running, paused, show_radius, frame, fps
