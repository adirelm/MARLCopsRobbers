"""Thin pygame executor + spectator window loop (T7.4/T7.5).

Runs the PURE draw-plan ops (:mod:`src.gui.draw_plan`) against a pygame surface and
hosts the window loop (input -> SDK frame -> render). Tested HEADLESS via pygame-ce
under ``SDL_VIDEODRIVER=dummy`` (conftest); the import is GUARDED so the repo still
imports where pygame is absent. The render DECISIONS are tested in ``test_draw_plan``;
``execute_plan`` / ``render_frame`` / ``_handle_key`` are tested in ``test_gui_render``;
only the interactive ``run_app`` while-loop is exercised manually (scripts/play.py).
"""

from __future__ import annotations

try:  # pygame-ce provides `pygame`; guarded so the repo imports where it's absent.
    import pygame
except ImportError:  # pragma: no cover - pygame optional
    pygame = None

from src.gui import palette
from src.gui.draw_plan import build_board_plan, build_hud_plan
from src.gui.input_map import command_for
from src.gui.transform import GridView


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


def render_frame(surface, font, frame, show_radius=False) -> None:
    """Render one SpectatorFrame (board + HUD) to ``surface``."""
    rows, cols = frame.grid
    view = GridView(surface.get_width(), surface.get_height(), cols, rows)
    execute_plan(surface, font, build_board_plan(frame, view, show_radius))
    execute_plan(surface, font, build_hud_plan(frame))


def run_app(client, width=720, height=560, fps=palette.FPS) -> None:  # pragma: no cover - requires pygame
    """Run the spectator window loop over a state ``client`` (reset/step -> frame)."""
    pygame.init()
    surface = pygame.display.set_mode((width, height))
    font = pygame.font.SysFont(None, palette.FONT_PX + 6)
    clock = pygame.time.Clock()
    frame, paused, show_radius, running = client.reset(), False, False, True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                running, paused, show_radius, frame = _handle_key(
                    event, client, running, paused, show_radius, frame
                )
        if not paused:
            frame = client.step()
        render_frame(surface, font, frame, show_radius)
        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()


def _handle_key(event, client, running, paused, show_radius, frame):  # noqa: PLR0913
    """Map a KEYDOWN to a spectator command; return the updated loop state."""
    command = command_for(pygame.key.name(event.key))
    if command == "quit":
        running = False
    elif command == "toggle_pause":
        paused = not paused
    elif command == "reset":
        frame = client.reset()
    elif command == "toggle_view_radius":
        show_radius = not show_radius
    return running, paused, show_radius, frame
