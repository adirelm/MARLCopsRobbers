"""Thin pygame executor + spectator window loop (T7.4/T7.5) — REQUIRES pygame.

Runs the PURE draw-plan ops (:mod:`src.gui.draw_plan`) against a pygame surface and
hosts the window loop (input -> SDK frame -> render). pygame can't build on py3.14
here, so this module is GUARDED — it IMPORTS cleanly without pygame (so the repo +
the draw-plan tests stay green) and its pygame-calling bodies are ``# pragma: no
cover`` (exercised on a pygame-capable machine). The render DECISIONS are tested in
``test_draw_plan``; only the thin pygame execution lives here.
"""

from __future__ import annotations

try:  # pygame is an optional extra (the `gui` group); absent on this build host.
    import pygame
except ImportError:  # pragma: no cover - pygame optional
    pygame = None

from src.gui import palette
from src.gui.draw_plan import build_board_plan, build_hud_plan
from src.gui.input_map import command_for
from src.gui.transform import GridView


def execute_plan(surface, font, plan) -> None:  # pragma: no cover - requires pygame
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


def render_frame(surface, font, frame, show_radius=False) -> None:  # pragma: no cover - requires pygame
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


def _handle_key(event, client, running, paused, show_radius, frame):  # noqa: PLR0913  # pragma: no cover
    """Map a KEYDOWN to a spectator command; return the updated loop state."""
    name = pygame.key.name(event.key)
    command = command_for({"return": "n"}.get(name, name))
    if command == "quit":
        running = False
    elif command == "toggle_pause":
        paused = not paused
    elif command == "reset":
        frame = client.reset()
    elif command == "toggle_view_radius":
        show_radius = not show_radius
    return running, paused, show_radius, frame
