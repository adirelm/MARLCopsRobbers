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
from src.gui.effects import TrailTracker
from src.gui.input_map import bindings, command_for
from src.gui.transform import GridView

_FPS_STEP = 2  # speed_up / slow_down increment (local UI behaviour, not a tuned param)
_FPS_MIN = 1
_FPS_MAX = 60


def _bounds(op) -> tuple[int, int, int, int]:
    """The op's bounding box — where an alpha layer must be allocated."""
    if "rect" in op:
        return tuple(op["rect"])
    points = op.get("points") or (op["start"], op["end"])
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return (min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)


def _paint(target, op, offset) -> None:
    """Draw one shape op onto ``target``, translated by ``offset``."""
    ox, oy = offset
    kind = op["kind"]
    if kind in ("fill", "rect", "circle", "ring"):
        x, y, w, h = op["rect"]
        box = (x + ox, y + oy, w, h)
        if kind == "fill":
            pygame.draw.rect(target, op["color"], box)
        elif kind == "rect":
            pygame.draw.rect(target, op["color"], box, width=palette.GRID_W + 2)
        elif kind == "circle":
            pygame.draw.ellipse(target, op["color"], box)
        else:
            pygame.draw.ellipse(target, op["color"], box, width=max(2, w // 14))
    elif kind == "line":
        start, end = op["start"], op["end"]
        pygame.draw.line(
            target, op["color"], (start[0] + ox, start[1] + oy), (end[0] + ox, end[1] + oy), palette.GRID_W
        )
    elif kind == "poly":
        pygame.draw.polygon(target, op["color"], [(px + ox, py + oy) for px, py in op["points"]])


def execute_plan(surface, font, plan) -> None:
    """Execute draw ops (background / fill / rect / circle / ring / line / poly / text).

    An op carrying ``alpha`` is painted onto a scratch RGBA layer sized to its bounding
    box and blitted — pygame's shape primitives take an opaque colour, so the halo,
    trails, ghost and shockwave could not be translucent any other way. The layer is
    per-op and box-sized rather than window-sized, so the cost scales with the shape.
    """
    for op in plan:
        kind = op["kind"]
        if kind == "background":
            surface.fill(op["color"])
        elif kind == "text":
            surface.blit(font.render(op["text"], True, op["color"]), op["pos"])
        elif op.get("alpha") is None:
            _paint(surface, op, (0, 0))
        else:
            x, y, w, h = _bounds(op)
            if w <= 0 or h <= 0:
                continue
            layer = pygame.Surface((w, h), pygame.SRCALPHA)
            _paint(layer, op, (-x, -y))
            layer.set_alpha(op["alpha"])
            surface.blit(layer, (x, y))


def render_frame(surface, font, frame, show_radius=False, supported_commands=None, trails=None) -> None:  # noqa: PLR0913 - one arg per render input
    """Render one SpectatorFrame to ``surface`` — the board letterboxed BELOW the HUD strip.

    ``supported_commands`` (optional) filters the HUD help line to what the frame
    source can honestly do (see :func:`_supported_commands`); ``None`` shows all keys.
    ``trails`` (optional) is the per-agent motion tail from :class:`~src.gui.effects.TrailTracker`.
    """
    rows, cols = frame.grid
    view = GridView(surface.get_width(), surface.get_height(), cols, rows, top_reserved=hud_height(frame))
    execute_plan(surface, font, build_board_plan(frame, view, show_radius, trails))
    execute_plan(surface, font, build_hud_plan(frame, supported_commands, width=surface.get_width()))


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


def run_app(client, width=palette.WINDOW_W, height=palette.WINDOW_H, fps=palette.FPS) -> None:
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
    tracker = TrailTracker(palette.TRAIL_LEN)
    frame, paused, show_radius, running = client.reset(), False, False, True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                running, paused, show_radius, frame, fps = _handle_key(
                    event, client, running, paused, show_radius, frame, fps
                )
        tracker.observe(frame)
        render_frame(surface, font, frame, show_radius, supported, _trails(tracker, frame))
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


def _trails(tracker, frame) -> dict:
    """Per-agent motion tails for ``frame``, keyed the way ``draw_board`` expects."""
    agents = [f"cop_{i}" for i in range(len(frame.cop_positions))] + ["thief"]
    return {agent: tracker.trail(agent) for agent in agents}
