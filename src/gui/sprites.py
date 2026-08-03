"""Arcade maze-chase sprite geometry — PURE point math, no pygame.

Original shapes built from primitives in the maze-chase idiom, chosen because the idiom's
two silhouettes map exactly onto this game's roles:

* the **thief** is the PURSUED, so it is an open-mouth wedge whose mouth faces its
  direction of travel;
* the **cops** are the PURSUERS, so they are ghost bodies whose eyes look where they head.

The character therefore CARRIES its own heading, which is why the separate arrow marker
that used to sit beside each token was removed rather than kept alongside these.

``pygame.draw.arc`` does not reliably fill, so the wedge is emitted as a polygon of arc
points plus the centre — which also keeps every shape expressible with the executor's
existing ``poly``/``circle`` ops.
"""

from __future__ import annotations

import math

from src.gui import palette

# Screen space: y grows DOWNWARD, so each heading's angle is negated on the y axis.
_HEADING_DEG = {"RIGHT": 0.0, "UP": 90.0, "LEFT": 180.0, "DOWN": 270.0}

MOUTH_DEG = 62.0  # total opening of the wedge
_ARC_STEPS = 22  # points along the wedge's arc; enough to read as round at any cell size
_GHOST_FEET = 3  # bumps along the ghost's hem
_DOME_STEPS = 14


def _unit(action: str | None) -> tuple[float, float] | None:
    """Return the (dx, dy) screen-space unit vector for ``action``, or ``None``."""
    degrees = _HEADING_DEG.get(action or "")
    if degrees is None:
        return None
    radians = math.radians(degrees)
    return (math.cos(radians), -math.sin(radians))


def pac_points(rect: tuple[int, int, int, int], action: str | None) -> tuple | None:
    """Return the pursued wedge's polygon, mouth facing ``action``.

    ``None`` when the frame reports no heading (the spawn tick, or a ``PLACE_BARRIER``
    that consumed the move without moving the agent) — the caller then draws a plain
    circle rather than inventing a direction the frame never claimed.
    """
    if action not in _HEADING_DEG:
        return None
    x, y, w, h = rect
    cx, cy, radius = x + w / 2, y + h / 2, min(w, h) / 2
    facing = _HEADING_DEG[action]
    start, end = facing + MOUTH_DEG / 2, facing + 360 - MOUTH_DEG / 2
    points = [(cx, cy)]  # the centre is what turns a disc into a wedge
    for step in range(_ARC_STEPS + 1):
        angle = math.radians(start + (end - start) * step / _ARC_STEPS)
        points.append((round(cx + radius * math.cos(angle)), round(cy - radius * math.sin(angle))))
    return tuple(points)


def ghost_body(rect: tuple[int, int, int, int]) -> tuple:
    """Return the pursuer's silhouette: a domed top over a waving hem, filling ``rect``."""
    x, y, w, h = rect
    radius = w / 2
    dome_cy = y + radius
    points = [
        (
            round(x + radius + radius * math.cos(math.radians(180 - 180 * s / _DOME_STEPS))),
            round(dome_cy - radius * math.sin(math.radians(180 - 180 * s / _DOME_STEPS))),
        )
        for s in range(_DOME_STEPS + 1)
    ]
    dip = h * 0.17
    steps = 2 * _GHOST_FEET
    # Hem walked right-to-left so the outline closes cleanly back up the left side.
    points += [
        (round(x + w - step * w / steps), round(y + h - (0 if step % 2 == 0 else dip)))
        for step in range(steps + 1)
    ]
    return tuple(points)


def ghost_eyes(rect: tuple[int, int, int, int], action: str | None) -> tuple:
    """Return ``((white_rect, pupil_centre), ...)`` for both eyes, gaze following ``action``.

    With no heading the pupils sit centred: a guessed gaze would assert a direction the
    frame does not report.
    """
    x, y, w, h = rect
    white_r, pupil_r = w * 0.15, w * 0.075
    eye_cy = y + h * 0.38
    gaze = _unit(action) or (0.0, 0.0)
    shift = white_r - pupil_r
    eyes = []
    for side in (-1, 1):
        eye_cx = x + w / 2 + side * w * 0.19
        size = max(1, round(2 * white_r))
        white = (round(eye_cx - white_r), round(eye_cy - white_r), size, size)
        pupil = (round(eye_cx + gaze[0] * shift), round(eye_cy + gaze[1] * shift), max(1, round(pupil_r)))
        eyes.append((white, pupil))
    return tuple(eyes)


def pursued_ops(rect: tuple, action: str | None, alpha: int | None) -> list[dict]:
    """The thief: an open-mouth wedge facing its heading, or a plain disc when it has none."""
    points = pac_points(rect, action)
    if points is None:  # spawn tick / PLACE_BARRIER — no heading to show
        disc = {"kind": "circle", "rect": rect, "color": palette.THIEF}  # rect is already inset
        if alpha is not None:
            disc["alpha"] = alpha
        return [disc]
    op = {"kind": "poly", "points": points, "color": palette.THIEF}
    if alpha is not None:
        op["alpha"] = alpha
    return [op]


def pursuer_ops(rect: tuple, action: str | None) -> list[dict]:
    """A cop: ghost body plus eyes whose pupils look where it is heading."""
    ops: list[dict] = [{"kind": "poly", "points": ghost_body(rect), "color": palette.COP}]
    for white, (px, py, pr) in ghost_eyes(rect, action):
        ops.append({"kind": "circle", "rect": white, "color": palette.EYE_WHITE})
        ops.append({"kind": "circle", "rect": (px - pr, py - pr, 2 * pr, 2 * pr), "color": palette.EYE_PUPIL})
    return ops


def capture_rings(view, frame) -> list[dict]:
    """Return concentric capture rings on the CAPTURING cop (nearest the thief)."""
    tr, tc = frame.thief_position
    cr, cc = min(frame.cop_positions, key=lambda pos: abs(pos[0] - tr) + abs(pos[1] - tc))
    x, y, w, h = view.cell_rect(cc, cr)
    ops = []
    for ring in range(palette.SHOCKWAVE_RINGS):
        grow = int(w * 0.18 * ring)
        ops.append(
            {
                "kind": "ring",
                "rect": (x - grow, y - grow, w + 2 * grow, h + 2 * grow),
                "color": palette.CAPTURE_FLASH,
                "alpha": max(20, palette.SHOCKWAVE_ALPHA - ring * 45),
            }
        )
    return ops
