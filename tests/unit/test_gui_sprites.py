"""Arcade sprite geometry — the pursued "pac" wedge and the pursuer "ghost".

The role mapping is the point: the THIEF is the pursued, so it gets the open-mouth wedge
whose mouth faces its direction of travel; the COPS are pursuers, so they get ghost bodies
whose eyes look where they are heading. Direction is therefore carried by the character
itself, which is why the old separate arrow marker was removed.

All geometry is pure — no pygame — so the shapes are verified as numbers here and only
rasterised by the thin executor.
"""

from __future__ import annotations

import math

from src.gui.sprites import ghost_body, ghost_eyes, pac_points

_RECT = (100, 200, 60, 60)  # x, y, w, h
_CX, _CY = 130, 230


def _centroid_angle(points) -> float:
    """Angle from the shape's cell centre to the mean of its points, in degrees."""
    mx = sum(p[0] for p in points) / len(points)
    my = sum(p[1] for p in points) / len(points)
    return math.degrees(math.atan2(-(my - _CY), mx - _CX)) % 360


def test_pac_is_a_closed_wedge_not_a_full_circle() -> None:
    """A mouth means the outline must include the centre — that is what cuts the wedge."""
    points = pac_points(_RECT, "RIGHT")
    assert any(abs(px - _CX) <= 1 and abs(py - _CY) <= 1 for px, py in points)


def test_pac_stays_inside_its_cell() -> None:
    """The sprite must not bleed into neighbouring cells at any mouth angle."""
    for action in ("UP", "DOWN", "LEFT", "RIGHT"):
        for px, py in pac_points(_RECT, action):
            assert 100 <= px <= 160 and 200 <= py <= 260


def test_pac_mouth_faces_the_direction_of_travel() -> None:
    """Mass sits AWAY from the mouth, so the centroid must lie opposite the heading."""
    # Screen y grows downward; _centroid_angle already flips it, so UP is +90 degrees.
    for action, mouth_deg in (("RIGHT", 0), ("UP", 90), ("LEFT", 180), ("DOWN", 270)):
        angle = _centroid_angle(pac_points(_RECT, action))
        opposite = (mouth_deg + 180) % 360
        gap = min((angle - opposite) % 360, (opposite - angle) % 360)
        assert gap < 45, f"{action}: centroid at {angle:.0f}, expected near {opposite}"


def test_pac_without_a_direction_is_a_full_circle() -> None:
    """No heading (spawn tick, or PLACE_BARRIER) means no mouth — never a guessed one."""
    assert pac_points(_RECT, None) is None
    assert pac_points(_RECT, "PLACE_BARRIER") is None


def test_ghost_body_has_a_domed_top_and_a_flat_footprint() -> None:
    """The silhouette must read as a dome over a skirt, filling the cell's lower edge."""
    points = ghost_body(_RECT)
    top = min(py for _px, py in points)
    bottom = max(py for _px, py in points)
    assert top <= 205 and bottom >= 255


def test_ghost_body_stays_inside_its_cell() -> None:
    """Same containment rule as the pac wedge."""
    for px, py in ghost_body(_RECT):
        assert 100 <= px <= 160 and 200 <= py <= 260


def test_ghost_skirt_actually_waves() -> None:
    """A flat hem would read as a bucket; the alternating hem is what makes it a ghost."""
    hem = [py for _px, py in ghost_body(_RECT) if py > _CY]
    assert len(set(hem)) > 1, "ghost hem is flat — no wave"


def test_ghost_eyes_are_a_pair_inside_the_dome() -> None:
    """Two eyes, both in the upper half of the sprite."""
    eyes = ghost_eyes(_RECT, "RIGHT")
    assert len(eyes) == 2
    for _white, pupil in eyes:
        assert pupil[1] < _CY


def test_ghost_pupils_track_the_heading() -> None:
    """Looking where you are going is the whole directional cue for the pursuer."""
    left = [pupil[0] for _w, pupil in ghost_eyes(_RECT, "LEFT")]
    right = [pupil[0] for _w, pupil in ghost_eyes(_RECT, "RIGHT")]
    assert sum(left) < sum(right)

    up = [pupil[1] for _w, pupil in ghost_eyes(_RECT, "UP")]
    down = [pupil[1] for _w, pupil in ghost_eyes(_RECT, "DOWN")]
    assert sum(up) < sum(down)


def test_ghost_pupils_centre_when_there_is_no_heading() -> None:
    """A guessed gaze would assert a direction the frame does not report."""
    centred = ghost_eyes(_RECT, None)
    moving = ghost_eyes(_RECT, "RIGHT")
    assert [p for _w, p in centred] != [p for _w, p in moving]


def test_sprites_scale_with_the_cell() -> None:
    """2x2 boards use much larger cells than 5x5 — nothing may be hard-coded in pixels."""
    small = ghost_body((0, 0, 20, 20))
    large = ghost_body((0, 0, 200, 200))
    assert max(px for px, _ in large) > 10 * max(px for px, _ in small) - 1
