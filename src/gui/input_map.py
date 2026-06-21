"""Input map — key-name -> spectator command (T7.5).

PURE (no pygame): a stable key-name -> command table keyed by ``pygame.key.name``
values (``space``/``escape``/``return``/``n``/``r``/``v``/``-``/``=``/``+``), so the
app maps an event with ``command_for(pygame.key.name(event.key))`` and no toolkit
constants leak into this module — keeping the control scheme testable headless.
"""

from __future__ import annotations

_BINDINGS = {
    "space": "toggle_pause",
    "=": "speed_up",  # the '+' key unshifted
    "+": "speed_up",  # numpad '+'
    "-": "slow_down",
    "n": "next_sub_game",
    "return": "next_sub_game",
    "r": "reset",
    "v": "toggle_view_radius",
    "escape": "quit",
}


def command_for(key_name: str) -> str | None:
    """Return the spectator command bound to ``key_name`` (or ``None`` if unbound)."""
    return _BINDINGS.get(key_name)


def bindings() -> dict[str, str]:
    """Return a copy of the key-name -> command bindings (for the HUD help line)."""
    return dict(_BINDINGS)
