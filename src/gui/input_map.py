"""Input map — key-name -> spectator command (T7.5).

PURE (no pygame): a stable key-name -> command table the app consults after it
translates a pygame key event to a name (``space``/``+``/``-``/``n``/``r``/``v``/
``esc``). Keeping the binding table pygame-free makes the control scheme testable
headless and decouples the GUI's intent from the toolkit's key constants.
"""

from __future__ import annotations

_BINDINGS = {
    "space": "toggle_pause",
    "+": "speed_up",
    "-": "slow_down",
    "n": "next_sub_game",
    "r": "reset",
    "v": "toggle_view_radius",
    "esc": "quit",
}


def command_for(key_name: str) -> str | None:
    """Return the spectator command bound to ``key_name`` (or ``None`` if unbound)."""
    return _BINDINGS.get(key_name)


def bindings() -> dict[str, str]:
    """Return a copy of the key-name -> command bindings (for the HUD help line)."""
    return dict(_BINDINGS)
