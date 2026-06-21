"""GUI architecture-purity gates (T7.7): import boundary, spectator purity, no leak.

Three hard gates: (1) no src/gui module imports marl/mcp/services/api/reporting —
the GUI talks ONLY to the SDK + sibling gui modules + pygame; (2) the rendered
SpectatorFrame is frozen; (3) the agent request_move schema is local-obs-only (no
global state), and the env FOGS an opponent beyond the view radius (the Dec-POMDP
§2.1 partial-observability invariant, asserted as a hard test). No pygame needed.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from src.gui.spectator import SpectatorFrame
from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.mcp.schemas import MoveRequest

_GUI_DIR = Path(__file__).resolve().parents[2] / "src" / "gui"
_FORBIDDEN = ("src.marl", "src.mcp", "src.services", "src.api", "src.reporting")


def test_gui_imports_only_sdk_gui_pygame():
    """No src/gui module reaches into marl/mcp/services/api/reporting internals."""
    offenders = []
    for py in _GUI_DIR.glob("*.py"):
        source = py.read_text(encoding="utf-8")
        for forbidden in _FORBIDDEN:
            if f"import {forbidden}" in source or f"from {forbidden}" in source:
                offenders.append(f"{py.name} -> {forbidden}")
    assert offenders == [], f"GUI imports beyond sdk/gui/pygame: {offenders}"


def test_spectator_frame_is_frozen():
    """The frame the GUI renders is an immutable snapshot (spectator purity)."""
    assert dataclasses.is_dataclass(SpectatorFrame)
    assert SpectatorFrame.__dataclass_params__.frozen


def test_move_request_is_local_obs_only():
    """The agent request_move schema carries ONLY local obs + session_id/tick."""
    fields = set(MoveRequest.model_fields)
    assert fields == {"session_id", "tick", "image", "scalars", "legal_mask"}
    assert not (fields & {"global_state", "totals", "cop_position", "thief_position", "scores"})


def test_env_fogs_opponent_beyond_view_radius(cfg):
    """Dec-POMDP §2.1: an opponent beyond the view radius is FOGGED in the agent obs."""
    env = CopsRobbersEnv(cfg, h=5, w=5, num_cops=1)
    radius = cfg["env"]["view_radius_by_grid"][5]
    for seed in range(50):
        obs, _info = env.reset(seed=seed)
        state = env.state()
        cop, thief = state.cop_pos[0], state.thief_pos
        if abs(cop[0] - thief[0]) + abs(cop[1] - thief[1]) > radius:
            other_visible = np.asarray(obs["cop_0"]["image"])[1]  # channel 1 = opponent plane
            assert other_visible.sum() == 0  # fogged beyond radius (no leak)
            return
    raise AssertionError("no far-apart spawn found within 50 seeds")
