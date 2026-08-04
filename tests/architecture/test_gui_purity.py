"""GUI architecture-purity gates (T7.7): import boundary, spectator purity, no leak.

Three hard gates: (1) every ``src.*`` import in src/gui is on an ALLOW-list
(src.gui / src.sdk / src.utils) — the GUI never reaches into env / MCP / services; (2) the rendered
SpectatorFrame is frozen; (3) the agent request_move schema is local-obs-only (no
global state), and the env FOGS an opponent beyond the view radius (the Dec-POMDP
§2.1 partial-observability invariant, asserted as a hard test). No pygame needed.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np

from src.gui.spectator import SpectatorFrame
from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.mcp.schemas import MoveRequest

_GUI_DIR = Path(__file__).resolve().parents[2] / "src" / "gui"
# ALLOW-list, not a deny-list. The previous deny-list named marl/mcp/services/api/reporting,
# so any NEW src package the GUI grew a dependency on would have passed silently while the
# test's own name promised "only sdk/gui/pygame" — and docs/UX.md repeated that promise.
_ALLOWED_SRC = ("src.gui", "src.sdk", "src.utils")
_SRC_IMPORT = re.compile(r"^\s*(?:from|import)\s+(src\.[\w.]+)", re.MULTILINE)


def test_gui_imports_only_sdk_gui_utils_pygame():
    """Every ``src.*`` import in src/gui is on the allow-list — checked, not merely named."""
    offenders = []
    for py in sorted(_GUI_DIR.glob("*.py")):
        for module in _SRC_IMPORT.findall(py.read_text(encoding="utf-8")):
            if not module.startswith(_ALLOWED_SRC):
                offenders.append(f"{py.name} -> {module}")
    assert offenders == [], f"GUI imports outside {_ALLOWED_SRC}: {offenders}"


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


def test_no_launcher_hardcodes_the_cop_count():
    """Both documented spectator launchers must read ``env.num_cops`` from config.

    `scripts/play.py` was fixed for this ("was a bare 1") but `src/gui/__main__.py` kept
    its literal, so the two launchers would have rendered different boards the moment the
    graded cop count changed — invisibly, because today's config value happens to be 1.
    """
    root = Path(__file__).resolve().parents[2]
    for rel in ("src/gui/__main__.py", "scripts/play.py", "scripts/capture_screens.py"):
        source = (root / rel).read_text(encoding="utf-8")
        assert "num_cops=1" not in source, f"{rel} hardcodes the cop count"
        assert 'cfg["env"]["num_cops"]' in source, f"{rel} does not read env.num_cops"


def test_the_two_view_radius_config_sources_agree():
    """The referee masks with ``mcp.observation.view_radius``; its replay verifier reads
    ``env.view_radius_by_grid``. One value, two keys — if they ever diverge the verifier
    checks a DIFFERENT mask than the referee produced, so an honest log would be rejected
    (or a wider-radius leak accepted) with no other symptom.
    """
    from src.marl.env.observation import view_radius  # noqa: PLC0415
    from src.utils.config_loader import load_config  # noqa: PLC0415

    cfg = load_config()
    grid = int(cfg["game"]["grid_size"])
    referee = int(cfg["mcp"]["observation"]["view_radius"])
    verifier = int(view_radius(grid, grid, cfg))
    assert referee == verifier, (
        f"mcp.observation.view_radius={referee} but env.view_radius_by_grid[{grid}]={verifier}"
    )
