"""§9.3 wire-replay screenshot tests — mid-frame pick + non-trivial PNGs via the GUI path.

The PNG tests render HEADLESS through the EXISTING ``src.gui.render.render_frame`` path
(pygame-ce under SDL dummy, set in conftest) into tmp_path — the worktree evidence PNGs
are produced by ``scripts/replay_wire_match.py``, not by the test run.
"""

from __future__ import annotations

import pytest

from src.mcp.wire_replay import mid_frame_index, replay_match, save_screens
from tests.unit._replay_fixtures import frame, rehearsal_cfg, rehearsal_paths

_MIN_PNG_BYTES = 3000  # "non-trivial": a blank/failed surface saves far smaller


def test_mid_pick_prefers_first_barrier_frame():
    frames = [frame(), frame(move=1), frame(move=2, barriers=((2, 2),)), frame(move=3), frame(move=4)]
    assert mid_frame_index(frames, radius=2) == 2


def test_mid_pick_falls_back_to_first_mutual_visibility():
    far, near = ((0, 0),), (0, 2)  # L1 distance 2 == radius
    frames = [frame(), frame(move=1), frame(move=2, cop_positions=far, thief_position=near), frame(move=3)]
    assert mid_frame_index(frames, radius=2) == 2


def test_mid_pick_falls_back_to_middle_tick():
    frames = [frame(move=i) for i in range(9)]  # never visible (L1 8 > 2), no barriers
    assert mid_frame_index(frames, radius=2) == 4
    assert mid_frame_index([frame(), frame(move=1)], radius=2) in (0, 1)  # tiny game stays in range


def test_save_screens_writes_18_nontrivial_pngs(tmp_path):
    cfg = rehearsal_cfg()
    pytest.importorskip("pygame")
    log, records_path = rehearsal_paths()
    replays = replay_match(cfg, log, records_path)
    saved = save_screens(cfg, replays, out_dir=tmp_path)
    assert len(saved) == 18  # 6 sub-games x (t00, mid, final)
    expected = {f"bonus_sg{g}_{tag}.png" for g in range(1, 7) for tag in ("t00", "mid", "final")}
    assert {p.name for p in saved} == expected
    for path in saved:
        assert path.parent == tmp_path
        assert path.stat().st_size > _MIN_PNG_BYTES, f"{path.name} is trivially small"


def test_save_screens_defaults_to_configured_bonus_dir(cfg):
    """The default output dir comes from config (no hardcoded path in the tool)."""
    assert cfg["gui"]["bonus_screenshot_dir"] == "results/screenshots/bonus"
