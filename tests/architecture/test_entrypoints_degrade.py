"""Documented entrypoints must ANSWER, not hang or traceback, on a bare clone.

A fresh-clone audit found two entrypoint defects that the unit suite could not see because
it never runs the entrypoints the way a grader does: ``python -m src.gui --help`` hung
forever (no argparse, straight into the window loop) and ``scripts/send_bonus_report.py``
raised a bare FileNotFoundError when the git-ignored real draft was absent.

Both are the same CLASS — a surface that assumes the author's machine — so both are pinned
here rather than only fixed. The rule these encode: a documented command either does its
job or explains why it cannot; it never blocks and never shows a raw traceback.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.config_loader import load_config


def test_the_gui_module_entrypoint_parses_argv_instead_of_ignoring_it() -> None:
    """``-m src.gui --help`` must exit(0) with usage — it used to hang on a headless host.

    Asserted through argparse rather than a subprocess so it runs without pygame: the bug
    was that argv reached no parser at all, which is exactly what ``main(["--help"])``
    proves. SystemExit(0) is argparse's success path for --help.
    """
    from src.gui.__main__ import main  # noqa: PLC0415 — imports pygame-free at module scope

    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0


def test_the_gui_entrypoint_rejects_an_unknown_flag() -> None:
    """A typo'd flag must ERROR, not be silently swallowed and ignored."""
    from src.gui.__main__ import main  # noqa: PLC0415

    with pytest.raises(SystemExit) as exit_info:
        main(["--not-a-real-flag"])
    assert exit_info.value.code != 0


def test_the_bonus_sender_falls_back_to_the_tracked_body_on_a_fresh_clone(tmp_path) -> None:
    """With the git-ignored real draft absent, load the TRACKED redacted body instead.

    This is the fresh-clone contract the config comment already promised and the replay
    tool already honoured. Simulated by pointing ``draft_report`` at a path that does not
    exist, so the test never depends on whether the real draft happens to be present.
    """
    from scripts.send_bonus_report import load_draft  # noqa: PLC0415

    cfg = load_config()
    cfg["wire_match"] = {**cfg["wire_match"], "draft_report": str(tmp_path / "absent.json")}
    tracked = Path(cfg["wire_match"]["redacted_records"]).read_text(encoding="utf-8")
    report = load_draft(cfg)
    assert report["sub_games"] == json.loads(tracked)["sub_games"]


def test_the_bonus_sender_exits_cleanly_when_no_body_exists_at_all(tmp_path) -> None:
    """Neither draft present: a stated SystemExit, never a raw FileNotFoundError."""
    from scripts.send_bonus_report import load_draft  # noqa: PLC0415

    cfg = load_config()
    cfg["wire_match"] = {
        **cfg["wire_match"],
        "draft_report": str(tmp_path / "a.json"),
        "redacted_records": str(tmp_path / "b.json"),
    }
    with pytest.raises(SystemExit, match="no bonus draft found"):
        load_draft(cfg)


# PNGs under figures_dir that are CAPTURED SCREENSHOTS (scripts/capture_comms.py,
# capture_screens.py), not matplotlib renders — their size comes from the terminal/window
# they photographed, so _DPI does not apply. Listed explicitly rather than guessed from
# dimensions: a new matplotlib figure is then checked automatically, and a new screenshot
# fails loudly until someone adds it here, which is the safer direction to be wrong in.
_SCREENSHOTS = frozenset(
    {"cloud_auth.png", "mcp_comms_cloud.png", "mcp_comms_http.png", "mcp_comms_local.png"}
)


def test_every_committed_matplotlib_figure_is_at_the_current_dpi() -> None:
    """F7 sat at 150 dpi while every sibling was 300, so running its own documented
    regeneration command rewrote it and turned the suite red via the notebook-freshness
    guard — a fresh clone could not run the docs without breaking the build.

    Pins the invariant rather than just re-rendering the one file. Every ``plots.py`` /
    ``plots_extra.py`` figure is saved at ``_DPI``, so a stale render is visible in the PNG
    header alone — no matplotlib import and no regeneration needed. Multi-panel figures
    are wider but share the same dpi, so HEIGHT is checked against the two known figsizes.
    """
    import struct  # noqa: PLC0415

    from src.results._plot_io import DPI, FIGSIZE, WIDE_FIGSIZE  # noqa: PLC0415

    fig_dir = Path(load_config()["paths"]["figures_dir"])
    if not fig_dir.exists():
        pytest.skip("figures not generated in this checkout")
    allowed = {round(FIGSIZE[1] * DPI), round(WIDE_FIGSIZE[1] * DPI)}
    stale = []
    for png in sorted(fig_dir.glob("*.png")):
        if png.name in _SCREENSHOTS:
            continue
        width, height = struct.unpack(">II", png.read_bytes()[16:24])
        if height not in allowed:
            stale.append(f"{png.name} is {width}x{height}, expected height in {sorted(allowed)}")
    assert not stale, f"figures rendered at a stale DPI (current DPI={DPI}): {stale}"


def test_every_script_with_a_main_parses_argv() -> None:
    """The entrypoint class was only HALF closed — this file pinned two surfaces, not the rule.

    A fresh-clone audit found ``scripts/plot_minimax_q.py --help`` silently starting its real
    job and still running ten minutes later, and ``sensitivity_sweep.py --help`` doing the
    same and dirtying a tracked file on the way. Both are README-documented. Pinning two
    examples of a class and calling it closed is how the third instance survives, so this
    sweeps EVERY script instead of naming them.

    argparse alone is enough — importing it and calling ``parse_args`` gives ``--help`` and
    rejects unknown flags, which is the whole contract being asserted.
    """
    scripts = sorted(Path(__file__).resolve().parents[2].joinpath("scripts").glob("*.py"))
    assert scripts, "no scripts found — the sweep would pass vacuously"
    missing = [
        path.name
        for path in scripts
        if "def main(" in (text := path.read_text(encoding="utf-8")) and "argparse" not in text
    ]
    assert not missing, f"scripts with a main() that ignore argv, so `--help` starts the real job: {missing}"
