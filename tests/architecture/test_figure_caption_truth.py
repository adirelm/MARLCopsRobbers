"""A README figure caption must name the stage its figure is actually rendered at.

The 2026-07-24 gap this closes: after the §5.1 re-render moved the learning/loss curves to
5x5 (while the 3-way comparison stayed at the 4x4 two-cop stage), the F2 caption still read
"at 4x4" — contradicting the figure's own rendered title. Captions are prose, so nothing
caught it; a grader comparing caption to image would. This pins only the STAGE CLAIM (the
grid label), so rewording a caption stays free.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.results._figure_stages import comparison_stage, final_stage
from src.results.aggregate import load_runs
from src.utils.config_loader import load_config

_ROOT = Path(__file__).resolve().parents[2]
_README = _ROOT / "README.md"

# figure file -> which stage-selector decides where make_figures renders it
_FIGURE_STAGE = {
    "learning_curves.png": "final",  # F1  — the §5.1 final test
    "loss_curves.png": "final",  # F2  — same stage as F1
    "baseline_comparison.png": "comparison",  # F5  — the multi-cop stage
    "final_distribution.png": "comparison",  # F8  — the seeds behind F5
}


def _caption_after(figure: str) -> str:
    """Return the italic caption block that follows the figure's markdown embed."""
    text = _README.read_text(encoding="utf-8")
    embed = re.search(rf"!\[[^\]]*\]\(results/figures/{re.escape(figure)}\)", text)
    assert embed, f"README does not embed {figure}"
    after = text[embed.end() :].lstrip("\n")
    return after.split("\n\n", 1)[0]  # the caption paragraph


@pytest.mark.parametrize("figure,selector", sorted(_FIGURE_STAGE.items()))
def test_caption_states_the_stage_the_figure_is_rendered_at(figure: str, selector: str) -> None:
    """The caption must mention the grid of the stage make_figures actually used."""
    cfg = load_config()
    records = load_runs(Path(cfg["paths"]["runs_dir"]) / "history.jsonl")
    if not records:
        pytest.skip("no run log in this checkout — the figure stages cannot be derived")
    stage = final_stage(records) if selector == "final" else comparison_stage(cfg, records)
    grid = next(rec["grid"] for rec in records if rec["stage"] == stage)
    caption = _caption_after(figure)
    # the prose uses either form of the times sign
    times = "[x\u00d7]"  # ASCII x or U+00D7 MULTIPLICATION SIGN (escaped: RUF001)
    assert re.search(rf"{grid}\s*{times}\s*{grid}", caption), (
        f"{figure} renders at stage {stage} ({grid}x{grid}) but its README caption never says so "
        f"— caption: {caption[:160]!r}"
    )
