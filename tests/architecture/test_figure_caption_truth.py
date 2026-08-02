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


def test_readme_config_table_covers_every_top_level_config_key() -> None:
    """README §Configuration claims it lists "every top-level key" — hold it to that.

    It silently fell to 20-of-23 as `wire_match` / `wire_agent` / `matchup_eval` were added
    (2026-07-30 audit). A documented completeness claim needs a completeness check.
    """
    import yaml  # noqa: PLC0415 — only this test needs the raw (un-interpolated) config

    keys = set(yaml.safe_load((_ROOT / "config" / "config.yaml").read_text(encoding="utf-8")))
    listed = set(re.findall(r"^\| `([a-z_]+)` \|", _README.read_text(encoding="utf-8"), re.M))
    assert not keys - listed, (
        "README's config table claims to cover every top-level key but omits: "
        + ", ".join(sorted(keys - listed))
    )


def test_experiment_manifest_hash_matches_the_current_config() -> None:
    """The manifest's config hash IS the R8 drift detector — so check the detector itself.

    README §7.3 cites it as proof of "zero README<->code drift", but nothing verified the
    committed hash still matched the live config: a 2026-08-02 fresh-clone run found it stale
    (a config key had been added without regenerating), i.e. the detector was silently RED.
    Fix by re-running ``uv run python -m src.results.make_figures`` and committing both.
    """
    import hashlib  # noqa: PLC0415 — only this test needs the manifest digest
    import json  # noqa: PLC0415

    from src.utils.config_loader import load_config  # noqa: PLC0415

    manifest_path = _ROOT / "results" / "figures" / "experiment_manifest.json"
    if not manifest_path.exists():
        pytest.skip("no experiment manifest in this checkout")
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))["config_sha256"]
    payload = json.dumps(load_config(), sort_keys=True, default=str).encode("utf-8")
    assert recorded == hashlib.sha256(payload).hexdigest()[:16], (
        "experiment_manifest.json config_sha256 is stale vs config/config.yaml — regenerate the "
        "figures (uv run python -m src.results.make_figures) and commit the manifest with them"
    )


def test_documented_test_count_matches_the_suite() -> None:
    """README/QUALITY advertise a test count — hold it to the number actually collected.

    It drifted twice (888->895, 907->908) because every other doc number is test-pinned and
    this one was not. Skipped on a targeted subset run, where the collected count is partial.
    """
    from tests.conftest import COLLECTED  # noqa: PLC0415 — the collection hook fills this

    if not COLLECTED["full_suite"]:
        pytest.skip("subset run — the collected count is not the whole suite")
    actual = int(COLLECTED["count"])
    stale: list[str] = []
    for rel in ("README.md", "docs/QUALITY.md"):
        for number, line in enumerate((_ROOT / rel).read_text(encoding="utf-8").splitlines(), start=1):
            for claimed in re.findall(r"\b(\d{3,4}) tests\b", line):
                if int(claimed) != actual:
                    stale.append(f"{rel}:{number} says {claimed}")
    assert not stale, f"docs advertise a stale test count (suite collects {actual}): " + "; ".join(stale)
