"""The committed §9 notebook must be EXECUTED and show the CURRENT figures.

The 2026-07-24 gap this closes: `analysis.ipynb` is committed *executed*, so a grader
reads its EMBEDDED images — but nothing checked those embeds still matched
``results/figures/``. After the 5x5 §5.1 re-render, four of its figures silently showed
the superseded 4x4 run while README §7.3 described the new ones. Five review rounds
missed it because every gate looked at source, never at the notebook's stored outputs.
Both the notebook and the figures are tracked, so this holds on a fresh clone.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_NOTEBOOK = _ROOT / "notebooks" / "analysis.ipynb"
_FIGURES = _ROOT / "results" / "figures"


def _embedded_pngs() -> list[bytes]:
    """Every PNG the committed notebook carries in its stored cell outputs."""
    notebook = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    images: list[bytes] = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            payload = output.get("data", {}).get("image/png")
            if payload:
                images.append(base64.b64decode("".join(payload) if isinstance(payload, list) else payload))
    return images


def test_notebook_is_committed_executed() -> None:
    """A source-only notebook would show a grader nothing — it must carry its outputs."""
    assert _embedded_pngs(), "analysis.ipynb has no embedded figure outputs — commit it EXECUTED"


def test_every_embedded_figure_matches_a_current_figure_on_disk() -> None:
    """No embed may be a SUPERSEDED render: each must equal a live results/figures PNG.

    Byte equality against the tracked figures is what makes staleness impossible to
    miss — regenerate the figures, re-run the notebook, and commit both together
    (``uv run --group notebook jupyter nbconvert --to notebook --execute --inplace`` —
    use ``uv run --group``, NOT ``uv sync --group notebook``, which would drop the gui extra).
    """
    on_disk = {hashlib.sha256(p.read_bytes()).hexdigest(): p.name for p in _FIGURES.glob("*.png")}
    stale = [
        f"embedded image #{i} ({len(raw) // 1024} KB)"
        for i, raw in enumerate(_embedded_pngs())
        if hashlib.sha256(raw).hexdigest() not in on_disk
    ]
    assert not stale, (
        "analysis.ipynb embeds figures that no longer match results/figures/: "
        + "; ".join(stale)
        + " — re-execute the notebook after regenerating the figures"
    )
