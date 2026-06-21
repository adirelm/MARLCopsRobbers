"""make_figures — regenerate F1/F2/F5/F6 from ``results/runs/`` + pin a manifest (T10.1/2).

``uv run python -m src.results.make_figures``. Reads the append-only run log, writes the
four PLOTTED figures (F1 learning curves, F2 loss, F5 IQL/VDN/QMIX comparison, F6 scaling)
to ``results/figures/``, and pins ``experiment_manifest.json`` (run count, algorithms,
stages, seeds, config hash) so figure drift is detectable (R8). F3 (GUI screenshots) + F4
(MCP comms) are captured artifacts, NOT regenerated here. The comparison stage is the
largest stage present (where IQL diverges from CTDE).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.results.aggregate import load_runs
from src.results.plots import plot_comparison, plot_curve_figure, plot_scaling
from src.utils.config_loader import load_config


def _manifest(cfg: dict, records: list[dict]) -> dict:
    """Pin the run provenance (combos / arms / seeds + a config hash) for drift detection."""
    config_bytes = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return {
        "runs": len({(r["algorithm"], r["seed"], r["stage"]) for r in records}),
        "algorithms": sorted({r["algorithm"] for r in records}),
        "stages": sorted({r["stage"] for r in records}),
        "seeds": sorted({r["seed"] for r in records}),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest()[:16],
    }


def main(cfg: dict | None = None) -> list[str]:
    """Regenerate the four plotted figures + the manifest; return the figure paths."""
    cfg = cfg or load_config()
    fig_dir = Path(cfg["paths"]["figures_dir"])
    records = load_runs(Path(cfg["paths"]["runs_dir"]) / "history.jsonl")
    if not records:
        raise SystemExit("no runs in results/runs/history.jsonl — run scripts/run_results.py first")
    stage = max(r["stage"] for r in records)  # the discriminating (largest) comparison stage
    saved = [
        str(
            plot_curve_figure(
                records,
                "capture_rate",
                stage,
                "Learning curves — capture rate (mean±SE)",
                "capture rate",
                fig_dir / "learning_curves.png",
            )
        ),
        str(
            plot_curve_figure(
                records,
                "loss",
                stage,
                "Training loss (mean±SE)",
                "loss",
                fig_dir / "loss_curves.png",
            )
        ),
        str(plot_comparison(records, "capture_rate", stage, fig_dir / "baseline_comparison.png")),
        str(plot_scaling(records, "capture_rate", fig_dir / "scaling.png")),
    ]
    (fig_dir / "experiment_manifest.json").write_text(
        json.dumps(_manifest(cfg, records), indent=2), encoding="utf-8"
    )
    print(f"[make_figures] {len(saved)} figures + manifest from {len(records)} records -> {fig_dir}")
    return saved


if __name__ == "__main__":  # pragma: no cover - module CLI entry (uv run -m)
    main()
