"""make_figures — regenerate F1/F2/F5/F6 from ``results/runs/`` + pin a manifest (T10.1/2).

``uv run python -m src.results.make_figures``. Reads the append-only run log, writes the
PLOTTED figures (F1 learning curves, F2 loss, F5 IQL/VDN/QMIX comparison, F6 curriculum
stages — board size AND team size vary together, plus
the §9.3 BOX final-distribution and HEATMAP capture matrix) to ``results/figures/``, and pins
``experiment_manifest.json`` (run count, algorithms, stages, seeds, config hash) so figure
drift is detectable (R8). F3 (GUI screenshots) + F4 (MCP comms) are captured artifacts, NOT
regenerated here. The comparison stage is the largest stage present (IQL diverges from CTDE).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.results._figure_stages import comparison_stage, final_stage, focus_stage
from src.results.aggregate import final_values_by_seed, load_runs
from src.results.plots import plot_comparison, plot_scaling, plot_two_agent_panels
from src.results.plots_extra import plot_capture_heatmap, plot_final_distribution
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


def _return_curves(cfg: dict, fig_dir: Path) -> list[str]:
    """F1b (§7.3a literal): BOTH agents' CUMULATIVE EPISODIC RETURN vs self-play round.

    §7.3(a) asks for "convergence of the cumulative reward" — the measured episodic return,
    not a proxy. The dedicated ``returns_history.jsonl`` (fields post-date the headline
    matrix) is preferred; a FRESH reproduction writes returns straight into ``history.jsonl``,
    so that is the fallback (codex W2 R2). Either way only records that actually CARRY the
    return columns are plotted — a record without them is EXCLUDED, never faked as 0.0 —
    and the figure is skipped entirely when no record qualifies.
    """
    runs_dir = Path(cfg["paths"]["runs_dir"])

    def _with_returns(recs: list[dict]) -> list[dict]:
        return [r for r in recs if "cop_return" in r and "thief_return" in r]

    records = _with_returns(load_runs(runs_dir / "returns_history.jsonl"))
    if not records:
        records = _with_returns(load_runs(runs_dir / "history.jsonl"))
    if not records:
        return []
    stage = focus_stage(records)
    grid = next(rec["grid"] for rec in records if rec["stage"] == stage)
    return [
        str(
            plot_two_agent_panels(
                records,
                "cop_return",
                stage,
                [
                    ("cop", "Cop cumulative return (cop-training rounds)", "episodic return", None),
                    ("thief", "Thief cumulative return (thief-training rounds)", "episodic return", None),
                ],
                f"Both agents' cumulative reward at {grid}x{grid} (mean±SE over seeds)",
                fig_dir / "return_curves.png",
                metric_by_panel={"thief": "thief_return"},
            )
        )
    ]


def _variety_figures(records: list[dict], stage: int, fig_dir: Path, last_k: int) -> list[str]:
    """V3 §9.3 chart variety: the BOX (per-seed spread) + HEATMAP (algorithm x stage) figures.

    Each is SKIPPED — never faked — when the log lacks its data: the box plot needs at least
    one algorithm with final rounds at the focus stage, the heatmap needs at least one stage.
    """
    algos = {rec["algorithm"] for rec in records}
    saved = []
    if any(final_values_by_seed(records, "capture_rate", a, stage, last_k) for a in algos):
        dist = fig_dir / "final_distribution.png"
        saved.append(str(plot_final_distribution(records, "capture_rate", stage, dist, last_k)))
    if any(rec["stage"] is not None for rec in records):
        heat = fig_dir / "capture_heatmap.png"
        saved.append(str(plot_capture_heatmap(records, "capture_rate", heat, last_k)))
    return saved


def main(cfg: dict | None = None) -> list[str]:
    """Regenerate the plotted figures + the manifest; return the figure paths."""
    cfg = cfg or load_config()
    fig_dir = Path(cfg["paths"]["figures_dir"])
    records = load_runs(Path(cfg["paths"]["runs_dir"]) / "history.jsonl")
    if not records:
        raise SystemExit("no runs in results/runs/history.jsonl — run scripts/run_results.py first")
    final = final_stage(records)  # §5.1 5x5 'final test' → the F1/F2 learning + loss curves
    cmp_stage = comparison_stage(cfg, records)  # the multi-cop stage where algos separate (F5/F8)
    last_k = int(cfg["results"]["final_window_rounds"])  # the §9 averaging window (was a literal)
    fgrid = next(rec["grid"] for rec in records if rec["stage"] == final)
    saved = [
        str(
            plot_two_agent_panels(  # §7.3a: BOTH agents' learning (cop capture / thief escape)
                records,
                "capture_rate",
                final,
                [
                    ("cop", "Cop learning — capture rate (cop-training rounds)", "capture rate", None),
                    (
                        "thief",
                        "Thief learning — escape rate (thief-training rounds)",
                        "escape rate (1 - capture)",
                        lambda m: 1.0 - m,
                    ),
                ],
                f"Both agents' learning at {fgrid}x{fgrid} — §5.1 final test (mean±SE over seeds)",
                fig_dir / "learning_curves.png",
            )
        ),
        str(
            plot_two_agent_panels(  # §7.3b: the two NETS' losses (pooling would interleave them)
                records,
                "loss",
                final,
                [
                    ("cop", "Cop net TD-loss (QMIX/VDN/IQL)", "TD loss", None),
                    ("thief", "Thief Double-DQN TD-loss", "TD loss", None),
                ],
                f"Per-network training loss at {fgrid}x{fgrid} — §5.1 final test (mean±SE over seeds)",
                fig_dir / "loss_curves.png",
            )
        ),
        str(plot_comparison(records, "capture_rate", cmp_stage, fig_dir / "baseline_comparison.png", last_k)),
        str(plot_scaling(records, "capture_rate", fig_dir / "scaling.png", last_k)),
    ]
    saved += _variety_figures(records, cmp_stage, fig_dir, last_k)  # V3 §9.3: BOX + HEATMAP
    saved += _return_curves(cfg, fig_dir)
    manifest_path = Path(cfg["paths"]["experiment_manifest"])  # config-driven (no hardcoded path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(_manifest(cfg, records), indent=2), encoding="utf-8")
    print(f"[make_figures] {len(saved)} figures + manifest from {len(records)} records -> {fig_dir}")
    return saved


if __name__ == "__main__":  # pragma: no cover - module CLI entry (uv run -m)
    main()
