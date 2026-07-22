"""make_figures — regenerate F1/F2/F5/F6 from ``results/runs/`` + pin a manifest (T10.1/2).

``uv run python -m src.results.make_figures``. Reads the append-only run log, writes the
PLOTTED figures (F1 learning curves, F2 loss, F5 IQL/VDN/QMIX comparison, F6 scaling, plus
the §9.3 BOX final-distribution and HEATMAP capture matrix) to ``results/figures/``, and pins
``experiment_manifest.json`` (run count, algorithms, stages, seeds, config hash) so figure
drift is detectable (R8). F3 (GUI screenshots) + F4 (MCP comms) are captured artifacts, NOT
regenerated here. The comparison stage is the largest stage present (IQL diverges from CTDE).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from src.results.aggregate import load_runs
from src.results.plots import plot_comparison, plot_scaling, plot_two_agent_panels
from src.results.plots_extra import final_values_by_seed, plot_capture_heatmap, plot_final_distribution
from src.utils.config_loader import load_config


def _focus_stage(records: list[dict]) -> int:
    """The LARGEST stage with the most algorithm coverage (robust to a partial run).

    A partial run may have one slow stage reached by only some algorithms; the F5
    comparison needs a stage where the most arms are present, preferring the largest.
    """
    algos_by_stage: dict[int, set] = defaultdict(set)
    for rec in records:
        algos_by_stage[rec["stage"]].add(rec["algorithm"])
    most = max(len(algos) for algos in algos_by_stage.values())
    return max(stage for stage, algos in algos_by_stage.items() if len(algos) == most)


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
    not a proxy. Those fields post-date the headline matrix, so they live in their own
    append-only log; the figure is skipped (not faked) when that log is absent.
    """
    path = Path(cfg["paths"]["runs_dir"]) / "returns_history.jsonl"
    if not path.exists():
        return []
    records = load_runs(path)
    stage = _focus_stage(records)
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


def _variety_figures(records: list[dict], stage: int, fig_dir: Path) -> list[str]:
    """V3 §9.3 chart variety: the BOX (per-seed spread) + HEATMAP (algorithm x stage) figures.

    Each is SKIPPED — never faked — when the log lacks its data: the box plot needs at least
    one algorithm with final rounds at the focus stage, the heatmap needs at least one stage.
    """
    algos = {rec["algorithm"] for rec in records}
    saved = []
    if any(final_values_by_seed(records, "capture_rate", a, stage) for a in algos):
        out = plot_final_distribution(records, "capture_rate", stage, fig_dir / "final_distribution.png")
        saved.append(str(out))
    if any(rec["stage"] is not None for rec in records):
        saved.append(str(plot_capture_heatmap(records, "capture_rate", fig_dir / "capture_heatmap.png")))
    return saved


def main(cfg: dict | None = None) -> list[str]:
    """Regenerate the plotted figures + the manifest; return the figure paths."""
    cfg = cfg or load_config()
    fig_dir = Path(cfg["paths"]["figures_dir"])
    records = load_runs(Path(cfg["paths"]["runs_dir"]) / "history.jsonl")
    if not records:
        raise SystemExit("no runs in results/runs/history.jsonl — run scripts/run_results.py first")
    stage = _focus_stage(records)  # the largest stage with the most algorithm coverage
    grid = next(rec["grid"] for rec in records if rec["stage"] == stage)
    saved = [
        str(
            plot_two_agent_panels(  # §7.3a: BOTH agents' learning (cop capture / thief escape)
                records,
                "capture_rate",
                stage,
                [
                    ("cop", "Cop learning — capture rate (cop-training rounds)", "capture rate", None),
                    (
                        "thief",
                        "Thief learning — escape rate (thief-training rounds)",
                        "escape rate (1 - capture)",
                        lambda m: 1.0 - m,
                    ),
                ],
                f"Both agents' learning at {grid}x{grid} (mean±SE over seeds)",
                fig_dir / "learning_curves.png",
            )
        ),
        str(
            plot_two_agent_panels(  # §7.3b: the two NETS' losses (pooling would interleave them)
                records,
                "loss",
                stage,
                [
                    ("cop", "Cop net TD-loss (QMIX/VDN/IQL)", "TD loss", None),
                    ("thief", "Thief Double-DQN TD-loss", "TD loss", None),
                ],
                f"Per-network training loss at {grid}x{grid} (mean±SE over seeds)",
                fig_dir / "loss_curves.png",
            )
        ),
        str(plot_comparison(records, "capture_rate", stage, fig_dir / "baseline_comparison.png")),
        str(plot_scaling(records, "capture_rate", fig_dir / "scaling.png")),
    ]
    saved += _variety_figures(records, stage, fig_dir)  # V3 §9.3: BOX + HEATMAP families
    saved += _return_curves(cfg, fig_dir)
    manifest_path = Path(cfg["paths"]["experiment_manifest"])  # config-driven (no hardcoded path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(_manifest(cfg, records), indent=2), encoding="utf-8")
    print(f"[make_figures] {len(saved)} figures + manifest from {len(records)} records -> {fig_dir}")
    return saved


if __name__ == "__main__":  # pragma: no cover - module CLI entry (uv run -m)
    main()
