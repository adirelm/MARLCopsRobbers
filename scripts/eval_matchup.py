"""Reproduce the ANALYSIS §12 exploitability table — 5 head-to-head arms via the SDK.

The serving cop (always greedy) plays 60-game blocks against: the scripted flee baseline,
a uniform-random thief, and our self-play thief at each ``matchup_eval.thief_eps_grid``
noise level. Run: ``uv run python scripts/eval_matchup.py`` (CPU, a few minutes). Block
shape comes from the ``matchup_eval`` config; all logic is in the SDK / services.
``--base-seed`` overrides the block start (defaults to the config value), ``--json-out``
writes the arm summaries to a JSON artifact — the committed §12 evidence blocks
``results/matchup/block_1000.json`` / ``block_5000.json`` are produced exactly this way.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.sdk.sdk import MarlSDK
from src.utils.config_loader import load_config


def main(cfg: dict | None = None, base_seed: int | None = None, json_out: str | None = None) -> list[dict]:
    """Run all arms; print one summary line each; return (and optionally dump) the arm dicts."""
    cfg = cfg or load_config()
    if base_seed is not None:
        cfg["matchup_eval"]["base_seed"] = int(base_seed)
    sdk = MarlSDK(cfg)
    arms: list[tuple[str, str, float]] = [
        ("cop vs scripted flee", "flee", 0.0),
        ("cop vs uniform random", "random", 0.0),
    ]
    arms += [
        (f"cop vs OUR thief (eps={eps:.2f})", "net", float(eps))
        for eps in cfg["matchup_eval"]["thief_eps_grid"]
    ]
    results = []
    for label, kind, eps in arms:
        out = sdk.run_matchup_eval(kind, thief_eps=eps)
        barriers = out["cop_actions"].get("PLACE_BARRIER", 0)
        print(
            f"{label:32s} captures {out['captures']:3d}/{out['games']}  "
            f"barriers {barriers:4d}  moves {out['moves']}"
        )
        results.append({"label": label, "thief_kind": kind, **out})
    if json_out:
        out_path = Path(json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"[eval_matchup] wrote {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ANALYSIS §12 exploitability arms")
    parser.add_argument("--base-seed", type=int, default=None, help="seed-block start (default: config)")
    parser.add_argument("--json-out", default=None, help="write the arm summaries to this JSON path")
    args = parser.parse_args()
    main(base_seed=args.base_seed, json_out=args.json_out)
