"""Reproduce the ANALYSIS §12 exploitability table — 5 head-to-head arms via the SDK.

The serving cop (always greedy) plays 60-game blocks against: the scripted flee baseline,
a uniform-random thief, and our self-play thief at each ``matchup_eval.thief_eps_grid``
noise level. Run: ``uv run python scripts/eval_matchup.py`` (CPU, a few minutes). Block
shape comes from the ``matchup_eval`` config; all logic is in the SDK / services.
"""

from __future__ import annotations

from src.sdk.sdk import MarlSDK
from src.utils.config_loader import load_config


def main(cfg: dict | None = None) -> list[dict]:
    """Run all arms; print one summary line each; return the arm dicts."""
    cfg = cfg or load_config()
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
        results.append({"label": label, **out})
    return results


if __name__ == "__main__":
    main()
