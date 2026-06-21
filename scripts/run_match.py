"""Thin local full-match orchestrator — routes through the SDK (T6.1; run manually).

Plays the 6-sub-game match over the in-memory MCP via ``SDK.run_local_match``,
assembling + validating the §3.5 report and DRY-RUN sending it (no email at P6).
Uses placeholder players (real names/ids live in git-ignored players.local.yaml).
Run: ``uv run python scripts/run_match.py``. All logic is in the SDK / services.
"""

from __future__ import annotations

from src.marl.nets.agent_net import RecurrentQNet
from src.reporting.players import load_players
from src.sdk.sdk import MarlSDK
from src.utils.config_loader import load_config


def main(cfg: dict | None = None) -> dict:
    """Run the full local match and print the assembled §3.5 totals + dry-run ack."""
    cfg = cfg or load_config()
    cop = RecurrentQNet(cfg, "cop", 2)
    thief = RecurrentQNet(cfg, "thief", 1)
    seed = int(cfg["training"]["seeds"][0])
    out = MarlSDK(cfg).run_local_match(cop, thief, load_players(), seed)
    print(f"[match] {out['num_games']} sub-games | totals={out['report']['totals']} | ack={out['ack']}")
    return out


if __name__ == "__main__":
    main()
