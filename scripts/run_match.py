"""Thin local full-match orchestrator — routes through the SDK (T6.1/T9.5; run manually).

Plays the 6-sub-game match over the in-memory MCP via ``SDK.run_local_match``,
assembling + validating the §3.5 report and DRY-RUN sending it. With ``--send`` it then
performs the REAL Gmail egress via ``SDK.send_final_report`` (needs ``GMAIL_SENDER`` /
``GMAIL_APP_PASSWORD`` in the env; idempotent — sends to the lecturer at most once).
Uses placeholder players unless players.local.yaml is present. Run:
``uv run python scripts/run_match.py [--send]``. All logic is in the SDK / services.
"""

from __future__ import annotations

import sys

from src.marl.nets.agent_net import RecurrentQNet
from src.reporting.players import load_players
from src.sdk.sdk import MarlSDK
from src.utils.config_loader import load_config


def main(cfg: dict | None = None, send: bool = False) -> dict:
    """Run the full local match; print §3.5 totals + dry-run ack. ``send`` emails for real."""
    cfg = cfg or load_config()
    sdk = MarlSDK(cfg)
    cop = RecurrentQNet(cfg, "cop", 2)
    thief = RecurrentQNet(cfg, "thief", 1)
    seed = int(cfg["training"]["seeds"][0])
    out = sdk.run_local_match(cop, thief, load_players(), seed)
    print(f"[match] {out['num_games']} sub-games | totals={out['report']['totals']} | ack={out['ack']}")
    if send:  # pragma: no cover - real Gmail egress; requires creds + network + explicit go
        result = sdk.send_final_report(out["report"])
        print(f"[email] sent={result['sent']} reason={result.get('reason', 'ok')} to={result.get('to')}")
    return out


if __name__ == "__main__":
    main(send="--send" in sys.argv)
