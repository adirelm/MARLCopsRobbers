"""Run the §9 bonus match over the wire — thin CLI over the SDK-safe surfaces.

Loads config, binds the four bearer ``WireClient``s from ``wire_match.groups`` (token
VALUES come from the env vars the config NAMES; never from tracked content), health-checks
every endpoint, drives the match via :class:`src.mcp.wire_referee.WireReferee`, writes the
draft §9.4 JSON to the git-ignored ``wire_match.draft_report`` path (it may carry real PII
once ``players*.local.yaml`` exist), and prints ``totals_by_group`` + both artifact paths.
Manual (real sockets; the P7 seed list must be filled first), not a CI gate:
``uv run python scripts/run_bonus_match.py``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.mcp.wire_client import WireClient
from src.mcp.wire_referee import WireReferee, build_draft_report
from src.mcp.wire_serve import resolve_token
from src.utils.config_loader import load_config

_ROLES = ("cop", "thief")
_GROUPS = ("group_1", "group_2")


def build_clients(cfg: dict, on_event) -> dict:
    """Return the referee's clients mapping, one bearer WireClient per group x role."""
    wire = cfg["wire_match"]
    clients: dict[str, dict[str, WireClient]] = {}
    for key in _GROUPS:
        spec = wire["groups"][key]
        clients[key] = {
            role: WireClient(
                spec[f"{role}_url"],
                resolve_token(spec),  # refuses an unset env var — never dials fail-open
                timeout_s=float(wire["timeout_s"]),
                retries=int(wire["retries"]),
                label=f"{key}-{role}",
                on_event=on_event,
            )
            for role in _ROLES
        }
    return clients


_REHEARSAL_SEEDS = [101, 202, 303, 404, 505, 606]  # the committed dress-rehearsal list


def _real_match_guards(cfg: dict) -> None:
    """Refuse to play a REAL match on rehearsal scaffolding (tripwires, not conveniences).

    A real match is one where group_2 is no longer the placeholder. Then: the P7 seed
    list must have been consciously replaced, and the partner's identity intake must
    exist — otherwise the draft silently ships '<PARTNER GROUP CODE>' placeholders.
    """
    wire = cfg["wire_match"]
    if wire["groups"]["group_2"]["name"] == "partner-group":
        return  # rehearsal mode — placeholders allowed everywhere
    if list(wire["seeds"]) == _REHEARSAL_SEEDS:
        raise SystemExit(
            "REAL match configured but wire_match.seeds is still the rehearsal list — freeze P7 seeds first"
        )
    if not Path("players.partner.local.yaml").exists():
        raise SystemExit(
            "REAL match configured but players.partner.local.yaml is missing — intake the partner identity"
        )


def main() -> dict:
    """Health-check, play the match, write the draft §9.4 JSON, print the artifacts."""
    cfg = load_config()
    wire = cfg["wire_match"]
    _real_match_guards(cfg)
    stamp = datetime.now(ZoneInfo(cfg["project"]["timezone"])).strftime("%Y%m%dT%H%M%S")
    referee = WireReferee(cfg, Path(wire["log_dir"]) / f"wire_log_{stamp}.jsonl")
    clients = build_clients(cfg, referee.log_event)
    for key in _GROUPS:
        for role in _ROLES:
            if not clients[key][role].health():
                raise SystemExit(f"health check FAILED for {key} {role} endpoint — not starting")
    result = referee.play_match(clients)
    draft = build_draft_report(cfg, result["sub_games"])
    draft_path = Path(wire["draft_report"])
    draft_path.parent.mkdir(parents=True, exist_ok=True)
    draft_path.write_text(json.dumps(draft, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[wire-match] totals_by_group: {result['totals_by_group']}")
    print(f"[wire-match] shareable JSONL log: {result['log_path']}")
    print(f"[wire-match] draft §9.4 report (git-ignored, mutual_agreement=false): {draft_path}")
    return {"result": result, "draft_path": str(draft_path)}


if __name__ == "__main__":
    main()
