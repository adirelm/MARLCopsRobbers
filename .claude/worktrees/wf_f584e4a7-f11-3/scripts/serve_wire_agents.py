"""Serve OUR §9 wire-agent endpoints (partner brief §2) — rehearsal / match-day server.

Starts ONE single-role stdlib HTTP wire agent per (group, role) named on argv —
default: every ``wire_match.groups`` entry whose base URLs point at the rehearsal
host — each acting with the SHIPPED lineup (trained serving cop net; the
AdaptiveThiefPolicy thief) on the port its configured base URL names. Bearer token
VALUES come from the env vars the config NAMES (``token_env``) — the script refuses
to serve unauthenticated. Blocks until Ctrl-C / SIGTERM. Manual (real sockets), not
a CI gate::

    WIRE_GROUP_1_TOKEN=... WIRE_GROUP_2_TOKEN=... \
        uv run python scripts/serve_wire_agents.py group_1 group_2
"""

from __future__ import annotations

import sys
import time

from src.mcp.wire_serve import conformance_policy, group_ports, local_group_keys, start_group_agents
from src.utils.config_loader import load_config


def main(argv: list[str] | None = None) -> None:
    """Start the requested groups' wire agents and serve until interrupted."""
    cfg = load_config()
    args = list(sys.argv[1:] if argv is None else argv)
    conformance = "--conformance" in args  # brief §2: scripted test agent, NOT the lineup
    keys = [a for a in args if not a.startswith("--")] or local_group_keys(cfg)
    if not keys:
        raise SystemExit(
            "no serveable groups: pass group keys or point wire_match.groups URLs at the rehearsal host"
        )
    factory = (lambda _key, role: conformance_policy(cfg, role)) if conformance else None
    agents = start_group_agents(cfg, keys, policy_factory=factory)
    try:
        for key in keys:
            for role, port in group_ports(cfg, key).items():
                print(f"[wire-serve] {key} {role} listening on port {port}")
        mode = "CONFORMANCE (scripted)" if conformance else "MATCH LINEUP"
        print(f"[wire-serve] READY — {mode}; serving {len(agents)} agent(s); Ctrl-C to stop")
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("[wire-serve] stopping")
    finally:
        for agent in agents:
            agent.close()


if __name__ == "__main__":
    main()
