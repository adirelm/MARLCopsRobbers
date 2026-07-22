"""§9 wire-agent SERVING factories — our endpoints for the rehearsal and match day.

The one place that binds the SHIPPED §9 policy lineup (README §9: the trained
serving cop net; the thief as :class:`AdaptiveThiefPolicy` over the trained net —
exactly what the cloud servers act with, see ``src.mcp.cloud._match_policy``) to
the brief-§2 HTTP adapters in :mod:`src.mcp.wire_agent`. Ports are PARSED from the
SAME ``wire_match.groups`` base URLs the referee dials (single source — a URL/port
mismatch is therefore impossible); bearer token VALUES come from the env vars the
config NAMES (``token_env``), never from tracked content. ``scripts/`` may import
this module (src.mcp.* is inside the sanctioned script import surface).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from urllib.parse import urlsplit

from src.mcp.wire_agent import WireAgent, make_wire_agent
from src.sdk.sdk import MarlSDK

_ROLES = ("cop", "thief")


def shipped_policy(cfg: dict, role: str, sdk: MarlSDK | None = None) -> object:
    """Return the §9 lineup acting policy for ``role`` over the TRAINED serving net.

    Cop: the serving net behind the SDK acting seam (greedy RecurrentPolicy).
    Thief: :class:`~src.services.bonus_policies.AdaptiveThiefPolicy` — flee primary,
    auto-switch to the trained net on first barrier sighting (the README §9 lineup).
    """
    sdk = sdk or MarlSDK(cfg)
    net = sdk.serving_net(role)
    if role == "thief":
        from src.services.bonus_policies import AdaptiveThiefPolicy  # noqa: PLC0415 — lazy (mirrors cloud.py)

        return sdk.build_policy(role, AdaptiveThiefPolicy(cfg, net), 1)
    return sdk.build_policy(role, net, 1)


def conformance_policy(cfg: dict, role: str) -> object:
    """A SCRIPTED test policy for the brief-§2 protocol-conformance sub-game.

    Uniform-random over legal actions for either role — enough to exercise every
    payload path while revealing NOTHING about the match lineup (the whole point
    of offering the conformance run with a scripted agent).
    """
    del role  # role-agnostic by design: legality comes from the mask alone
    from src.services.matchup_eval import UniformRandomPolicy  # noqa: PLC0415 — lazy (script surface)

    return UniformRandomPolicy()


def group_ports(cfg: dict, group_key: str) -> dict[str, int]:
    """Return ``role -> TCP port`` parsed from the group's configured base URLs.

    Raises:
        ValueError: When a URL names no explicit port — we then cannot know where
            to bind, so refuse rather than guess (the P8 fault would be silent).
    """
    spec = cfg["wire_match"]["groups"][group_key]
    ports: dict[str, int] = {}
    for role in _ROLES:
        port = urlsplit(spec[f"{role}_url"]).port
        if port is None:
            raise ValueError(f"{group_key} {role}_url names no port — cannot bind a wire agent to it")
        ports[role] = int(port)
    return ports


def local_group_keys(cfg: dict) -> list[str]:
    """Return the group keys whose BOTH base URLs point at the rehearsal host.

    These are the endpoints WE can serve on this machine; on match day the
    partner's group carries their remote base URLs and is filtered out here.
    """
    host = cfg["wire_match"]["rehearsal"]["host"]
    groups = cfg["wire_match"]["groups"]
    return [
        k for k, spec in groups.items() if all(urlsplit(spec[f"{r}_url"]).hostname == host for r in _ROLES)
    ]


def resolve_token(spec: dict) -> str:
    """Resolve the group's bearer token VALUE from the env var the config NAMES.

    Raises:
        ValueError: When the env var is unset/empty — an empty token would make
            every POST effectively unauthenticated, so refuse to serve.
    """
    token = os.environ.get(spec["token_env"], "")
    if not token:
        raise ValueError(
            f"bearer token env var {spec['token_env']} is unset — refusing to serve unauthenticated"
        )
    return token


def start_group_agents(
    cfg: dict,
    group_keys: list[str],
    policy_factory: Callable[[str, str], object] | None = None,
    sdk: MarlSDK | None = None,
) -> list[WireAgent]:
    """Start ONE single-role wire agent per (group, role); return the started handles.

    Args:
        cfg: The loaded config (reads ``wire_match.groups``; bind host per wire_agent).
        group_keys: ``wire_match.groups`` keys to serve (e.g. both for the rehearsal).
        policy_factory: ``(group_key, role) -> policy`` — a FRESH acting policy per
            server (each holds its own hidden state / flee memory). Defaults to
            :func:`shipped_policy`, the lineup we field in the real match.
        sdk: Optional shared ``MarlSDK`` (only used by the default factory).

    Returns:
        The started :class:`WireAgent` handles, cop before thief per group.

    Raises:
        ValueError: On a missing token/port; any already-started agents are closed.
    """
    if policy_factory is None:
        shared = sdk or MarlSDK(cfg)

        def policy_factory(_key: str, role: str) -> object:
            return shipped_policy(cfg, role, shared)

    agents: list[WireAgent] = []
    try:
        for key in group_keys:
            spec = cfg["wire_match"]["groups"][key]
            token = resolve_token(spec)
            for role, port in group_ports(cfg, key).items():
                agents.append(make_wire_agent(cfg, role, policy_factory(key, role), token, port))
    except Exception:
        for agent in agents:
            agent.close()
        raise
    return agents
