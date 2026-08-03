"""Net factories behind the SDK facade — the UNTRAINED vs SERVING distinction.

Input: the loaded config, a role (``"cop"`` / ``"thief"``) and an optional agent count.
Output: a ``RecurrentQNet`` — randomly initialised (:func:`fresh_net`) or restored from
the committed trained bundle (:func:`serving_net`).
Setup: :func:`serving_net` reads ``paths.cop_model`` / ``paths.thief_model``; both
bundles are tracked in-repo, so no download or training run is needed.

WHY THE SPLIT (the 2026-07-22 defect): every deliverable-producing entry point —
the §3.5 match orchestrator, the CLI, the F4 comms capture and the localhost demo
servers — called ``fresh_net`` and therefore played with RANDOM weights. Nothing
seeds torch globally outside the trainer, so each process drew different nets and the
§3.5 report was irreproducible (four consecutive runs: cop 75/30/105/45). The trained
bundles were sitting in ``deploy/model/`` being served by the CLOUD servers only.
``serving_net`` is the one way to obtain the policy we actually ship; ``fresh_net``
survives for training and tests, where an untrained net is the point.
"""

from __future__ import annotations

from pathlib import Path

_DEFAULT_AGENTS = {"cop": 2, "thief": 1}
_MODEL_KEY = {"cop": "cop_model", "thief": "thief_model"}


def _agents(role: str, n_agents: int | None) -> int:
    """Resolve the agent count — explicit wins, else the role default (cop 2 / thief 1)."""
    return n_agents if n_agents is not None else _DEFAULT_AGENTS.get(role, 1)


def fresh_net(cfg: dict, role: str, n_agents: int | None = None) -> object:
    """Build an UNTRAINED role net (random weights).

    For training and for tests that need a blank net. NOT for anything that produces a
    graded artifact — use :func:`serving_net` there.
    """
    from src.marl.nets.agent_net import RecurrentQNet  # noqa: PLC0415 — lazy: keep import light

    return RecurrentQNet(cfg, role, _agents(role, n_agents))


def serving_net(cfg: dict, role: str, n_agents: int | None = None) -> object:
    """Load the TRAINED role net from its configured bundle — the policy we ship.

    Raises:
        ValueError: If ``role`` is not cop/thief, or its ``paths.*_model`` key is absent.
        FileNotFoundError: If the configured bundle is missing. Deliberately fail-fast:
            silently falling back to an untrained net is the exact defect this replaces,
            and a silent fallback would make a bad report look like a good one.
    """
    from src.services.checkpoints import load_agent_weights  # noqa: PLC0415 — lazy: torch import

    if role not in _MODEL_KEY:
        raise ValueError(f"serving_net: role must be cop/thief, got {role!r}")
    key = _MODEL_KEY[role]
    path = cfg.get("paths", {}).get(key)
    if not path:
        raise ValueError(f"serving_net: config paths.{key} is not set — cannot serve a trained {role}")
    if not Path(path).exists():
        raise FileNotFoundError(f"serving_net: trained {role} bundle missing at {path} (paths.{key})")
    return load_agent_weights(path, cfg, role, _agents(role, n_agents))
