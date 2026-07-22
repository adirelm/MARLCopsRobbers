"""Every deliverable path must serve the TRAINED policy, never a random net.

The 2026-07-22 defect: the §3.5 match orchestrator, the CLI, the F4 comms capture and
both localhost demo servers all built their players with ``fresh_net`` — untrained,
unseeded, different weights every process. The §3.5 report was therefore irreproducible
(four consecutive runs: cop 75 / 30 / 105 / 45) and did not describe the shipped policy.

These tests assert the property that was missing, at the level it was missing: the
ENTRY POINTS, not the loader. A test that only checked ``serving_net`` in isolation
would still have passed while every caller used ``fresh_net``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.sdk.sdk import MarlSDK
from src.utils.config_loader import load_config

# The modules whose output is graded — each must serve trained weights.
_DELIVERABLE_SOURCES = (
    "scripts/run_match.py",  # §3.5 emailed report
    "src/cli.py",  # user-facing match
    "src/results/comms.py",  # F4 MCP comms figure
    "src/mcp/localhost_cop.py",  # §5.3 Stage-1 demo server
    "src/mcp/localhost_thief.py",
)


@pytest.mark.parametrize("source", _DELIVERABLE_SOURCES)
def test_deliverable_entry_points_do_not_use_untrained_nets(source):
    """No graded path may call ``fresh_net`` — that is the defect, spelled out."""
    text = Path(source).read_text(encoding="utf-8")
    assert "fresh_net" not in text, (
        f"{source} builds players with fresh_net (UNTRAINED, unseeded random weights). "
        "Graded artifacts must use MarlSDK.serving_net."
    )


@pytest.mark.parametrize("role", ["cop", "thief"])
def test_configured_trained_bundle_exists(role):
    """``paths.{role}_model`` must be set AND present — a fresh clone must reproduce the match."""
    path = load_config()["paths"][f"{role}_model"]
    assert Path(path).exists(), f"trained {role} bundle missing at {path}"


@pytest.mark.parametrize("role", ["cop", "thief"])
def test_serving_net_is_reproducible_across_calls(role):
    """Two ``serving_net`` calls give bit-identical weights; ``fresh_net`` does not.

    This is the actual determinism property the §3.5 report depends on. The fresh_net
    half is asserted too, so the test fails if serving_net ever silently degrades into
    a random-net factory (which would otherwise look like a pass).
    """
    sdk = MarlSDK(load_config())

    def fingerprint(net):
        return float(torch.cat([p.flatten() for p in net.parameters()]).sum())

    assert fingerprint(sdk.serving_net(role)) == fingerprint(sdk.serving_net(role))
    assert fingerprint(sdk.fresh_net(role)) != fingerprint(sdk.fresh_net(role))


def test_serving_net_fails_loudly_when_the_bundle_is_missing(tmp_path):
    """A missing bundle raises rather than falling back to an untrained net.

    Silent fallback is what made the original defect invisible: a bad report looked
    exactly like a good one.
    """
    cfg = load_config()
    cfg["paths"] = {**cfg["paths"], "cop_model": str(tmp_path / "absent.pt")}
    with pytest.raises(FileNotFoundError, match="bundle missing"):
        MarlSDK(cfg).serving_net("cop")


def test_serving_net_rejects_an_unknown_role():
    """An unknown role is a ValueError, not a KeyError from deep inside the loader."""
    with pytest.raises(ValueError, match="cop/thief"):
        MarlSDK(load_config()).serving_net("referee")
