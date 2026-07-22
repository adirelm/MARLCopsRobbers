"""V3 §16 building-block contract — SDK facade, ApiGatekeeper, SelfPlayTrainer.

Every graded building block must (a) document Input/Output/Setup, (b) reject a bad
CONFIG with a specific ValueError via ``_validate_config``, (c) reject a wrong-typed
public INPUT with TypeError via ``_validate_input``, and (d) still accept everything
the real config + the existing callers legitimately pass. The MCP server factory and
the report send path are covered in ``test_building_blocks_edges.py``.
"""

from __future__ import annotations

import copy
import importlib
import json

import pytest

from src.api.gatekeeper import ApiGatekeeper
from src.sdk.sdk import MarlSDK
from src.services.trainer import SelfPlayTrainer

_BLOCKS = [
    "src.sdk.sdk",
    "src.api.gatekeeper",
    "src.services.trainer",
    "src.mcp.server_builder",
    "src.reporting.send",
]


@pytest.mark.parametrize("module_name", _BLOCKS)
def test_every_building_block_documents_input_output_setup(module_name):
    """Each targeted component's module docstring carries the literal I/O/Setup block."""
    doc = importlib.import_module(module_name).__doc__ or ""
    for section in ("Input:", "Output:", "Setup:"):
        assert section in doc, f"{module_name} docstring is missing {section!r}"


def test_sdk_validate_config_rejects_a_missing_section(cfg):
    """A config without the `compute` block the SDK reads is rejected at construction."""
    bad = copy.deepcopy(cfg)
    del bad["compute"]
    with pytest.raises(ValueError, match="compute"):
        MarlSDK(bad)


def test_sdk_validate_config_rejects_a_non_positive_thread_cap(cfg):
    """compute.num_threads must be a positive int (0 would disable the §5 governance)."""
    bad = copy.deepcopy(cfg)
    bad["compute"]["num_threads"] = 0
    with pytest.raises(ValueError, match="num_threads"):
        MarlSDK(bad)


def test_sdk_validate_input_rejects_a_non_string_algorithm(cfg):
    """train() is the public training input — a non-str algorithm is a TypeError."""
    with pytest.raises(TypeError, match="algorithm"):
        MarlSDK(cfg).train(7, seed=1)


def test_sdk_validate_input_rejects_a_non_int_seed(cfg):
    """A stringly-typed seed is rejected before any training work starts."""
    with pytest.raises(TypeError, match="seed"):
        MarlSDK(cfg).train("qmix", seed="1")


def test_sdk_accepts_the_real_config_and_valid_input(cfg):
    """Happy path: the real config constructs, and a valid-typed unknown arm still ValueErrors."""
    sdk = MarlSDK(cfg)
    assert sdk.build_env(h=2, w=2, num_cops=1) is not None
    with pytest.raises(ValueError, match="unknown algorithm"):
        sdk.train("ppo", seed=1)  # passes _validate_input; fails on the algorithm switch


def _limits(tmp_path, spec: dict):
    """Write a rate-limit spec to disk and return its path."""
    path = tmp_path / "rate_limits.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def test_gatekeeper_validate_config_rejects_a_spec_without_max_queue(tmp_path):
    """The overflow bound is mandatory — without it the FIFO queue is unbounded."""
    path = _limits(tmp_path, {"limits": {"gmail": {"per_minute": 5, "burst": 1}}})
    with pytest.raises(ValueError, match="max_queue"):
        ApiGatekeeper(path=path)


def test_gatekeeper_validate_config_rejects_a_non_positive_rate(tmp_path):
    """A channel with per_minute <= 0 could never drain its queue."""
    path = _limits(tmp_path, {"limits": {"gmail": {"per_minute": 0, "burst": 1}}, "max_queue": 4})
    with pytest.raises(ValueError, match="per_minute"):
        ApiGatekeeper(path=path)


def test_gatekeeper_validate_input_rejects_a_non_callable_call():
    """execute() takes a zero-arg thunk — anything else is a TypeError, not a KeyError."""
    with pytest.raises(TypeError, match="call"):
        ApiGatekeeper(clock=lambda: 0.0).execute("gmail", "not-a-thunk")


def test_gatekeeper_validate_input_rejects_a_non_string_channel():
    """The channel name is a str key into the configured limits."""
    with pytest.raises(TypeError, match="channel"):
        ApiGatekeeper(clock=lambda: 0.0).execute(7, lambda: None)


def test_gatekeeper_accepts_the_versioned_config_and_valid_input():
    """Happy path: the tracked config/rate_limits.json admits a valid call immediately."""
    assert ApiGatekeeper(clock=lambda: 0.0).execute("gmail", lambda: "ok") == "ok"


def _tiny(cfg: dict) -> dict:
    """Shrink the self-play knobs so a real stage constructs/runs fast."""
    c = copy.deepcopy(cfg)
    c["selfplay"]["episodes_per_round"] = 1
    c["selfplay"]["update_ratio"] = 1
    c["selfplay"]["rounds"] = 1
    c["algo"]["batch_episodes"] = 2
    c["replay"]["buffer_episodes"] = 8
    return c


def test_trainer_validate_config_rejects_a_missing_selfplay_key(cfg):
    """selfplay.rounds is read by train_stage — its absence fails fast at construction."""
    bad = _tiny(cfg)
    del bad["selfplay"]["rounds"]
    with pytest.raises(ValueError, match="rounds"):
        SelfPlayTrainer(bad, seed=0, h=2, w=2, num_cops=1)


def test_trainer_validate_config_rejects_a_non_positive_batch(cfg):
    """algo.batch_episodes drives every learner update — it must be >= 1."""
    bad = _tiny(cfg)
    bad["algo"]["batch_episodes"] = 0
    with pytest.raises(ValueError, match="batch_episodes"):
        SelfPlayTrainer(bad, seed=0, h=2, w=2, num_cops=1)


def test_trainer_validate_input_rejects_a_float_seed(cfg):
    """The stage dims + master seed are ints — a float seed is a TypeError."""
    with pytest.raises(TypeError, match="seed"):
        SelfPlayTrainer(_tiny(cfg), seed=1.5, h=2, w=2, num_cops=1)


def test_trainer_validate_input_rejects_a_float_round_count(cfg):
    """train_stage(rounds=...) is the public input — a float round count is a TypeError."""
    trainer = SelfPlayTrainer(_tiny(cfg), seed=0, h=2, w=2, num_cops=1)
    with pytest.raises(TypeError, match="rounds"):
        trainer.train_stage(rounds=1.5)


def test_trainer_accepts_the_real_config_and_valid_input(cfg):
    """Happy path: a valid stage still trains and returns one record per round."""
    history = SelfPlayTrainer(_tiny(cfg), seed=0, h=2, w=2, num_cops=1).train_stage(rounds=1)
    assert len(history) == 1 and history[0]["round"] == 0
