"""P2 runtime exit-gate: reset()/step() NEVER leak the global state (T2.3).

This is the single most-graded modeling requirement (CTDE train/exec split,
config §52/§6.1): the executor-facing reset()/step() payloads carry ONLY LOCAL
Observations + action_mask / timestamp / winner — never a GlobalState, never a
GlobalState field name as a dict key, never the full barrier set or an absolute
opponent position. The global state EXISTS but is reachable ONLY via the
sanctioned train-only env.state() accessor.

The recursive scanner has TEETH: ``_assert_no_global_leak`` walks every nested
value in the reset()/step() obs_dict + info — including ``set``/``frozenset``
members and object-dtype ``np.ndarray`` cells (the obs ``image``) — and would
FAIL if any GlobalState instance or GlobalState-field-named key were smuggled
into the payload. The negative-control test plants a deliberate leak in an info
value, an obs object-array, AND a set, proving the scanner catches each.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.marl.env.actions import Action
from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.marl.env.types import GlobalState, Observation

# Every GlobalState field name is FORBIDDEN as a dict key anywhere in an
# obs/info payload — derived from the dataclass so a newly-added train-only
# field is guarded automatically (never a hand-maintained list). The explicit
# board-dim / global-bag keys the executor must never expose are FOLDED IN here
# so there is a SINGLE source of forbidden keys (no parallel list to drift).
GLOBAL_FIELD_NAMES = {f.name for f in dataclasses.fields(GlobalState)} | {
    "state",
    "global_state",
}
LOCAL_OBS_KEYS = {"image", "scalars"}


def _assert_no_global_leak(node: object, path: str = "root") -> None:
    """Recursively assert ``node`` carries NO GlobalState / global field key.

    Teeth: asserts no value is a GlobalState instance and no dict key equals any
    GlobalState field name; recurses through dicts, lists, tuples, sets, and
    object-dtype numpy arrays (numeric arrays hold no python objects, so they
    are skipped — a leaked GlobalState could never live inside a float buffer).

    Args:
        node: The value (possibly nested) to scan.
        path: A dotted breadcrumb used in the assertion message on failure.
    """
    assert not isinstance(node, GlobalState), f"GlobalState leaked at {path}"
    if isinstance(node, dict):
        for key, value in node.items():
            assert key not in GLOBAL_FIELD_NAMES, f"global field key {key!r} at {path}"
            _assert_no_global_leak(value, f"{path}.{key}")
    elif isinstance(node, (set, frozenset)):
        for i, value in enumerate(node):
            _assert_no_global_leak(value, f"{path}{{{i}}}")
    elif isinstance(node, np.ndarray):
        if node.dtype == object:
            for i, value in enumerate(node.flat):
                _assert_no_global_leak(value, f"{path}.flat[{i}]")
    elif isinstance(node, (list, tuple)):
        for i, value in enumerate(node):
            _assert_no_global_leak(value, f"{path}[{i}]")


def _scan_payload(obs_dict: dict, info: dict) -> None:
    """Apply the no-leak scan + the per-Observation key contract to a payload."""
    _assert_no_global_leak(obs_dict, "obs_dict")
    _assert_no_global_leak(info, "info")
    for agent, obs in obs_dict.items():
        assert set(obs.keys()) == LOCAL_OBS_KEYS, f"{agent} obs keys != {LOCAL_OBS_KEYS}"


def _greedy_joint(env: CopsRobbersEnv, num_cops: int) -> dict[str, Action]:
    """Build a joint action that drives every cop toward the thief (pursuit).

    Uses the sanctioned train-only ``state()`` to pick each cop's distance-
    reducing step — a deterministic pursuit policy that forces a capture within
    a few ticks on a small grid (so the leak scan exercises a CAPTURE terminal).
    """
    state = env.state()
    thief = state.thief_pos
    joint: dict[str, Action] = {"thief": Action.UP}
    for i in range(num_cops):
        cr, cc = state.cop_pos[i]
        if cr != thief[0]:
            joint[f"cop_{i}"] = Action.DOWN if thief[0] > cr else Action.UP
        else:
            joint[f"cop_{i}"] = Action.RIGHT if thief[1] > cc else Action.LEFT
    return joint


def test_full_episode_never_leaks_global_state(cfg):
    env = CopsRobbersEnv(cfg, h=3, w=3, num_cops=1)
    obs_dict, info = env.reset(seed=11)
    _scan_payload(obs_dict, info)

    max_moves = cfg["game"]["max_moves"]
    for _ in range(max_moves):
        joint = {"cop_0": Action.DOWN, "thief": Action.UP}
        obs_dict, _reward, terminated, info = env.step(joint)
        _scan_payload(obs_dict, info)
        if terminated:
            break


def test_two_cop_stage_never_leaks_global_state(cfg):
    """The 4x4 2-cop training stage (§7.2 credit assignment) must not leak."""
    env = CopsRobbersEnv(cfg, h=4, w=4, num_cops=2)
    obs_dict, info = env.reset(seed=3)
    _scan_payload(obs_dict, info)
    expected = {"cop_0", "cop_1", "thief"}
    assert set(obs_dict.keys()) == expected

    max_moves = cfg["game"]["max_moves"]
    for _ in range(max_moves):
        joint = {"cop_0": Action.DOWN, "cop_1": Action.UP, "thief": Action.RIGHT}
        obs_dict, _reward, terminated, info = env.step(joint)
        _scan_payload(obs_dict, info)
        if terminated:
            break


def test_capture_episode_never_leaks_global_state(cfg):
    """A CAPTURE terminal (winner=='cop') payload must also stay leak-free."""
    env = CopsRobbersEnv(cfg, h=4, w=4, num_cops=2)
    env.reset(seed=9)
    max_moves = cfg["game"]["max_moves"]
    winner = None
    captured = False
    for _ in range(max_moves):
        joint = _greedy_joint(env, num_cops=2)
        obs_dict, _reward, terminated, info = env.step(joint)
        _scan_payload(obs_dict, info)
        if terminated:
            winner, captured = info["winner"], info["capture"]
            break
    # Greedy pursuit on a 4x4 must end in a capture (winner == "cop").
    assert captured is True
    assert winner == "cop"


def test_state_accessor_is_sanctioned_globalstate(cfg):
    env = CopsRobbersEnv(cfg, h=3, w=3, num_cops=1)
    env.reset(seed=11)
    # The global state EXISTS — but ONLY via the sanctioned train-only accessor.
    assert isinstance(env.state(), GlobalState)


def test_scanner_catches_leak_in_info_value(cfg):
    """Negative control (i): a GlobalState smuggled into an info value is caught."""
    leaked = GlobalState((0, 0), (1, 1), frozenset(), 0, 0, 3, 3)
    with pytest.raises(AssertionError):
        _assert_no_global_leak({"winner": "cop", "sneaky": leaked}, "info")


def test_scanner_catches_leak_in_obs_object_array(cfg):
    """Negative control (ii): a GlobalState in an obs object-array is caught."""
    leaked = GlobalState((0, 0), (1, 1), frozenset(), 0, 0, 3, 3)
    image = np.array([1.0, leaked], dtype=object)
    obs = {"thief": {"image": image, "scalars": np.zeros(6, dtype=np.float32)}}
    with pytest.raises(AssertionError):
        _assert_no_global_leak(obs, "obs_dict")


def test_scanner_catches_leak_in_set(cfg):
    """Negative control (iii): a GlobalState planted inside a set is caught."""
    leaked = GlobalState((0, 0), (1, 1), frozenset(), 0, 0, 3, 3)
    with pytest.raises(AssertionError):
        _assert_no_global_leak({"members": {leaked}}, "info")


def test_scanner_catches_global_field_named_key():
    """A GlobalState-field-named key (e.g. ``barriers``) must be caught."""
    with pytest.raises(AssertionError):
        _assert_no_global_leak({"barriers": frozenset()}, "info")


def test_scanner_passes_numeric_image_array():
    """A plain numeric obs image (no python objects) scans clean (no false fail)."""
    image = np.zeros((5, 5, 5), dtype=np.float32)
    _assert_no_global_leak({"thief": {"image": image, "scalars": image[0, 0]}}, "obs_dict")


def test_observation_keys_are_exactly_local():
    assert set(Observation.__annotations__.keys()) == LOCAL_OBS_KEYS
