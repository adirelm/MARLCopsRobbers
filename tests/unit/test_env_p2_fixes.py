"""P2-fix unit tests for CopsRobbersEnv (split out of test_env.py for size).

Covers the newer post-P2 fixes: a CAPTURE driven through env.step (winner=="cop",
capture True, scores present); step() on a terminated episode raises; eval_mode
zeroes reward shaping; per-agent visibility memory resets to 0 when an opponent
enters view and is re-initialized on reset(); plus the pre-reset guards on
step() and state() (both raise RuntimeError before the first reset()).
"""

from __future__ import annotations

import pytest

from src.marl.env.actions import Action
from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.marl.env.types import GlobalState


def _capture_setup(cfg) -> CopsRobbersEnv:
    """Reset an env, then plant a state where the cop captures next step.

    Cop at (0, 1), thief at (0, 0) on a 5x5 board: the cop moves LEFT onto the
    thief while the thief's UP move runs into the top edge (a no-op stay), so the
    cop lands on the thief -> capture this tick (white-box; the env exposes no
    public state setter for an exit-gate test).
    """
    env = CopsRobbersEnv(cfg, h=5, w=5, num_cops=1)
    env.reset(seed=7)
    env._state = GlobalState(
        cop_pos=((0, 1),),
        thief_pos=(0, 0),
        barriers=frozenset(),
        barriers_used=0,
        step=0,
        h=5,
        w=5,
        terminal=False,
    )
    return env


def test_capture_through_step(cfg):
    # Drive a real CAPTURE through env.step: cop LEFT onto the thief (thief UP is
    # a no-op into the top edge). winner=="cop", capture True, scores present.
    env = _capture_setup(cfg)
    _, _, terminated, info = env.step({"cop_0": Action.LEFT, "thief": Action.UP})
    assert terminated is True
    assert info["winner"] == "cop"
    assert info["capture"] is True
    assert info["scores"] is not None
    assert set(info["scores"].keys()) == {"cop", "thief"}


def test_step_after_termination_raises(cfg):
    # Stepping a terminated episode must raise (not silently re-advance).
    env = _capture_setup(cfg)
    _, _, terminated, _ = env.step({"cop_0": Action.LEFT, "thief": Action.UP})
    assert terminated is True
    with pytest.raises(RuntimeError, match="terminated episode"):
        env.step({"cop_0": Action.LEFT, "thief": Action.UP})


def test_step_before_reset_raises(cfg):
    # Calling step() before the first reset() must raise (covers env.py:105).
    env = CopsRobbersEnv(cfg, h=5, w=5, num_cops=1)
    with pytest.raises(RuntimeError, match="before reset"):
        env.step({})


def test_state_before_reset_raises(cfg):
    # Calling state() before the first reset() must raise (covers env.py:133).
    env = CopsRobbersEnv(cfg, h=5, w=5, num_cops=1)
    with pytest.raises(RuntimeError, match="before reset"):
        env.state()


def test_eval_mode_zeroes_reward_shaping(cfg):
    # With shaping ON (train) the cop reward differs from the eval-mode reward,
    # which gates shaping OFF (shaping_eval_enabled=False) -> only the step
    # penalty remains. The shaping delta must be non-trivial here.
    assert cfg["reward"]["shaping_enabled"] is True
    assert cfg["reward"]["shaping_eval_enabled"] is False
    env_train = CopsRobbersEnv(cfg, h=5, w=5, num_cops=1)
    env_train.reset(seed=7)
    _, r_train, _, _ = env_train.step({"cop_0": Action.DOWN, "thief": Action.UP})
    env_eval = CopsRobbersEnv(cfg, h=5, w=5, num_cops=1)
    env_eval.reset(seed=7)
    _, r_eval, _, _ = env_eval.step({"cop_0": Action.DOWN, "thief": Action.UP}, eval_mode=True)
    # Eval-mode cop reward == train cop reward minus the (non-zero) shaping term.
    assert r_train["cop_0"] != r_eval["cop_0"]
    assert r_eval["cop_0"] == pytest.approx(-cfg["reward"]["step_penalty"])


def test_visibility_memory_resets_when_opponent_in_view(cfg):
    # Plant the cop OUTSIDE then INSIDE the thief's view radius and assert the
    # per-agent steps_since_seen counter actually increments then resets to 0.
    env = CopsRobbersEnv(cfg, h=5, w=5, num_cops=1)
    env.reset(seed=7)
    far = GlobalState(
        cop_pos=((4, 4),),
        thief_pos=(0, 0),
        barriers=frozenset(),
        barriers_used=0,
        step=1,
        h=5,
        w=5,
        terminal=False,
    )
    env._state = far
    env._memory["thief"].steps_since_seen = 3
    # cop UP from (4,4)->(3,4): manhattan to thief(0,0) == 7 > radius 2 -> not seen.
    env.step({"cop_0": Action.UP, "thief": Action.DOWN})
    assert env._memory["thief"].steps_since_seen == 4
    # Now place the cop adjacent to the thief -> in view -> counter resets to 0.
    env._state = GlobalState(
        cop_pos=((0, 1),),
        thief_pos=(0, 0),
        barriers=frozenset(),
        barriers_used=0,
        step=2,
        h=5,
        w=5,
        terminal=False,
    )
    env.step({"cop_0": Action.DOWN, "thief": Action.DOWN})
    assert env._memory["thief"].steps_since_seen == 0


def test_visibility_memory_resets_on_reset(cfg):
    env = CopsRobbersEnv(cfg, h=5, w=5, num_cops=1)
    env.reset(seed=7)
    # Force a stale per-agent counter, then confirm reset() re-initializes it.
    env._memory["thief"].steps_since_seen = 17
    env._memory["cop_0"].steps_since_seen = 9
    env.reset(seed=7)
    # reset() rebuilds fresh memory then updates it from the spawn (dist>radius),
    # so the spawn-tick counters are 1 (opponent out of view at a >radius spawn).
    assert env._memory["thief"].steps_since_seen == 1
    assert env._memory["cop_0"].steps_since_seen == 1
