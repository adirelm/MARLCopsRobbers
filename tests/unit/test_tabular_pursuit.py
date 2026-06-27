"""RED->GREEN tests for the 3x3 zero-sum tabular pursuit adapter (P-bonus, L11 §5)."""

from __future__ import annotations

import pytest

from src.marl.baselines.tabular_pursuit import TabularPursuit


def _toward(src: int, dst: int, grid: int) -> int:
    """Action index (0..3) that moves cell ``src`` one step toward cell ``dst``."""
    sr, sc = divmod(src, grid)
    dr, dc = divmod(dst, grid)
    if dr != sr and abs(dr - sr) >= abs(dc - sc):
        return 1 if dr > sr else 0  # DOWN / UP
    return 3 if dc > sc else 2  # RIGHT / LEFT


def test_reset_returns_position_pair(cfg):
    """reset returns a ``(cop_cell, thief_cell)`` key, each in ``[0, grid*grid)``."""
    env = TabularPursuit(cfg)
    key = env.reset(seed=1)
    assert isinstance(key, tuple) and len(key) == 2
    assert all(isinstance(c, int) and 0 <= c < env.grid * env.grid for c in key)


def test_n_actions_is_four(cfg):
    """The tabular game uses the four directional moves only (no barrier / stay)."""
    assert TabularPursuit(cfg).n_actions == 4


def test_step_signature(cfg):
    """step returns ``(state_key, float zero-sum reward, bool terminated)``."""
    env = TabularPursuit(cfg)
    env.reset(seed=1)
    key, r, done = env.step(0, 0)
    assert isinstance(key, tuple) and isinstance(r, float) and isinstance(done, bool)


def test_chase_yields_capture_reward(cfg):
    """Cop + thief moving toward each other collide → winner=cop → +capture_reward."""
    env = TabularPursuit(cfg)
    key = env.reset(seed=3)
    done, r = False, 0.0
    for _ in range(env.max_moves):
        cop, thief = key
        key, r, done = env.step(_toward(cop, thief, env.grid), _toward(thief, cop, env.grid))
        if done:
            break
    assert done and r == pytest.approx(cfg["minimax_q"]["capture_reward"])
