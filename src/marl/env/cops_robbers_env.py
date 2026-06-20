"""The CopsRobbersEnv runtime — the CTDE train/exec boundary (T2.3).

`CopsRobbersEnv` drives one sub-game: ``reset`` seeds a spawn (cop+thief at
Manhattan distance > view radius) and returns per-agent LOCAL observations;
``step`` resolves the simultaneous joint action (transition.py), updates the
env-owned per-role visibility memory, and returns the
``(obs_dict, reward_dict, terminated, info)`` 4-tuple. Crucially, reset/step
payloads carry ONLY LOCAL Observations + action_mask / timestamp / winner — the
global state is reachable ONLY via the sanctioned train-only ``state()``
accessor (the single most-graded CTDE requirement; see test_step_no_leak.py).
All tunables come from config; nothing is hardcoded (CLAUDE.md §4).
"""

from __future__ import annotations

import datetime as dt
from random import Random

from src.marl.env import _env_helpers as helpers
from src.marl.env.actions import Action
from src.marl.env.observation import VisibilityMemory, view_radius
from src.marl.env.reward import RewardModel
from src.marl.env.scorer import Scorer
from src.marl.env.transition import resolve_joint_action
from src.marl.env.types import GlobalState, Observation


class CopsRobbersEnv:
    """One MARL Cops & Robbers sub-game with a strict CTDE train/exec split."""

    def __init__(
        self, cfg: dict, h: int | None = None, w: int | None = None, num_cops: int | None = None
    ) -> None:
        """Bind the env to a grid size + cop count (defaults from config).

        Args:
            cfg: The loaded project config (config/config.yaml).
            h: Board rows; defaults to ``game.grid_size``.
            w: Board columns; defaults to ``game.grid_size``.
            num_cops: Number of cops; defaults to ``env.num_cops``.
        """
        self._cfg = cfg
        grid = cfg["game"]["grid_size"]
        self._h = grid if h is None else h
        self._w = grid if w is None else w
        self._num_cops = cfg["env"]["num_cops"] if num_cops is None else num_cops
        self._reward = RewardModel(cfg)
        self._scorer = Scorer(cfg)
        self._state: GlobalState | None = None
        self._memory: dict[str, VisibilityMemory] = helpers.fresh_memory(self._num_cops)

    def reset(self, seed: int | None = None) -> tuple[dict[str, Observation], dict]:
        """Seed a spawn and return per-agent LOCAL obs + reset info.

        Args:
            seed: Optional seed for the spawn RNG (reproducible episodes).

        Returns:
            ``(obs_dict, info)`` where ``obs_dict`` is keyed ``cop_0.. + thief``
            with LOCAL Observations and ``info`` carries ``action_mask`` (per
            agent) and ``started_at`` (ISO-8601 timestamp). NO GlobalState.
        """
        rng = Random(seed)
        radius = view_radius(self._h, self._w, self._cfg)
        cop, thief = helpers.sample_positions(rng, self._h, self._w, radius, self._num_cops)
        self._state = GlobalState(
            cop_pos=cop,
            thief_pos=thief,
            barriers=frozenset(),
            barriers_used=0,
            step=0,
            h=self._h,
            w=self._w,
            terminal=False,
        )
        self._memory = helpers.fresh_memory(self._num_cops)
        helpers.update_memory(self._memory, self._state, self._cfg)
        obs = helpers.build_obs_dict(self._state, self._memory, self._num_cops, self._cfg)
        info = {
            "action_mask": helpers.build_mask_dict(self._state, self._num_cops, self._cfg),
            "started_at": dt.datetime.now(dt.UTC).isoformat(),
        }
        return obs, info

    def step(
        self, joint_a: dict[str, Action], eval_mode: bool = False
    ) -> tuple[dict[str, Observation], dict[str, float], bool, dict]:
        """Resolve one simultaneous joint action and return the 4-tuple.

        Args:
            joint_a: Per-agent action dict keyed ``cop_0.. + thief``.
            eval_mode: When True, potential-based shaping is gated OFF (the
                policy-invariance eval path); passed to ``RewardModel.compute``.

        Returns:
            ``(obs_dict, reward_dict, terminated, info)``. ``info`` carries
            ``action_mask`` / ``winner`` / ``capture`` / ``scores`` (scores only
            when terminated). NO GlobalState ever appears in the payload.

        Raises:
            RuntimeError: If called before :meth:`reset`, or after the episode
                has already terminated (call :meth:`reset` first).
        """
        if self._state is None:
            raise RuntimeError("step() called before reset()")
        if self._state.terminal:
            raise RuntimeError("step() on a terminated episode; call reset()")
        prev = self._state
        result = resolve_joint_action(prev, joint_a, self._cfg)
        self._state = result.next_state
        helpers.update_memory(self._memory, self._state, self._cfg)
        obs = helpers.build_obs_dict(self._state, self._memory, self._num_cops, self._cfg)
        reward_dict = self._reward.compute(prev, self._state, result.winner, eval_mode)
        terminated = self._state.terminal
        info = {
            "action_mask": helpers.build_mask_dict(self._state, self._num_cops, self._cfg),
            "winner": result.winner,
            "capture": result.capture,
            "scores": self._scorer.score(result.winner) if terminated else None,
        }
        return obs, reward_dict, terminated, info

    def state(self) -> GlobalState:
        """Return the full GlobalState (the sanctioned TRAIN-ONLY accessor).

        Returns:
            The current global state. Reachable here only — never via step/reset.

        Raises:
            RuntimeError: If called before the first :meth:`reset`.
        """
        if self._state is None:
            raise RuntimeError("state() called before reset()")
        return self._state
