"""Zero-sum tabular pursuit adapter over the real env (P-bonus, L11 §5).

A thin, FULLY-OBSERVED wrapper around :class:`~src.marl.env.cops_robbers_env.CopsRobbersEnv`
at ``grid x grid`` with 1 cop, exposing the minimal interface a tabular Minimax-Q needs:
a flat ``(cop_cell, thief_cell)`` state key, the four directional moves, and a **zero-sum**
payoff derived from the terminal ``winner`` (cop capture = ``+capture_reward``; timeout/escape
= ``escape_reward``). It REUSES the real transition/capture rules (DRY) rather than re-deriving
them — only the observation/shaping machinery is bypassed, which the tabular game does not use.
"""

from __future__ import annotations

from src.marl.env.actions import Action
from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.marl.env.types import Pos


class TabularPursuit:
    """A zero-sum, full-observability tabular adapter over ``CopsRobbersEnv`` (1 cop vs thief)."""

    n_actions = 4  # the four directional moves UP/DOWN/LEFT/RIGHT (no barrier / stay)

    def __init__(self, cfg: dict, grid: int | None = None, max_moves: int | None = None) -> None:
        """Bind a ``grid x grid`` 1-cop env + the zero-sum payoffs from ``cfg['minimax_q']``.

        Args:
            cfg: The loaded project config (reads the ``minimax_q.*`` block).
            grid: Override the board side (default ``minimax_q.grid``).
            max_moves: Override the per-episode horizon (default ``minimax_q.max_moves``).
        """
        mq = cfg["minimax_q"]
        self.grid = int(grid if grid is not None else mq["grid"])
        self.max_moves = int(max_moves if max_moves is not None else mq["max_moves"])
        self._capture = float(mq["capture_reward"])
        self._escape = float(mq["escape_reward"])
        self._env = CopsRobbersEnv(cfg, h=self.grid, w=self.grid, num_cops=1)
        self._steps = 0

    def reset(self, seed: int) -> tuple[int, int]:
        """Seed a spawn; return the ``(cop_cell, thief_cell)`` tabular state key."""
        self._env.reset(seed)
        self._steps = 0
        return self._key()

    def step(self, a_cop: int, a_thief: int) -> tuple[tuple[int, int], float, bool]:
        """Apply one joint move; return ``(state_key, zero_sum_reward, terminated)``.

        ``a_cop`` / ``a_thief`` are directional move indices 0..3. The reward is the
        ROW (cop) zero-sum payoff: nonzero only on termination.
        """
        joint = {"cop_0": Action(a_cop), "thief": Action(a_thief)}
        _obs, _reward, terminated, info = self._env.step(joint)
        self._steps += 1
        winner = info["winner"]
        if not terminated and self._steps >= self.max_moves:
            terminated, winner = True, "thief"  # horizon cap == thief escape
        reward = (self._capture if winner == "cop" else self._escape) if terminated else 0.0
        return self._key(), reward, terminated

    def _key(self) -> tuple[int, int]:
        """Encode the global (cop, thief) positions into a flat ``(cell, cell)`` key."""
        st = self._env.state()
        return self._cell(st.cop_pos[0]), self._cell(st.thief_pos)

    def _cell(self, pos: Pos) -> int:
        """Flatten a ``(row, col)`` position to a single cell index in ``[0, grid*grid)``."""
        return pos[0] * self.grid + pos[1]
