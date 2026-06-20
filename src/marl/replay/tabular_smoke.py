"""THROWAWAY P3 smoke — replaced by the GRU CTDE learner in P4 (delete then).

A tiny tabular Q-learner over the EXACT discrete 2x2 state space, used only to
prove the P3 pipeline end-to-end: collect (cop=learner epsilon-greedy vs
thief=heuristic) on the 2x2 stage, push each whole episode through the
:class:`~src.marl.replay.episode_buffer.CentralizedReplayBuffer`, train tabularly,
then assert the greedy (eval) cop captures the heuristic thief within the minimum
Manhattan step budget from every distinct 2x2 start, for ALL seeds, NaN-free.
This is a SANITY smoke, never a graded benchmark; all hyperparams come from the
``p3_smoke`` config block (CLAUDE.md §4 no-hardcode).
"""

from __future__ import annotations

import math

from src.marl.env.actions import Action
from src.marl.env.types import GlobalState
from src.marl.replay import _smoke_solve as solve


def state_key(state: GlobalState) -> tuple:
    """Return the exact discrete key (cop_pos, thief_pos, barriers, step).

    Barriers are sorted into a tuple so the key is deterministic and hashable;
    every distinct discrete state maps to a distinct key.

    Args:
        state: The global state to key.

    Returns:
        A hashable ``(cop_pos, thief_pos, sorted_barriers, step)`` tuple.
    """
    return (state.cop_pos, state.thief_pos, tuple(sorted(state.barriers)), state.step)


class TabularQLearner:
    """Throwaway dict-backed Q-learner; ``Q[(s_key, agent_key)][action] = value``."""

    def __init__(self, cfg: dict) -> None:
        """Bind alpha/epsilon from ``p3_smoke`` and gamma from ``algo.gamma``.

        Args:
            cfg: The loaded project config (config/config.yaml).
        """
        smoke = cfg["p3_smoke"]
        self.alpha = float(smoke["alpha"])
        self.epsilon = float(smoke["epsilon"])
        self.gamma = float(cfg["algo"]["gamma"])
        self._q: dict[tuple, dict[int, float]] = {}

    def q_value(self, s_key: tuple, agent_key: str, action: int) -> float:
        """Return ``Q(s, agent, action)`` (0.0 when unseen)."""
        return self._q.get((s_key, agent_key), {}).get(int(action), 0.0)

    def set_q_value(self, s_key: tuple, agent_key: str, action: int, value: float) -> None:
        """Set ``Q(s, agent, action)`` directly (test/seed helper)."""
        self._q.setdefault((s_key, agent_key), {})[int(action)] = float(value)

    def update(  # noqa: PLR0913 - one arg per Q-learning operand (pinned signature)
        self,
        s_key: tuple,
        agent_key: str,
        action: int,
        reward: float,
        next_s_key: tuple,
        next_legal,
        done: bool = False,
    ) -> None:
        """Apply ``Q += alpha * (target - Q)`` in place (Q-learning TD step).

        The bootstrap target is ``r + gamma * max_legal Q'`` on a non-terminal
        transition, and ``r`` ALONE at a terminal transition (``done=True``) — a
        terminal state has no successor, so bootstrapping ``gamma * max Q'`` there
        leaks a spurious value and is the canonical tabular-Q correctness bug.

        Args:
            s_key: Current discrete state key.
            agent_key: The acting agent (e.g. ``"cop_0"``).
            action: The taken action index.
            reward: The observed scalar reward.
            next_s_key: The next discrete state key.
            next_legal: Boolean legality mask at the next state (masks the max).
            done: Whether the transition is terminal (suppresses the bootstrap).
        """
        best_next = 0.0 if done else self._max_legal_q(next_s_key, agent_key, next_legal)
        target = reward + self.gamma * best_next
        current = self.q_value(s_key, agent_key, action)
        self.set_q_value(s_key, agent_key, action, current + self.alpha * (target - current))

    def _max_legal_q(self, s_key: tuple, agent_key: str, legal) -> float:
        """Return ``max`` Q over legal actions at ``s_key`` (0.0 if none legal)."""
        values = [self.q_value(s_key, agent_key, i) for i, ok in enumerate(legal) if ok]
        return max(values) if values else 0.0

    def greedy_action(self, s_key: tuple, agent_key: str, legal_mask) -> Action:
        """Return the highest-Q legal action (lowest Action index breaks ties).

        Args:
            s_key: The discrete state key.
            agent_key: The acting agent key.
            legal_mask: Boolean legality mask over the action set.

        Returns:
            The chosen legal :class:`Action`.

        Raises:
            ValueError: If no action is legal.
        """
        best_idx, best_val = None, -math.inf
        for i, ok in enumerate(legal_mask):
            if not ok:
                continue
            val = self.q_value(s_key, agent_key, i)
            if val > best_val:
                best_idx, best_val = i, val
        if best_idx is None:
            raise ValueError("greedy_action called with an all-False legal mask")
        return Action(best_idx)


def q_is_nan_free(learner: TabularQLearner) -> bool:
    """Return whether every stored Q-value is finite (no NaN/inf leaked in)."""
    return all(math.isfinite(v) for table in learner._q.values() for v in table.values())


def run_smoke(cfg: dict, seeds) -> dict:
    """Collect -> tabular-train -> assert 2x2 optimal capture over ``seeds``.

    For each seed a fresh :class:`TabularQLearner` is trained with exploring
    starts (cop=learner epsilon-greedy vs thief=heuristic) on the 2x2 stage, with
    every whole episode pushed through the centralized replay buffer. The greedy
    (eval) cop is then scored from every distinct 2x2 start: it must CAPTURE the
    heuristic thief within the game-theoretic optimal step budget (exactly optimal
    on the interceptable diagonal starts), with NaN-free Q-values, for ALL seeds.

    Args:
        cfg: The loaded project config.
        seeds: An iterable of training seeds (e.g. ``training.seeds[:2]``).

    Returns:
        A metrics dict with ``optimal`` / ``all_captured`` / ``nan_free`` flags and
        the evaluated ``seeds`` list.
    """
    seed_list = list(seeds)
    nan_free, all_captured = True, True
    for seed in seed_list:
        learner = solve.train_seed(cfg, seed, TabularQLearner(cfg))
        nan_free = nan_free and q_is_nan_free(learner)
        all_captured = all_captured and solve.eval_optimal(cfg, learner)
    optimal = all_captured and nan_free
    return {
        "optimal": optimal,
        "all_captured": all_captured,
        "nan_free": nan_free,
        "seeds": seed_list,
    }
