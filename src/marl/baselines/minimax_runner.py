"""Minimax-Q self-play training loop on the tabular pursuit (P-bonus, L11 §5).

Drives :class:`~src.marl.baselines.tabular_pursuit.TabularPursuit` for ``minimax_q.episodes``
episodes: each tick the cop samples its **maximin** mixed strategy and the thief its **minimax**
mixed strategy (both ε-explored), and the shared :class:`~src.marl.baselines.minimax_q.MinimaxQ`
Q-table is updated (eq 2.1). Returns a per-window learning history (rolling capture-rate + the
game value at a fixed reference start state) for the F7 figure + the README §7.2 comparison.
"""

from __future__ import annotations

import numpy as np

from src.marl.baselines.minimax_q import MinimaxQ
from src.marl.baselines.tabular_pursuit import TabularPursuit


def _sample(strategy: np.ndarray, rng: np.random.Generator, epsilon: float) -> int:
    """Sample an action: uniform-random with prob ``epsilon``, else from the mixed strategy."""
    n = len(strategy)
    if rng.random() < epsilon:
        return int(rng.integers(n))
    p = np.clip(strategy, 0.0, None)
    total = p.sum()
    return int(rng.integers(n)) if total <= 0 else int(rng.choice(n, p=p / total))


def train_minimax_q(cfg: dict, seed: int) -> list[dict]:
    """Train tabular Minimax-Q via self-play; return the per-window learning history.

    Args:
        cfg: The loaded config (reads the ``minimax_q.*`` block).
        seed: Master seed (reproducible episode spawns + strategy sampling).

    Returns:
        ``[{"episode", "capture_rate", "ref_value"}, ...]`` — one row per ``minimax_q.window``
        episodes: the rolling cop capture-rate and the game value at the first episode's start.
    """
    mq = cfg["minimax_q"]
    env = TabularPursuit(cfg)
    learner = MinimaxQ(env.n_actions, env.n_actions, mq["alpha_start"], mq["gamma"])
    rng = np.random.default_rng(seed)
    episodes, window = int(mq["episodes"]), int(mq["window"])
    eps, e_end, e_decay = float(mq["eps_start"]), float(mq["eps_end"]), float(mq["eps_decay"])
    a_end, a_decay = float(mq["alpha_end"]), float(mq["alpha_decay"])
    captures: list[float] = []
    history: list[dict] = []
    ref: tuple[int, int] | None = None
    for ep in range(episodes):
        state = env.reset(int(rng.integers(1, 2**31)))
        ref = state if ref is None else ref
        captured = False
        for _ in range(env.max_moves):
            a_cop = _sample(learner.cop_strategy(state), rng, eps)
            a_thief = _sample(learner.thief_strategy(state), rng, eps)
            nxt, reward, done = env.step(a_cop, a_thief)
            learner.update(state, a_cop, a_thief, reward, nxt, done)
            state = nxt
            if done:
                captured = reward > 0
                break
        captures.append(1.0 if captured else 0.0)
        learner.anneal(a_end, a_decay)  # decay the learning rate (Littman 1994 convergence)
        eps = max(e_end, eps * e_decay)  # GLIE: anneal exploration toward greedy
        if (ep + 1) % window == 0:
            history.append(
                {
                    "episode": ep + 1,
                    "capture_rate": float(np.mean(captures[-window:])),
                    "ref_value": learner.value(ref),
                }
            )
    return history
