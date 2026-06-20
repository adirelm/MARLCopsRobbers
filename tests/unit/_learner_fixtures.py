"""Shared fixtures for the CTDE learner unit tests (T4.2 / P4b Stage 3).

Builds tiny hand-shaped padded-episode BATCH dicts (the exact contract that
``CentralizedReplayBuffer.sample`` returns) decoupled from the real grid, so the
``QmixLearner`` smokes stay fast and deterministic. Lives in a helper module so
each learner test file stays well under the 150-LOC gate while sharing one
batch-builder (DRY). Dims mirror :func:`tests.unit._buffer_fixtures` (5x5x5
image, 6 scalars, state_dim 77) so a built batch matches the live obs encoder.
"""

from __future__ import annotations

import numpy as np

_C = 5
_W = 5
_SCALARS = 6
_STATE_DIM = 77
_N_ACTIONS = 5


def make_batch(  # noqa: PLR0913 — one kwarg per hand-shaped batch dim is intentional
    b: int,
    t: int,
    n: int,
    active: list[bool],
    reward: float = 1.0,
    seed: int = 0,
    filled: list[bool] | None = None,
) -> dict:
    """Build a tiny BATCH dict matching ``CentralizedReplayBuffer.sample``.

    Args:
        b: Batch size (episodes).
        t: Per-episode real-step horizon (the unroll spans ``t + 1`` frames).
        n: Agent-axis width.
        active: Episode-constant per-slot occupancy (length ``n``), broadcast
            over the batch.
        reward: Constant per-step team reward written into every cop slot.
        seed: RNG seed for the obs/state/action draws (determinism).
        filled: Optional per-step length mask (length ``t``); defaults to all
            real steps.

    Returns:
        A dict of numpy arrays with the buffer-sample shapes (obs/scalars/
        global_state on ``T+1``; actions/reward/done/filled/next_legal_mask on
        ``T``; active on ``N``).
    """
    rng = np.random.default_rng(seed)
    active_arr = np.array(active, dtype=bool)
    filled_arr = np.ones((b, t), dtype=bool) if filled is None else np.tile(filled, (b, 1)).astype(bool)
    done = np.zeros((b, t), dtype=bool)
    done[:, -1] = True
    return {
        "obs": rng.standard_normal((b, t + 1, n, _C, _W, _W)).astype(np.float32),
        "scalars": rng.standard_normal((b, t + 1, n, _SCALARS)).astype(np.float32),
        "global_state": rng.standard_normal((b, t + 1, _STATE_DIM)).astype(np.float32),
        "actions": rng.integers(0, _N_ACTIONS, size=(b, t, n)).astype(np.int64),
        "reward": np.full((b, t, n), reward, dtype=np.float32),
        "done": done,
        "filled": filled_arr,
        "next_legal_mask": np.ones((b, t, n, _N_ACTIONS), dtype=bool),
        "active": np.tile(active_arr, (b, 1)),
        "hidden_seed": np.zeros(b, dtype=np.int64),
    }
