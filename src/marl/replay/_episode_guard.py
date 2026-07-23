"""Step-semantics guard for episodes entering the replay ring (codex W2 M5).

:class:`~src.marl.replay.episode_buffer.CentralizedReplayBuffer` validates the
agent-axis WIDTHS itself; this helper validates the STEP semantics the recurrent
BPTT unroll assumes: ``filled`` must be a NON-EMPTY PREFIX of the horizon (a hole
would make the GRU carry hidden state across a gap of fake zero steps) and
``done`` may be True only at the LAST real step (no real steps after a terminal,
no terminal flag inside the zero pad). The in-repo producer
(:mod:`src.services.episode_pad`) always satisfies this — the guard exists so a
malformed episode fails LOUDLY at the ingestion boundary instead of being
silently trained on.
"""

from __future__ import annotations

import numpy as np


def validate_step_semantics(episode: dict, t_max: int) -> None:
    """Fail loud on a semantically invalid episode (only the stored ``[:t_max]`` slice).

    Args:
        episode: The episode dict handed to ``add_episode`` (reads ``filled``/``done``).
        t_max: The buffer's padded horizon (fields beyond it are never stored).

    Raises:
        ValueError: If ``filled`` is empty/all-False or not a prefix (a hole),
            or if ``done`` is True anywhere but the last filled step.
    """
    filled = np.asarray(episode["filled"], dtype=bool)[:t_max]
    done = np.asarray(episode["done"], dtype=bool)[: filled.shape[0]]
    real_steps = int(filled.sum())
    if real_steps == 0 or not filled[:real_steps].all():
        raise ValueError(
            "episode['filled'] must be a NON-EMPTY PREFIX of the horizon "
            f"(got {real_steps} real steps over {filled.tolist()})"
        )
    if done[: real_steps - 1].any() or done[real_steps:].any():
        raise ValueError(
            "episode['done'] may be True only at the LAST real (filled) step "
            f"(filled prefix {real_steps}, done {done.tolist()})"
        )
