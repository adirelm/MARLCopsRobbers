"""Graduated Sanity-Check curriculum ladder for the CopsRobbersEnv (T2.4).

`Curriculum` walks the ``env.curriculum.stages`` ladder (2x2 -> 3x3 -> 4x4 ->
5x5) with the parallel ``env.curriculum.num_cops_by_stage`` cop counts (BRIEF
§5.1 Table 2; FR-ENV-8). ``current`` reports the active stage's
``(h, w, num_cops)``; ``maybe_promote`` advances the stage index when the
cop capture-rate clears ``env.curriculum.promotion_threshold`` (clamped at the
final stage); ``make_env`` builds a fresh :class:`CopsRobbersEnv` for the active
stage (a cop-count change between stages REQUIRES a new env — the GlobalState
``cop_pos`` tuple length differs per stage). All bounds come from config —
nothing is hardcoded (CLAUDE.md §4).
"""

from __future__ import annotations

from src.marl.env.cops_robbers_env import CopsRobbersEnv


class Curriculum:
    """The success-rate-gated curriculum stage scheduler (FR-ENV-8)."""

    def __init__(self, cfg: dict) -> None:
        """Bind the ladder from ``env.curriculum`` and start at stage 0.

        Args:
            cfg: The loaded project config (config/config.yaml). Reads
                ``env.curriculum.{stages, num_cops_by_stage, promotion_threshold,
                promotion_window}``. Bound as ``self._cfg`` — the SINGLE config
                source used by both :meth:`current` and :meth:`make_env`.

        Raises:
            ValueError: If ``stages`` and ``num_cops_by_stage`` differ in length
                (the per-stage cop counts must be a parallel list).
        """
        self._cfg = cfg
        curriculum = cfg["env"]["curriculum"]
        self._stages: list[tuple[int, int]] = [tuple(s) for s in curriculum["stages"]]
        self._num_cops_by_stage: list[int] = list(curriculum["num_cops_by_stage"])
        if len(self._stages) != len(self._num_cops_by_stage):
            raise ValueError(
                "env.curriculum.stages and num_cops_by_stage must be parallel "
                f"lists: got {len(self._stages)} stages vs "
                f"{len(self._num_cops_by_stage)} cop counts"
            )
        self._threshold: float = curriculum["promotion_threshold"]
        self._window: int = curriculum["promotion_window"]
        self._idx = 0

    @property
    def promotion_window(self) -> int:
        """Return the episode window over which the capture-rate is gated."""
        return self._window

    def current(self) -> tuple[int, int, int]:
        """Return ``(h, w, num_cops)`` for the active curriculum stage.

        Returns:
            The active stage's board rows, columns, and cop count.
        """
        h, w = self._stages[self._idx]
        return h, w, self._num_cops_by_stage[self._idx]

    def maybe_promote(self, capture_rate: float) -> bool:
        """Advance one stage when ``capture_rate`` clears the threshold.

        Promotion is clamped at the final stage: once there, no further advance
        is possible and the call returns ``False`` even at a perfect rate.

        Args:
            capture_rate: The cop capture-rate over ``promotion_window`` episodes.

        Returns:
            ``True`` iff this call advanced the stage index, else ``False``.
        """
        if capture_rate < self._threshold:
            return False
        if self._idx >= len(self._stages) - 1:
            return False
        self._idx += 1
        return True

    def make_env(self, cfg: dict | None = None) -> CopsRobbersEnv:
        """Build a fresh CopsRobbersEnv for the active stage.

        Uses the SINGLE config bound at construction (``self._cfg``) for both the
        stage dims (via :meth:`current`) AND the game rules passed to the env —
        no split-brain across two configs. A new env is required on every
        cop-count change (the GlobalState ``cop_pos`` tuple length differs per
        stage); non-square ``h != w`` stages are supported by the egocentric obs
        pad (FR-ENV-7).

        Args:
            cfg: Deprecated/optional override of the env config. Defaults to the
                bound ``self._cfg`` (the single source) when ``None``.

        Returns:
            A :class:`CopsRobbersEnv` bound to the active stage's
            ``(h, w, num_cops)``.
        """
        h, w, num_cops = self.current()
        env_cfg = self._cfg if cfg is None else cfg
        return CopsRobbersEnv(env_cfg, h=h, w=w, num_cops=num_cops)
