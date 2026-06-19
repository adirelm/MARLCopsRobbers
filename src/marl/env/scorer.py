"""Report-only §3.4 scoreboard — NEVER the RL training signal (T1.4).

This module is a DISTINCT seam from the reward path (see reward.py, which must
never import this file). It maps a sub-game winner to the BRIEF Table-1 points
(cop_win=20, thief_win=10, cop_loss=5, thief_loss=5), used solely for the
human-readable scoreboard and the Gmail report — never for training.
"""

from __future__ import annotations


class Scorer:
    """Map a sub-game winner to the §3.4 Table-1 scoreboard points.

    Reads ``game.scoring.{cop_win,thief_win,cop_loss,thief_loss}`` from config.
    The output is report-only and is never consumed by the RL reward.
    """

    def __init__(self, cfg: dict) -> None:
        """Bind the §3.4 scoring table from the validated config.

        Args:
            cfg: The loaded project config (see config/config.yaml).
        """
        scoring = cfg["game"]["scoring"]
        self._cop_win = scoring["cop_win"]
        self._thief_win = scoring["thief_win"]
        self._cop_loss = scoring["cop_loss"]
        self._thief_loss = scoring["thief_loss"]

    def score(self, winner: str) -> dict[str, int]:
        """Return the {cop, thief} point totals for ``winner``.

        Args:
            winner: ``"cop"`` (capture) or ``"thief"`` (timeout/evade).

        Returns:
            ``{"cop": cop_win, "thief": thief_loss}`` on capture, or
            ``{"cop": cop_loss, "thief": thief_win}`` on a thief win.

        Raises:
            ValueError: If ``winner`` is not ``"cop"`` or ``"thief"``.
        """
        if winner == "cop":
            return {"cop": self._cop_win, "thief": self._thief_loss}
        if winner == "thief":
            return {"cop": self._cop_loss, "thief": self._thief_win}
        raise ValueError(f"unknown winner {winner!r} (expected 'cop' or 'thief')")
