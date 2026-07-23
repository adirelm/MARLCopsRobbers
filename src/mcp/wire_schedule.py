"""P7 seed schedule + the agreed void amendment — the consumer of ``VoidSubGame``.

Split out of :mod:`src.mcp.wire_client` for the 150-LOC cap (the client still
re-exports :class:`SeedSchedule`, so ``from src.mcp.wire_client import SeedSchedule``
keeps working). The client RAISES the §3.7 technical void; this schedule decides what
it means for the match: a void replays the SAME sub-game with the SAME seed, and only
after ``max_void_replays`` CONSECUTIVE voids of one sub-game does the next unused spare
seed replace s_k for the whole pair k/k+3 (replaying the base game if already played).
"""

from __future__ import annotations


class SeedSchedule:
    """P7 seed schedule + void amendment (agreed before implementation, brief P7/P8)."""

    def __init__(self, seeds: list, num_games: int, max_void_replays: int) -> None:
        """Freeze the agreed ORDERED list: first ``num_games/2`` = pair seeds, rest = spares."""
        if len(seeds) < num_games:
            raise ValueError(f"P7 requires >= {num_games} jointly agreed seeds, got {len(seeds)}")
        self._pairs = num_games // 2  # §9.1: ids 1..3 mirror 4..6 -> one seed per pair
        self._seeds = [int(s) for s in seeds]
        self._pair_seed = {k: self._seeds[k] for k in range(self._pairs)}
        self._spare = self._pairs
        self._pending = list(range(1, num_games + 1))
        self._done: set[int] = set()
        self._voids = 0
        self._max_voids = int(max_void_replays)

    def next_game(self) -> tuple[int, int] | None:
        """Return ``(game_id, seed)`` for the next sub-game, or None when the match is done."""
        if not self._pending:
            return None
        return self._pending[0], self._pair_seed[(self._pending[0] - 1) % self._pairs]

    def record_result(self, game_id: int) -> None:
        """Mark ``game_id`` validly completed; the consecutive-void counter resets."""
        self._pending.remove(game_id)
        self._done.add(game_id)
        self._voids = 0

    def record_void(self, game_id: int) -> list[int]:
        """Register one technical void; return the ids whose completed records became stale.

        Below the threshold: replay the SAME sub-game, SAME seed (empty list). At the
        threshold: the next spare seed replaces s_k for the pair k/k+3, and an
        already-played base game is re-queued FIRST (and returned as stale).

        Raises:
            RuntimeError: When escalation is required but every spare seed is used up.
        """
        self._voids += 1
        if self._voids < self._max_voids:
            return []
        if self._spare >= len(self._seeds):
            raise RuntimeError(f"P7 spare seeds exhausted while voiding sub-game {game_id}")
        pair = (game_id - 1) % self._pairs
        self._pair_seed[pair] = self._seeds[self._spare]
        self._spare += 1
        self._voids = 0
        base = pair + 1
        if base != game_id and base in self._done:
            self._done.discard(base)
            self._pending.insert(0, base)
            return [base]
        return []
