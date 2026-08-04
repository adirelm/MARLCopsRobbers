"""Per-tick replay verification — positions AND the P5 masking the referee must withhold.

Split from :mod:`src.mcp.wire_replay` at the 150-LOC cap. Kept together because they answer
one question: does the logged payload match what an honest referee would have sent from the
replayed state? Position checks alone left a real hole — see :func:`_verify_masking`.
"""

from __future__ import annotations

from src.marl.env.observation import view_radius
from src.mcp._replay_log import ReplayMismatchError
from src.mcp.wire_referee import mask_payload

_ROLES = ("cop", "thief")


def verify_escalation_budget(cfg: dict, seeds_played: list[int], total_voids: int) -> int:
    """Every SPARE seed the match resolved must be PAID FOR by logged void re-hellos.

    P7 hands out a spare only after ``max_void_replays`` CONSECUTIVE voids, and each
    escalation consumes its own run of them, so an honest log always carries at least
    ``needed * spares_used`` voids in total. Without this the spare list was free money:
    a referee could shop the seeds for a favourable layout, play the spare for real and
    log no escalation at all. Spawn-matching cannot see that — the spare it played IS the
    spare it logged, so the spawns agree; only the missing voids give it away.

    Deliberately a MATCH-wide budget rather than a per-session one. Per-session looks
    tighter and is wrong: escalation re-seeds the whole pair k/k+3 and re-queues an
    already-played base game under the new seed, so THAT session shows a spare with a
    single re-hello of its own while the voids that bought it sit in its sibling. Billing
    it locally rejects a match no honest referee could have played differently.

    Returns:
        The number of distinct spare seeds the match resolved.

    Raises:
        ReplayMismatchError: When the log cannot afford the spares it claims to have used.
    """
    pairs = int(cfg["game"]["num_games"]) // 2
    spare_list = [int(s) for s in cfg["wire_match"]["seeds"][pairs:]]
    needed = int(cfg["wire_match"]["max_void_replays"])
    reached = [spare_list.index(int(s)) for s in seeds_played if int(s) in spare_list]
    # Billed by the HIGHEST spare index reached, not by how many distinct spares appear.
    # SeedSchedule consumes spares strictly IN ORDER, so the 2nd spare costs two escalations
    # and the 3rd costs three. Counting distinct spares priced every spare at one escalation,
    # which handed the referee a free CHOICE of layout for the price of the cheapest one —
    # seed shopping, exactly what this is here to stop. Measured: the frozen base seed is
    # unwinnable for the cop, spare #1 likewise, but spares #2/#3 are not, and all three cost
    # the same 3 voids.
    used = len(set(reached))
    owed = needed * (max(reached) + 1) if reached else 0
    if total_voids < owed:
        raise ReplayMismatchError(
            f"match reached SPARE #{max(reached) + 1} of the P7 order, which costs {owed} "
            f"consecutive voids (spares are consumed in order), but the log shows "
            f"{total_voids} void re-hello(s) in total — the escalation was never paid for"
        )
    # ...and a match-wide CEILING. Note this bound is GENEROUS on purpose: the real tightening
    # is per-session (verify_session_voids), and a tight match-wide formula turned out to
    # false-positive an honest match. Escalation RE-QUEUES an already-played base game, adding
    # a result-run the naive count misses, and `used` under-counts escalations when one pair
    # escalates twice — a legal 17-void SeedSchedule trace was rejected by the first version.
    spares_available = len(cfg["wire_match"]["seeds"]) - pairs
    ceiling = needed * spares_available + (needed - 1) * (len(seeds_played) + spares_available)
    if total_voids > ceiling:
        raise ReplayMismatchError(
            f"log shows {total_voids} void re-hello(s) across {len(seeds_played)} sub-game(s), "
            f"but with {spares_available} spare(s) available P7 cannot reach more than "
            f"{ceiling} — the extra voids describe re-rolls that never escalated"
        )
    return used


def verify_session_voids(cfg: dict, sid: str, voids: int, seed: int) -> None:
    """A sub-game still on its BASE seed may carry at most ``max_void_replays - 1`` voids.

    This is the bound that actually stops seed re-rolling, and the match-wide ceiling could
    not: with no spare resolved, a 6-sub-game match had 12 match-wide voids to spend, so 12
    re-rolls could be concentrated on the ONE sub-game that decided the match ("replay the
    layout until we win this one"). Verified: 12 concentrated re-rolls of our only cop win
    passed the match-wide ceiling.

    Sound because :meth:`SeedSchedule.record_void` FORCES escalation on the n-th consecutive
    void and zeroes its counter — so a session that ends on its base seed provably never
    reached n. Sessions on a spare are exempt here (their escalation is what the match-wide
    budget prices), which also keeps this from re-introducing the per-session FLOOR that
    wrongly rejected honest mirror halves.

    Raises:
        ReplayMismatchError: When a base-seed session logs a full escalation's worth of voids.
    """
    needed = int(cfg["wire_match"]["max_void_replays"])
    pairs = int(cfg["game"]["num_games"]) // 2
    spare_list = [int(s) for s in cfg["wire_match"]["seeds"][pairs:]]
    # Exempting spare-seed sessions entirely re-opened the very hole this function was written
    # to close: a sub-game the attacker first escalated then had NO per-session bound at all,
    # so 27 undisclosed replays of the deciding sub-game passed. The cap scales with WHICH
    # spare was reached — reaching spare #n costs n escalations, plus at most needed-1
    # non-escalating voids before the run that finally succeeded.
    index = spare_list.index(int(seed)) + 1 if int(seed) in spare_list else 0
    cap = needed * index + (needed - 1)
    if voids > cap:
        where = f"SPARE #{index}" if index else f"BASE seed {seed}"
        raise ReplayMismatchError(
            f"{sid}: {voids} void re-hello(s) on a sub-game that resolved to its {where}, but "
            f"P7 forces escalation every {needed} consecutive voids, so at most {cap} are "
            f"reachable — the rest describe re-rolling one layout"
        )


def verify_tick(cfg: dict, sid: str, tick: int, sess: dict, state) -> None:
    """Check the replayed PRE-MOVE state against the tick's logged request payloads.

    The log carries per-tick ground truth (each role's ``your_pos`` + ``barriers_left``),
    so a divergence ANYWHERE in a 25-move game is caught at the tick it happens — not
    only when it survives to the terminal summary (the silent-divergence hole).
    """
    truth = sess["states"].get(tick, {})
    env_pos = {"cop": tuple(state.cop_pos[0]), "thief": tuple(state.thief_pos)}
    left = int(cfg["game"]["max_barriers"]) - int(state.barriers_used)
    for role in _ROLES:
        logged = truth.get(role)
        if logged is None:
            raise ReplayMismatchError(f"{sid} tick {tick}: no logged request payload for {role}")
        if logged["your_pos"] != env_pos[role] or logged["barriers_left"] != left:
            raise ReplayMismatchError(
                f"{sid} tick {tick} {role}: logged {logged} != replayed "
                f"{{'your_pos': {env_pos[role]}, 'barriers_left': {left}}}"
            )
        _verify_masking(cfg, sid, tick, role, logged, state, env_pos)


def _verify_masking(cfg, sid, tick, role, logged, state, env_pos) -> None:  # noqa: PLR0913
    """Prove the referee WITHHELD what P5 says it must, not just that positions were legal.

    Rebuilds the payload with the referee's own ``mask_payload`` from the replayed state and
    compares the two masking fields. Checking positions alone leaves a real hole: a log in
    which the referee handed its own agent ``opponent_pos`` outside radius 2 — full board
    visibility — verified perfectly clean, which is exactly the cheat the shared log is
    supposed to make impossible.
    """
    # The masking fields are REQUIRED, not optional. An earlier version skipped the check
    # when both were absent, as back-compat for logs predating masking capture — but every
    # committed log carries them on 100% of payloads, so the hatch protected nothing while
    # handing a cheater a one-line bypass: strip the two keys on exactly the ticks you are
    # lying about and the check disappears for those ticks only. Verified exploitable before
    # removal: stripping them from all 262 request_move payloads made a leaked-position log
    # verify clean.
    if "opponent_pos" not in logged or "barriers" not in logged:
        raise ReplayMismatchError(
            f"{sid} tick {tick} {role}: request payload is missing the P5 masking fields "
            "(opponent_pos / barriers) — the log cannot be verified against §9's fairness rule"
        )
    other = "thief" if role == "cop" else "cop"
    expect = mask_payload(
        sid,
        tick,
        env_pos[role],
        env_pos[other],
        state.barriers,
        logged["barriers_left"],
        view_radius(state.h, state.w, cfg),
    )
    for field in ("opponent_pos", "barriers"):
        if logged.get(field) != expect[field]:
            raise ReplayMismatchError(
                f"{sid} tick {tick} {role}: P5 masking violated — logged {field}="
                f"{logged.get(field)!r} but radius-{view_radius(state.h, state.w, cfg)} "
                f"masking gives {expect[field]!r}"
            )
