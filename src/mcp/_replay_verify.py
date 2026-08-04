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
    spares = {int(s) for s in cfg["wire_match"]["seeds"][pairs:]}
    needed = int(cfg["wire_match"]["max_void_replays"])
    used = len({int(s) for s in seeds_played if int(s) in spares})
    if total_voids < needed * used:
        raise ReplayMismatchError(
            f"match resolved {used} SPARE seed(s), which P7 grants only after {needed} "
            f"consecutive voids each, but the log shows {total_voids} void re-hello(s) "
            f"in total — the escalation was never paid for"
        )
    return used


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
