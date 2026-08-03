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
    if logged.get("opponent_pos") is None and logged.get("barriers") is None:
        return  # log predates masking capture; positions are still verified above
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
