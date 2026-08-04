"""P7 seed resolution + void-attempt verification (split from wire_replay.py at the LOC cap).

Two questions about the SAME thing — which seed a logged sub-game was really played on.
:func:`seeded_env` answers it for the attempt that SURVIVED; :func:`verify_void_attempts`
answers it for every attempt that was VOIDED along the way. Keeping them together is the
point: a void only counts as evidence if it was a real attempt at the same schedule.
"""

from __future__ import annotations

from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.mcp._replay_log import ReplayMismatchError
from src.mcp._replay_verify import verify_tick

_ROLES = ("cop", "thief")


def seeded_env(cfg: dict, sid: str, sess: dict, gid: int) -> tuple[CopsRobbersEnv, int]:
    """Return the env + seed for ``sid``, spawn-verified against BOTH logged hellos.

    PRIMARY source: the seed the referee RECORDED in the session's JSONL ``result`` event
    — exact, and still cross-checked against the logged spawns (the authoritative tamper
    guard). FALLBACK, for logs predating seed events only: s_k then the spares in order
    by spawn match — ambiguous in principle, because distinct seeds can collide on the
    (cop, thief) spawn pair (~1/396 per candidate on the 5x5 board), so a decoy spare
    earlier in the order could silently win; the recorded seed removes that risk.
    """
    seeds, grid = [int(s) for s in cfg["wire_match"]["seeds"]], int(cfg["game"]["grid_size"])
    pairs = int(cfg["game"]["num_games"]) // 2
    allowed = (seeds[(gid - 1) % pairs], *seeds[pairs:])  # P7: s_k or a spare — nothing else is legal
    recorded = sess.get("seed")
    if recorded is not None and recorded not in allowed:
        raise ReplayMismatchError(
            f"{sid}: recorded result seed {recorded} is neither s_k nor a spare in {seeds}"
        )
    # A spare must be PAID FOR by logged voids — but that bill is settled MATCH-wide in
    # replay_match (verify_escalation_budget), not here: escalation re-seeds the pair
    # k/k+3, so one half can legitimately show a spare with no voids of its own.
    for seed in allowed if recorded is None else (recorded,):
        env = CopsRobbersEnv(cfg, h=grid, w=grid, num_cops=1)
        env.reset(seed=seed)
        state = env.state()
        spawn_of = {"cop": tuple(state.cop_pos[0]), "thief": tuple(state.thief_pos)}
        if all(sess["spawns"].get(role) == spawn_of[role] for role in _ROLES):
            return env, seed
    raise ReplayMismatchError(
        f"{sid}: logged spawns {sess['spawns']} match neither s_k nor any spare seed in {seeds}"
        if recorded is None
        else f"{sid}: logged spawns {sess['spawns']} do not match the recorded seed {recorded}"
    )


def verify_void_attempts(cfg: dict, sid: str, sess: dict, gid: int) -> None:
    """Each RETAINED void attempt must be a real seeded opening, not three lines of text.

    Counting a void is not evidence of one. Every attempt kept by the parser is re-seeded
    here from the P7 candidates by SPAWN match, and its tick-0 payloads are then run through
    the same :func:`verify_tick` the surviving attempt gets. Forging a void therefore costs a
    genuine opening position plus correctly P5-masked payloads for both roles — instead of an
    off-board ``your_pos`` with no masking fields, which is what actually bought one before.
    """
    seeds, grid = [int(s) for s in cfg["wire_match"]["seeds"]], int(cfg["game"]["grid_size"])
    pairs = int(cfg["game"]["num_games"]) // 2
    candidates = (seeds[(gid - 1) % pairs], *seeds[pairs:])
    for index, attempt in enumerate(sess.get("void_attempts", [])):
        for seed in candidates:
            env = CopsRobbersEnv(cfg, h=grid, w=grid, num_cops=1)
            env.reset(seed=seed)
            state = env.state()
            spawn_of = {"cop": tuple(state.cop_pos[0]), "thief": tuple(state.thief_pos)}
            if all(attempt["spawns"].get(role) == spawn_of[role] for role in _ROLES):
                verify_tick(cfg, f"{sid} void#{index}", 0, attempt, state)
                break
        else:
            raise ReplayMismatchError(
                f"{sid}: void attempt #{index} has spawns {attempt['spawns']} matching no P7 "
                f"seed — a void must be a real attempt at the sub-game, not an asserted one"
            )
