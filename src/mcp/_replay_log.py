"""Wire-log parsing for the §9.3 replay (kept out of wire_replay.py for the 150-LOC cap).

Input: the referee's per-request JSONL log path.
Output: per-session dicts — spawns, per-tick action pairs, and per-tick GROUND-TRUTH
request payloads (``your_pos`` / ``barriers_left`` / the P5 masking fields per role) used to
verify EVERY replayed tick, not just the terminal summary.
Setup: none — pure parsing; imported by :mod:`src.mcp.wire_replay`.
"""

from __future__ import annotations

import json
from pathlib import Path

_ROLES = ("cop", "thief")


class ReplayMismatchError(RuntimeError):
    """The deterministic replay diverged from the log/records — the evidence is invalid."""


def _new_session() -> dict:
    """The per-session accumulator shape — ONE definition so the two call sites cannot drift.

    They had already drifted: the ``result`` branch omitted ``voids``, so any log whose
    result event reached the file before that session's requests raised KeyError on the
    first void re-hello instead of verifying the match.
    """
    return {"spawns": {}, "actions": {}, "states": {}, "voids": 0, "void_attempts": []}


def gid_of(session_id: str) -> int:
    """Return the 1-based sub-game id for a referee session id (``sg-0`` -> 1)."""
    return int(session_id.rsplit("-", 1)[1]) + 1


def parse_wire_log(path: str | Path) -> dict[str, dict]:
    """Parse the JSONL log into per-session spawns + per-tick actions AND ground truth.

    Returns ``{sid: {"spawns": {role: pos}, "actions": {tick: {role: str}},
    "states": {tick: {role: {"your_pos": pos, "barriers_left": int}}}}`` plus, when the
    referee logged a ``result`` event for the session, ``"seed"`` — the EXACT seed the
    completed run was played under (last result wins, matching the last-hello-wins run).
    A void re-hello supersedes the session's earlier run (spawns, actions AND states are
    wiped) — but an IDENTICAL re-hello (same role, same pos) with NO moves recorded
    since is a P8 TRANSPORT RETRY of the hello itself: an idempotent no-op that must
    NOT wipe the other role's spawn.
    """
    sessions: dict[str, dict] = {}
    pending: dict[str, tuple[str, int, str]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        direction, label = entry.get("direction"), entry.get("label")
        if direction == "request":
            payload = entry["payload"]
            sess = sessions.setdefault(payload["session_id"], _new_session())
            if entry["url"].endswith("/new_sub_game"):
                role, pos = payload["your_role"], tuple(payload["your_pos"])
                retry = sess["spawns"].get(role) == pos and not sess["actions"] and not sess["states"]
                if role in sess["spawns"] and not retry:  # re-hello: last run wins
                    # A void is counted ONLY when the superseded attempt actually STARTED —
                    # i.e. it logged at least one request_move. P7's void amendment is about
                    # a technical void DURING a sub-game (§3.7); a re-hello with nothing
                    # behind it is the P8 idempotent hello retry, not an escalation.
                    #
                    # The superseded attempt is KEPT, not just counted, because counting is
                    # not evidence. An earlier version of this comment claimed the logged
                    # request_move "must survive verify_tick against the seeded env" — it did
                    # not: the states were discarded on the very next line, so an off-board
                    # your_pos with no P5 masking fields at all minted a void for three lines
                    # of text. wire_replay now verifies each retained attempt against a real
                    # seeded env before it will spend it, which is what makes the price real.
                    #
                    # Under-counting is still the safe direction: it only makes a spare
                    # HARDER to justify.
                    if sess["states"]:
                        sess["voids"] += 1
                        sess["void_attempts"].append(
                            {"spawns": dict(sess["spawns"]), "states": dict(sess["states"])}
                        )
                    sess["spawns"], sess["actions"], sess["states"] = {}, {}, {}
                sess["spawns"][role] = pos
            else:
                tick, role = int(payload["tick"]), label.rsplit("-", 1)[1]
                sess["states"].setdefault(tick, {})[role] = {
                    "your_pos": tuple(payload["your_pos"]),
                    "barriers_left": int(payload["barriers_left"]),
                    # P5 masking fields: kept so the replay can prove the referee actually
                    # withheld what it was supposed to withhold, not merely that positions
                    # advanced legally. Without these a log in which the referee fed its own
                    # agent full board visibility replays perfectly clean.
                    # Copied only when PRESENT: the verifier now treats absence as a
                    # malformed log, so turning a missing key into None here would silently
                    # re-open the bypass it closes.
                    **{k: payload[k] for k in ("opponent_pos", "barriers") if k in payload},
                }
                pending[label] = (payload["session_id"], tick, role)
        elif direction == "response" and label in pending:
            sid, tick, role = pending.pop(label)
            if isinstance(reply := entry.get("response"), dict) and isinstance(reply.get("action"), str):
                sessions[sid]["actions"].setdefault(tick, {})[role] = reply["action"]
        elif direction == "result" and isinstance(sub := entry.get("sub_game"), dict):
            if "session_id" in sub and "seed" in sub:  # the referee's EXACT per-run seed (last wins)
                sess = sessions.setdefault(sub["session_id"], _new_session())
                sess["seed"] = int(sub["seed"])
                # Keep the OUTCOME too, not just the seed. It was read for the seed alone,
                # so a log could state one winner while the published records stated another
                # and the replay still reported success — two claims by the same referee
                # about the same sub-game, never held against each other.
                sess["result"] = {k: sub[k] for k in ("winner", "moves", "scores") if k in sub}
    return sessions


def ordered_actions(sid: str, sess: dict) -> list[dict[str, str]]:
    """Return the session's action pairs ordered by tick; raise on gaps/missing roles."""
    ticks = sorted(sess["actions"])
    if ticks != list(range(len(ticks))):
        raise ReplayMismatchError(f"{sid}: non-contiguous logged ticks {ticks}")
    pairs = [sess["actions"][t] for t in ticks]
    for tick, pair in enumerate(pairs):
        if set(pair) != set(_ROLES):
            raise ReplayMismatchError(f"{sid} tick {tick}: roles {sorted(pair)} != ['cop', 'thief']")
    return pairs


def select_log_and_records(cfg: dict) -> tuple[Path, Path]:
    """Return a log and records that describe the SAME match — never a mismatched pair.

    Choosing them independently is a real bug we shipped: the log default took the newest
    timestamped file while the records default fell back to the committed REHEARSAL records
    whenever the git-ignored real draft was absent. On any fresh clone that pairs the real
    §9 match log with rehearsal records, and the README's documented replay command dies
    with ReplayMismatchError before printing anything.

    Preference order: the git-ignored real draft (local only), then the TRACKED redacted
    §9.4 body, then the rehearsal records. The redacted copy is what makes a fresh clone
    work at all — it masks both student blocks and both repo URLs but keeps ``sub_games``
    intact, so a grader who clones the public repo replays the REAL graded match rather
    than being handed rehearsal records the config's seed list can no longer verify.

    Matching on (id, winner, moves) rather than on filenames means adding more logs later
    cannot silently re-pair them.

    Raises:
        SystemExit: When no committed log matches any available records file.
    """
    log_dir = Path(cfg["wire_match"]["log_dir"])
    logs = sorted(log_dir.glob("wire_log_[0-9]*.jsonl"), reverse=True)  # newest first
    candidates = [
        Path(cfg["wire_match"]["draft_report"]),  # local real draft (git-ignored: carries PII)
        Path(cfg["wire_match"]["redacted_records"]),  # TRACKED — what a fresh clone gets
        Path(cfg["wire_match"]["rehearsal"]["records"]),
    ]
    for records in candidates:
        if not records.exists():
            continue
        want = [
            (g["id"], g["winner"], g["moves"])
            for g in json.loads(records.read_text(encoding="utf-8"))["sub_games"]
        ]
        for log in logs:
            results = [
                json.loads(line)["sub_game"]
                for line in log.read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line).get("direction") == "result"
            ]
            if [(r["id"], r["winner"], r["moves"]) for r in results] == want:
                return log, records
    raise SystemExit(
        f"no log under {log_dir} matches any available records — pass --log and --records explicitly"
    )
