"""Bind the §9.4 RECORDS to the match they claim to describe (kept out of wire_replay.py).

The replay re-runs every MOVE, which proves the sub-games were played as logged. It said
nothing about the fields that decide the OUTCOME — who played cop, how many sub-games there
were, whether the scores follow the §3.4 table. Those are derived quantities, so a records
file could disagree with its own log and still replay perfectly clean; an adversarial pass
flipped a 60-40 win to 45-55 that way, using the completely unmodified log.

Scope: this cannot prove the PARTNER's honesty. No signature binds a logged reply to its
author, so a self-consistent fabrication passes by construction — §9.3's independent
byte-compare is what covers that. This covers the unilateral drift.
"""

from __future__ import annotations

from src.mcp._replay_log import ReplayMismatchError


def agreed_group_names(cfg: dict, groups: dict | None) -> tuple[str, str]:
    """Return the two group names, bound to CONFIG and cross-checked against the body.

    The names come from ``wire_match.groups.*.name`` — the jointly frozen agreement — not
    from the body being audited. Reading them from the body made the alternation check
    self-referential: swapping the ``groups`` block AND every ``cop_group`` together is
    internally consistent, so §9.1 was satisfied against a mapping the body declared about
    itself. On a lost match that edit manufactures a win.

    ``groups`` is also REQUIRED. Treating a missing block as "a redacted copy carries no
    identity" was empirically wrong — redaction strips ``students_*`` and ``github_repo_*``
    and always keeps ``groups`` — and it handed away the whole role check for one deleted key.

    Raises:
        ReplayMismatchError: When the block is absent or disagrees with the agreement.
    """
    agreed = tuple(str(cfg["wire_match"]["groups"][k]["name"]) for k in ("group_1", "group_2"))
    if not groups:
        raise ReplayMismatchError(
            f"the §9.4 body has no 'groups' block — it is required, and §9.1 role alternation "
            f"cannot be checked without it (the agreement names are {agreed})"
        )
    claimed = tuple(str(groups.get(k, "")) for k in ("group_1", "group_2"))
    if claimed != agreed:
        raise ReplayMismatchError(
            f"the body's groups {claimed} != the jointly frozen agreement {agreed} — "
            f"group_1/group_2 decide who is cop first, so they are not the body's to declare"
        )
    return agreed


def verify_record_structure(
    cfg: dict, records: dict[int, dict], groups: dict | None, full_match: bool = True
) -> None:
    """Check the §9.1 id set and role alternation before any sub-game is replayed.

    Args:
        cfg: Loaded config (``game.num_games`` fixes the id set and the midpoint;
            ``wire_match.groups.*.name`` fixes who group_1 and group_2 are).
        records: ``{id: record}`` from the §9.4 body.
        groups: The body's ``groups`` block. Required — see :func:`_agreed_group_names`.
        full_match: Whether these records claim to BE a complete §9.1 match. True for every
            real artifact. Unit tests that deliberately replay a two-sub-game subset to probe
            seed resolution pass False. Deliberately an explicit argument rather than
            inferred from the id set: inferring "this looks partial, skip the check" would
            restore the exact hole being closed, since deleting a lost sub-game is what makes
            a match look partial.

    Raises:
        ReplayMismatchError: On a wrong id set or a role assignment §9.1 does not permit.
    """
    games = int(cfg["game"]["num_games"])
    if full_match and sorted(records) != list(range(1, games + 1)):
        raise ReplayMismatchError(
            f"sub-game ids {sorted(records)} != the §9.1 match of exactly {games} games "
            f"{list(range(1, games + 1))} — a match may not drop or add sub-games"
        )
    group_1, group_2 = agreed_group_names(cfg, groups)
    for gid, record in sorted(records.items()):
        # §9.1 alternation: group_1 is cop for the first half, group_2 for the second. The
        # margin is computed entirely from this assignment, so it is derived, never free.
        cop = group_1 if gid <= games // 2 else group_2
        thief = group_2 if cop == group_1 else group_1
        # Indexed, NEVER .get(field, expected). Defaulting an ABSENT field to the expected
        # value made the check vacuous: deleting cop_group/thief_group from every sub-game
        # passed, because absent compared equal to correct. Both are `required` in
        # bonus.schema.json, so nothing legitimate relies on a default.
        if record.get("cop_group") != cop or record.get("thief_group") != thief:
            raise ReplayMismatchError(
                f"sub-game {gid}: §9.1 role alternation violated — record says cop="
                f"{record.get('cop_group')!r}/thief={record.get('thief_group')!r}, "
                f"but §9.1 fixes cop={cop!r}/thief={thief!r}"
            )


def verify_record_scores(cfg: dict, gid: int, record: dict) -> None:
    """Require a record's scores to be the §3.4 Table-1 row for its own winner.

    The replay already compares scores against the env, so a wrong row is caught there too —
    but only as an opaque replay divergence. Checking the table directly names the actual
    problem, and it holds for the records ALONE, before a single move is replayed.

    Raises:
        ReplayMismatchError: When the scores are not the winner's Table-1 row.
    """
    scoring = cfg["game"]["scoring"]
    winner = record.get("winner")
    if winner not in ("cop", "thief"):
        raise ReplayMismatchError(f"sub-game {gid}: winner {winner!r} is neither 'cop' nor 'thief'")
    want = (
        {"cop": int(scoring["cop_win"]), "thief": int(scoring["thief_loss"])}
        if winner == "cop"
        else {"cop": int(scoring["cop_loss"]), "thief": int(scoring["thief_win"])}
    )
    got = {role: int(record["scores"][role]) for role in ("cop", "thief")}
    if got != want:
        raise ReplayMismatchError(
            f"sub-game {gid}: winner={winner} scores {got} != the §3.4 Table-1 row {want}"
        )


def verify_result_event(sid: str, sess: dict, record: dict) -> None:
    """Cross-check the referee's own logged ``result`` event against the record.

    The log's result event was read for its ``seed`` and nothing else, so a log could state
    one outcome while the records published another and the replay reported success. They
    are two claims about the same sub-game by the same referee; they must agree.

    Raises:
        ReplayMismatchError: When the logged result and the record disagree.
    """
    logged = sess.get("result")
    if logged is None:  # pre-seed-event logs carry no result events at all
        return
    for field in ("winner", "moves"):
        if field in logged and logged[field] != record[field]:
            raise ReplayMismatchError(
                f"{sid}: the log's result event says {field}={logged[field]!r} but the record "
                f"says {record[field]!r} — the referee cannot have logged both"
            )
    if "scores" in logged and {k: int(v) for k, v in logged["scores"].items()} != {
        k: int(v) for k, v in record["scores"].items()
    }:
        raise ReplayMismatchError(
            f"{sid}: the log's result event scores {logged['scores']} != the record's {record['scores']}"
        )
