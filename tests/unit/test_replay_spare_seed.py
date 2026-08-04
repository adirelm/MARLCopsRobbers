"""P7 escalation must be PAID FOR in the log — a spare seed needs its voids.

The verifier accepted `s_k` or ANY spare with nothing tying a spare to the escalation that
earns it. Spawn-matching does not close that: a referee shopping for a favourable layout
plays the spare for real, so the spawns legitimately match it, and simply logs no voids.
"""

from __future__ import annotations

import pytest

from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.mcp._replay_log import ReplayMismatchError
from src.mcp.wire_replay import _seeded_env
from src.utils.config_loader import load_config


def _spawns(cfg, seed):
    grid = int(cfg["game"]["grid_size"])
    env = CopsRobbersEnv(cfg, h=grid, w=grid, num_cops=1)
    env.reset(seed=seed)
    st = env.state()
    return {"cop": tuple(st.cop_pos[0]), "thief": tuple(st.thief_pos)}


def _session(cfg, seed, voids):
    return {"spawns": _spawns(cfg, seed), "actions": {}, "states": {}, "seed": seed, "voids": voids}


def test_the_pair_seed_needs_no_voids() -> None:
    """Baseline: the legitimate s_k path must stay open."""
    cfg = load_config()
    s_k = int(cfg["wire_match"]["seeds"][0])
    _seeded_env(cfg, "sg-0", _session(cfg, s_k, 0), 1)


def test_a_spare_with_no_voids_is_rejected() -> None:
    """The seed-shopping case: real spawns for the spare, but no escalation logged."""
    cfg = load_config()
    pairs = int(cfg["game"]["num_games"]) // 2
    spare = int(cfg["wire_match"]["seeds"][pairs])
    with pytest.raises(ReplayMismatchError, match="requires 3 before a spare"):
        _seeded_env(cfg, "sg-0", _session(cfg, spare, 0), 1)


def test_a_spare_with_too_few_voids_is_rejected() -> None:
    """Two voids is not three — the bound is exact, not 'some escalation happened'."""
    cfg = load_config()
    pairs = int(cfg["game"]["num_games"]) // 2
    spare = int(cfg["wire_match"]["seeds"][pairs])
    with pytest.raises(ReplayMismatchError, match="void re-hello"):
        _seeded_env(cfg, "sg-0", _session(cfg, spare, 2), 1)


def test_a_spare_backed_by_the_full_escalation_is_accepted() -> None:
    """A genuine P7 escalation must still replay — the guard blocks fraud, not the rule."""
    cfg = load_config()
    pairs = int(cfg["game"]["num_games"]) // 2
    spare = int(cfg["wire_match"]["seeds"][pairs])
    needed = int(cfg["wire_match"]["max_void_replays"])
    _env, seed = _seeded_env(cfg, "sg-0", _session(cfg, spare, needed), 1)
    assert seed == spare


def test_the_parser_counts_void_re_hellos() -> None:
    """The count the guard relies on must come from the log, not be assumed."""
    from src.mcp._replay_log import parse_wire_log  # noqa: PLC0415

    cfg = load_config()
    from pathlib import Path  # noqa: PLC0415

    void_log = Path(cfg["wire_match"]["log_dir"]) / "wire_log_voidtest.jsonl"
    if not void_log.exists():
        pytest.skip("committed void-drill log absent")
    sessions = parse_wire_log(void_log)
    assert any(s.get("voids", 0) > 0 for s in sessions.values()), "no void counted in the void drill"
