"""Head-to-head matchup evaluation — the exploitability probe behind ANALYSIS §12.

Input: loaded config, two policy factories (callables returning fresh ``act()``+``reset()``
policies), a consecutive seed block, and an optional thief exploration epsilon.
Output: an arm summary dict — ``captures`` / ``games`` / ``moves`` / ``cop_actions``.
Setup: pure service, no I/O; the SDK seam (``MarlSDK.run_matchup_eval``) wires the serving
nets and the named opponent arms, reading ``matchup_eval`` config for the block shape.
DI-testable: the factories can be scripted test doubles on a tiny board.

The greedy (eps=0) arms here play EXACTLY the shipped match policy: the harness was
verified bit-identical to the MCP referee path (300/300 ticks on the match seeds 7-12).
"""

from __future__ import annotations

import collections
from collections.abc import Callable

import numpy as np

from src.marl.env.actions import Action
from src.marl.env.cops_robbers_env import CopsRobbersEnv


class UniformRandomPolicy:
    """Uniform choice over legal actions — the non-adaptive control floor."""

    def reset(self) -> None:
        """No episode state to clear."""

    def act(self, _obs_list: list, legal_masks: list, _epsilon: float, rng, state: object = None) -> list:
        """Sample one legal action index uniformly."""
        legal = [i for i, ok in enumerate(legal_masks[0]) if ok]
        return [int(rng.choice(legal))]


def play_matchup(  # noqa: PLR0913 — cfg + 2 policies + seed + stage + eps are distinct
    cfg: dict,
    cop_policy: object,
    thief_policy: object,
    seed: int,
    stage: tuple[int, int, int] = (5, 5, 1),
    thief_eps: float = 0.0,
) -> tuple[bool, collections.Counter, int]:
    """One sub-game; returns ``(captured, cop-action Counter, moves)``.

    The cop always plays greedy (eps=0, the serving convention); ``thief_eps`` injects
    exploration noise into the THIEF only — the perturbation probe for the
    deterministic-lock finding (ANALYSIS §12). The cop's ``act`` also receives the
    sanctioned train-only ``env.state()`` via ``state=``: net/scripted local policies
    ignore it, while privileged scripted opponents (the foreign-cop battery's
    ``OracleBfsCop``) read it. The THIEF never receives state (its CTDE seal holds).
    """
    h, w, num_cops = stage
    env = CopsRobbersEnv(cfg, h=h, w=w, num_cops=num_cops)
    obs, info = env.reset(seed=seed)
    cop_policy.reset()
    thief_policy.reset()
    rng = np.random.default_rng(seed)
    tally: collections.Counter = collections.Counter()
    terminated, moves = False, 0
    while not terminated:
        masks = info["action_mask"]
        cop_masks = [list(map(bool, masks["cop_0"]))]
        cop_a = cop_policy.act([obs["cop_0"]], cop_masks, 0.0, rng, state=env.state())[0]
        thief_a = thief_policy.act([obs["thief"]], [list(map(bool, masks["thief"]))], thief_eps, rng)[0]
        tally[Action(int(cop_a)).name] += 1
        obs, _r, terminated, info = env.step({"cop_0": Action(int(cop_a)), "thief": Action(int(thief_a))})
        moves += 1
    return bool(info.get("capture")), tally, moves


def evaluate_matchup(  # noqa: PLR0913 — cfg + 2 factories + block shape + eps are distinct
    cfg: dict,
    cop_factory: Callable[[], object],
    thief_factory: Callable[[], object],
    n_games: int,
    base_seed: int,
    thief_eps: float = 0.0,
    stage: tuple[int, int, int] = (5, 5, 1),
) -> dict:
    """Aggregate ``n_games`` sub-games on consecutive seeds ``base_seed..+n_games-1``.

    One cop policy instance is reused (reset per game, matching serving semantics);
    the thief factory is called per game so scripted opponents start fresh.
    """
    captures, total_moves = 0, 0
    tally: collections.Counter = collections.Counter()
    cop = cop_factory()
    for i in range(n_games):
        captured, actions, moves = play_matchup(cfg, cop, thief_factory(), base_seed + i, stage, thief_eps)
        captures += captured
        tally += actions
        total_moves += moves
    return {
        "captures": captures,
        "games": n_games,
        "moves": total_moves,
        "cop_actions": dict(tally),
        "thief_eps": thief_eps,
        "base_seed": base_seed,
    }


def run_arm(sdk: object, cfg: dict, thief_kind: str, thief_eps: float = 0.0) -> dict:
    """Run one named arm of the ANALYSIS §12 table with the SERVING cop.

    ``thief_kind``: ``net`` (our self-play thief), ``flee`` (the scripted baseline:
    uniform-random over legal moves before first opponent sighting, deterministic
    greedy distance-maximising after; ignores epsilon), or ``random`` (uniform floor).

    Raises:
        ValueError: On an unknown ``thief_kind``.
    """
    from src.services.bonus_policies import ObsFleePolicy  # noqa: PLC0415 — lazy: keep import light

    block = cfg["matchup_eval"]
    cop_net = sdk.serving_net("cop")
    if thief_kind == "net":  # load the bundle ONCE, not per game
        thief_net = sdk.serving_net("thief")
        factories = {"net": lambda: sdk.build_policy("thief", thief_net, 1)}
    else:
        factories = {"flee": lambda: ObsFleePolicy(cfg), "random": UniformRandomPolicy}
    if thief_kind not in factories:
        raise ValueError(f"run_arm: thief_kind must be one of ['flee', 'net', 'random'], got {thief_kind!r}")
    return evaluate_matchup(
        cfg,
        lambda: sdk.build_policy("cop", cop_net, 1),
        factories[thief_kind],
        int(block["n_games"]),
        int(block["base_seed"]),
        thief_eps,
    )
