"""MarlSDK — the single business-logic entry point (CLAUDE.md §3, BRIEF §6).

A thin FACADE: every public method routes through ``src.marl.*`` building blocks
(env / replay / tabular smoke / render seam) with no business logic duplicated
here. The P3 surface wires the 2x2 pipeline: build an env, collect a padded
episode for the centralized replay buffer, run the throwaway 2x2 smoke (collect
-> tabular-train -> assert optimal), and write a minimal headless sub-game JSON.
Private wiring lives in :mod:`src.sdk._helpers` so this file stays <=150 LOC.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.marl.replay import tabular_smoke
from src.sdk import _helpers

Policy = Callable[..., object]


class MarlSDK:
    """The single sanctioned entry point for MARL Cops & Robbers business logic."""

    def __init__(self, cfg: dict) -> None:
        """Bind the SDK to a loaded config.

        Args:
            cfg: The loaded project config (config/config.yaml via config_loader).
        """
        self._cfg = cfg

    def build_env(
        self, h: int | None = None, w: int | None = None, num_cops: int | None = None
    ) -> CopsRobbersEnv:
        """Construct a P2 :class:`CopsRobbersEnv` (config defaults when args None).

        Args:
            h: Board rows; defaults to ``game.grid_size``.
            w: Board columns; defaults to ``game.grid_size``.
            num_cops: Cop count; defaults to ``env.num_cops``.

        Returns:
            A ready-to-reset env.
        """
        return _helpers.make_env(self._cfg, h, w, num_cops)

    def collect_episode(
        self, env: CopsRobbersEnv, cop_policy: Policy, thief_policy: Policy, seed: int
    ) -> dict:
        """Collect ONE padded episode (buffer schema) under the given policies.

        Args:
            env: A reset-able single-cop env.
            cop_policy: ``policy(state, mask, cfg) -> Action`` for the cop.
            thief_policy: ``policy(state, mask, cfg) -> Action`` for the thief.
            seed: The spawn seed (reproducible episode).

        Returns:
            The padded episode dict matching the ``CentralizedReplayBuffer`` schema.
        """
        return _helpers.run_episode(env, cop_policy, thief_policy, seed, self._cfg)

    def run_p3_smoke(self, seeds, out_dir: str | Path | None = None) -> dict:
        """Run the 2x2 collect->train->export smoke and write the P3 artifacts.

        Delegates the optimal-capture gate to
        :func:`src.marl.replay.tabular_smoke.run_smoke`, then writes a minimal
        sub-game JSON whose SHAPE conforms to ``docs/schema/subgame.schema.json``
        and attaches a headless god-view ``render_state`` dict (no GlobalState).
        The write performs NO runtime schema validation (no validator is added at
        P3); the integration test asserts the emitted record validates.

        Args:
            seeds: The training seeds to evaluate (e.g. ``training.seeds[:2]``).
            out_dir: Optional sub-game output dir; defaults to ``paths.subgames_dir``.

        Returns:
            The ``run_smoke`` metrics dict augmented with ``subgame_path`` and the
            ``render_state`` god-view dict.
        """
        seed_list = list(seeds)
        metrics = tabular_smoke.run_smoke(self._cfg, seed_list)
        game_id = f"p3-smoke-{int(seed_list[0])}"
        record, god_view = _helpers.build_p3_artifacts(self._cfg, seed_list, game_id)
        base = Path(out_dir) if out_dir is not None else Path(self._cfg["paths"]["subgames_dir"])
        path = base / f"{game_id}.json"
        self.write_subgame_json(record, path)
        return {**metrics, "subgame_path": str(path), "render_state": god_view}

    def write_subgame_json(self, result: dict, path: str | Path) -> None:
        """Write a minimal sub-game record to ``path`` as pretty JSON.

        Args:
            result: A minimal sub-game record (see subgame.schema.json).
            path: Destination ``.json`` path (parent dirs are created).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
