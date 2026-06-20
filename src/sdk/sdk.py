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
from src.marl.nets.olora_linear import wrap_encoder
from src.marl.replay import tabular_smoke
from src.sdk import _helpers
from src.sdk._train_helpers import cfg_for_algo
from src.services.checkpoints import export_agent_weights, load_agent_weights
from src.services.finetune import finetune_curriculum, stage_params
from src.services.policy import RecurrentPolicy
from src.services.trainer import SelfPlayTrainer
from src.utils.compute import apply_compute_limits

Policy = Callable[..., object]


class MarlSDK:
    """The single sanctioned entry point for MARL Cops & Robbers business logic."""

    def __init__(self, cfg: dict) -> None:
        """Bind the SDK to a loaded config and apply compute (thread) governance.

        Capping the torch thread pools at the single entry point means every
        SDK-driven training path is bounded and can never grab all cores and
        freeze the host (:func:`src.utils.compute.apply_compute_limits`).

        Args:
            cfg: The loaded project config (config/config.yaml via config_loader).
        """
        self._cfg = cfg
        apply_compute_limits(cfg)

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

    def train(self, algorithm: str, seed: int, stage_idx: int = 0) -> list[dict]:
        """Train one curriculum stage via self-play (the single training entry).

        Routes through :class:`~src.services.trainer.SelfPlayTrainer` (compute
        thread caps applied on its construction, so a full run cannot freeze the
        host). UIs / scripts / sweeps call ONLY this — never the trainer directly.

        Args:
            algorithm: ``"qmix"`` (primary), ``"vdn"`` (ablation), or ``"iql"``.
            seed: Master training seed (reproducible).
            stage_idx: Curriculum stage index (``env.curriculum.stages``); 0 == 2x2.

        Returns:
            The per-round self-play history (``round`` / ``role`` / ``loss`` /
            ``capture_rate`` dicts).
        """
        cfg = cfg_for_algo(self._cfg, algorithm)
        h, w, num_cops = stage_params(cfg, stage_idx)
        return SelfPlayTrainer(cfg, seed, h, w, num_cops).train_stage()

    def finetune(  # noqa: PLR0913 — base nets + seed + stages + olora flag + rounds are all distinct
        self,
        seed: int,
        stage_indices: list[int],
        cop_net: object,
        thief_net: object,
        olora: bool = True,
        rounds_per_stage: int | None = None,
    ) -> dict:
        """OLoRA-wrapped CTDE curriculum finetune across stages (the T4.5 loop).

        Wraps the BC-pretrained base nets with OLoRA (encoder adapters; frozen
        base + GRU) when ``olora`` is set, then carries them through the
        curriculum via :func:`~src.services.curriculum.finetune_curriculum`. Always
        the QMIX arm. Routes through the SDK so scripts stay thin.

        Args:
            seed: Master finetune seed.
            stage_indices: Ordered curriculum stage indices (e.g. ``[1, 2, 3]``).
            cop_net: The BC-pretrained cop base net (``n_agents=2``).
            thief_net: The BC-pretrained thief base net (``n_agents=1``).
            olora: Wrap the encoders with OLoRA before finetuning (default True).
            rounds_per_stage: Rounds per stage (default ``selfplay.rounds``).

        Returns:
            ``{"history", "cop_net", "thief_net"}`` from the curriculum loop.
        """
        cfg = cfg_for_algo(self._cfg, "qmix")
        if olora:
            cop_net = wrap_encoder(cop_net, cfg)
            thief_net = wrap_encoder(thief_net, cfg)
        return finetune_curriculum(cfg, seed, stage_indices, cop_net, thief_net, rounds_per_stage)

    def build_policy(self, role: str, net: object, n_agents: int = 1) -> RecurrentPolicy:
        """Return an acting :class:`RecurrentPolicy` over ``net`` for ``n_agents``.

        The single local-obs acting seam UIs / the MCP controller use (carries the
        GRU hidden state, picks legal ε-greedy actions, never sees global state).
        ``net`` is any trained role net (dense or OLoRA-wrapped/bundle-loaded).
        """
        return RecurrentPolicy(net, n_agents)

    def export_weights(self, net: object, role: str, path: str | Path) -> Path:
        """Export ONLY the agent net + shape sidecar (no mixer/global state); return the sidecar."""
        return export_agent_weights(net, path, self._cfg, role)

    def load_weights(self, role: str, path: str | Path, n_agents: int) -> object:
        """Rebuild a dense role net from an exported ``state_dict`` (inverse of export_weights)."""
        return load_agent_weights(path, self._cfg, role, n_agents)

    def write_subgame_json(self, result: dict, path: str | Path) -> None:
        """Write a minimal sub-game record to ``path`` as pretty JSON.

        Args:
            result: A minimal sub-game record (see subgame.schema.json).
            path: Destination ``.json`` path (parent dirs are created).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
