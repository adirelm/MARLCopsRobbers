"""MarlSDK — the single business-logic entry point (CLAUDE.md §3, BRIEF §6).

A thin FACADE: every public method routes through ``src.*`` building blocks (env /
self-play trainer / OLoRA finetune / checkpoints / render seam) with no business
logic duplicated here. It is the ONLY entry UIs / the MCP / scripts / the sweep
import: build an env, ``train`` (qmix/vdn/iql) one curriculum stage via self-play,
OLoRA-``finetune`` up the curriculum, ``build_policy`` for acting, and ``export``/
``load`` the agent weights. (The throwaway P3 tabular smoke was removed in P4.)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.marl.env.cops_robbers_env import CopsRobbersEnv
from src.marl.nets.olora_linear import wrap_encoder
from src.sdk._train_helpers import cfg_for_algo
from src.services.checkpoints import export_agent_weights, load_agent_weights
from src.services.finetune import finetune_curriculum, stage_params
from src.services.policy import RecurrentPolicy
from src.services.spectator import SpectatorSession
from src.services.trainer import SelfPlayTrainer
from src.utils.compute import apply_compute_limits


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
        return CopsRobbersEnv(self._cfg, h=h, w=w, num_cops=num_cops)

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
        curriculum via :func:`~src.services.finetune.finetune_curriculum`. Always
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

    def run_local_match(self, cop_net, thief_net, players, seed, stage=(5, 5, 1), num_games=None) -> dict:  # noqa: PLR0913
        """Play a full local match over MCP; assemble + validate the §3.5 report (no send).

        The single entry the orchestrator script uses: both servers play the
        referee-driven match over the canonical tool contract, then the §3.5 body
        is assembled + validated and dry-run sent. Returns ``{report, num_games, ack}``.
        """
        from src.mcp.match import run_local_match as _run  # noqa: PLC0415 — lazy: breaks the sdk<->mcp cycle

        return asyncio.run(_run(self._cfg, cop_net, thief_net, players, seed, stage, num_games))

    def spectator_session(self, h: int = 5, w: int = 5, num_cops: int = 1, seed: int = 0) -> SpectatorSession:
        """Return a god-view :class:`SpectatorSession` (the GUI's single entry, T7.1)."""
        return SpectatorSession(self._cfg, h, w, num_cops, seed)

    def write_subgame_json(self, result: dict, path: str | Path) -> None:
        """Write a minimal sub-game record to ``path`` as pretty JSON.

        Args:
            result: A minimal sub-game record (see subgame.schema.json).
            path: Destination ``.json`` path (parent dirs are created).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    def send_final_report(self, report: dict, sender: object = None, date_str: str | None = None) -> dict:
        """Idempotently send the §3.5 report via Gmail + write a redacted copy (T9.5).

        The single send entry. Defaults to the real :class:`GmailMailer` (env creds
        ``GMAIL_SENDER`` / ``GMAIL_APP_PASSWORD``); pass a ``FakeEmailSender`` for a dry
        run. ``date_str`` defaults to today in the configured project timezone. The send
        is sentinel-guarded so the lecturer is emailed EXACTLY once. Returns the result.
        """
        from src.reporting.mailer import GmailMailer  # noqa: PLC0415 — lazy keeps import light
        from src.reporting.send import build_date, send_report  # noqa: PLC0415 — lazy

        sender = sender or GmailMailer(self._cfg)
        return send_report(self._cfg, report, sender, date_str or build_date(self._cfg))
