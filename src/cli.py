"""CLI entrypoint — ``python -m src.cli {train,play}`` routed through the SDK (§4.1).

Thin argument dispatch over :class:`MarlSDK` (the single business-logic seam): ``train
--algo {qmix,vdn,iql} --stage N`` runs one curriculum stage of self-play; ``play`` runs
the 6-sub-game match over the in-memory MCP and prints the §3.5 totals (DRY-RUN — the
lecturer email is a separate, explicit step, never triggered here). No business logic
lives here; every command calls the SDK.
"""

from __future__ import annotations

import argparse

from src.reporting.players import load_players
from src.sdk.sdk import MarlSDK
from src.utils.config_loader import load_config


def _train(sdk: object, cfg: dict, args: argparse.Namespace) -> dict:
    """Train one curriculum stage via self-play; print + return the final-round record."""
    seed = int(cfg["training"]["seeds"][0])
    history = sdk.train(args.algo, seed, args.stage)
    last = history[-1]
    print(
        f"[cli train] {args.algo} stage={args.stage} rounds={len(history)} capture={last['capture_rate']:.3f}"
    )
    return last


def _play(sdk: object, cfg: dict, args: argparse.Namespace) -> dict:
    """Run the 6-sub-game match over the in-memory MCP (dry-run); print + return the result."""
    out = sdk.run_local_match(
        sdk.fresh_net("cop"), sdk.fresh_net("thief"), load_players(), int(cfg["training"]["seeds"][0])
    )
    print(f"[cli play] {out['num_games']} sub-games | totals={out['report']['totals']}")
    return out


def build_parser() -> argparse.ArgumentParser:
    """Build the ``train`` / ``play`` subcommand parser."""
    parser = argparse.ArgumentParser(prog="src.cli", description="MARL Cops & Robbers CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="train one curriculum stage via self-play")
    train.add_argument("--algo", default="qmix", choices=["qmix", "vdn", "iql"])
    train.add_argument("--stage", type=int, default=0)
    train.set_defaults(func=_train)
    play = sub.add_parser("play", help="run a 6-sub-game match over the MCP (dry-run)")
    play.set_defaults(func=_play)
    return parser


def main(argv: list[str] | None = None, sdk: object = None) -> object:
    """Parse ``argv`` and dispatch the subcommand through the SDK (injectable for tests)."""
    cfg = load_config()
    args = build_parser().parse_args(argv)
    return args.func(sdk or MarlSDK(cfg), cfg, args)


if __name__ == "__main__":  # pragma: no cover - module CLI entry (python -m src.cli)
    main()
