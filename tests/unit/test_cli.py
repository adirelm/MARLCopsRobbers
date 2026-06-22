"""CLI tests (T11/§4.1) — parser + train/play dispatch route through the SDK (fake, no training)."""

from __future__ import annotations

from argparse import Namespace

from src.cli import _play, _train, build_parser, main


def test_parser_train_and_play():
    parser = build_parser()
    train = parser.parse_args(["train", "--algo", "vdn", "--stage", "1"])
    assert train.command == "train" and train.algo == "vdn" and train.stage == 1
    assert parser.parse_args(["play"]).command == "play"


class _FakeSDK:
    def train(self, algo, seed, stage):
        return [{"round": 0, "role": "cop", "loss": 0.1, "capture_rate": 0.5}]

    def fresh_net(self, role, n_agents=None):
        return f"net:{role}"

    def run_local_match(self, *args, **kwargs):
        return {"num_games": 6, "report": {"totals": {"cop": 20, "thief": 5}}}


def test_train_dispatch_uses_sdk(cfg):
    last = _train(_FakeSDK(), cfg, Namespace(algo="qmix", stage=0))
    assert last["capture_rate"] == 0.5


def test_play_dispatch_uses_sdk(cfg):
    out = _play(_FakeSDK(), cfg, Namespace())
    assert out["num_games"] == 6 and out["report"]["totals"]["cop"] == 20


def test_main_routes_injected_sdk():
    out = main(["play"], sdk=_FakeSDK())
    assert out["num_games"] == 6
