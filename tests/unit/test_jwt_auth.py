"""Cloud JWT auth tests (T8.1) — RS256 verify + jti revocation + the build_verifier seam."""

from __future__ import annotations

import asyncio
import os

import pytest
from fastmcp.server.auth import StaticTokenVerifier
from fastmcp.server.auth.providers.jwt import RSAKeyPair

from src.marl.nets.agent_net import RecurrentQNet
from src.mcp.auth import build_verifier
from src.mcp.cloud import build_cloud_server
from src.mcp.jwt_auth import RevocableJWTVerifier, build_jwt_verifier, revoked_jtis


def _mint(keypair, cfg, role, jti="jti-1"):
    auth = cfg["mcp"]["auth"]
    return keypair.create_token(
        subject=f"marl-{role}",
        issuer=auth["issuer"],
        audience=auth[f"{role}_audience"],
        scopes=list(auth["required_scopes"]),
        additional_claims={"jti": jti},
    )


def test_jwt_verifier_accepts_a_valid_token(cfg, monkeypatch):
    keypair = RSAKeyPair.generate()
    monkeypatch.setenv("MCP_PUBLIC_KEY", keypair.public_key)
    access = asyncio.run(build_jwt_verifier(cfg, "cop").verify_token(_mint(keypair, cfg, "cop")))
    assert access is not None and access.claims["aud"] == "marl-cop"


def test_jwt_verifier_rejects_revoked_jti(cfg, monkeypatch):
    keypair = RSAKeyPair.generate()
    monkeypatch.setenv("MCP_PUBLIC_KEY", keypair.public_key)
    monkeypatch.setenv("REVOKED_TOKEN_JTIS", "bad-jti, other")
    verifier = build_jwt_verifier(cfg, "cop")
    assert asyncio.run(verifier.verify_token(_mint(keypair, cfg, "cop", jti="bad-jti"))) is None
    assert asyncio.run(verifier.verify_token(_mint(keypair, cfg, "cop", jti="ok"))) is not None


def test_jwt_verifier_rejects_wrong_audience(cfg, monkeypatch):
    keypair = RSAKeyPair.generate()
    monkeypatch.setenv("MCP_PUBLIC_KEY", keypair.public_key)
    # a thief-audience token must not authenticate against the cop verifier
    assert asyncio.run(build_jwt_verifier(cfg, "cop").verify_token(_mint(keypair, cfg, "thief"))) is None


def test_build_jwt_verifier_requires_public_key(cfg, monkeypatch):
    monkeypatch.delenv("MCP_PUBLIC_KEY", raising=False)
    with pytest.raises(ValueError, match="MCP_PUBLIC_KEY"):
        build_jwt_verifier(cfg, "cop")


def test_revoked_jtis_parses_csv():
    assert revoked_jtis("a, b ,, c") == ("a", "b", "c")
    assert revoked_jtis("") == ()


def test_build_verifier_dispatches_to_jwt_when_env_set(cfg, monkeypatch):
    keypair = RSAKeyPair.generate()
    monkeypatch.setenv("MCP_AUTH_MODE", "jwt")
    monkeypatch.setenv("MCP_PUBLIC_KEY", keypair.public_key)
    assert isinstance(build_verifier(cfg, "cop"), RevocableJWTVerifier)


def test_build_verifier_defaults_to_static(cfg, monkeypatch):
    monkeypatch.delenv("MCP_AUTH_MODE", raising=False)
    assert isinstance(build_verifier(cfg, "cop", token="dev-token"), StaticTokenVerifier)


def test_build_cloud_server_is_jwt_and_restores_env(cfg, monkeypatch):
    keypair = RSAKeyPair.generate()
    monkeypatch.delenv("MCP_AUTH_MODE", raising=False)
    monkeypatch.setenv("MCP_PUBLIC_KEY", keypair.public_key)
    server = build_cloud_server("cop", net=RecurrentQNet(cfg, "cop", 2), cfg=cfg)
    assert server.name == "marl-cop"
    assert "MCP_AUTH_MODE" not in os.environ  # the scoped env was restored after the build


def test_build_cloud_server_restores_prior_auth_mode(cfg, monkeypatch):
    keypair = RSAKeyPair.generate()
    monkeypatch.setenv("MCP_AUTH_MODE", "static")  # a prior value must be restored, not dropped
    monkeypatch.setenv("MCP_PUBLIC_KEY", keypair.public_key)
    server = build_cloud_server("thief", net=RecurrentQNet(cfg, "thief", 1), cfg=cfg)
    assert server.name == "marl-thief"
    assert os.environ["MCP_AUTH_MODE"] == "static"
