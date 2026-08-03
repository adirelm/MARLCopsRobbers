"""The §9.4 bonus send path — agreement gate, compare digest, no accidental egress."""

from __future__ import annotations

import json

import pytest

from scripts.send_bonus_report import compare_digest, load_draft, main
from src.utils.config_loader import load_config


def _draft_exists(cfg) -> bool:
    from pathlib import Path  # noqa: PLC0415

    return Path(cfg["wire_match"]["draft_report"]).exists()


@pytest.fixture
def draft_cfg():
    cfg = load_config()
    if not _draft_exists(cfg):
        pytest.skip("git-ignored real draft absent (CI) — never assert a PII artifact exists")
    return cfg


def test_agreement_is_false_unless_explicitly_agreed(draft_cfg) -> None:
    """§9.3's gate must be an ACT: loading the draft never implies both groups compared."""
    assert load_draft(draft_cfg)["mutual_agreement"] is False
    assert load_draft(draft_cfg, agreed=True)["mutual_agreement"] is True


def test_compare_digest_is_canonical_and_identity_free(draft_cfg) -> None:
    """The exchanged digest must not depend on key order, and must carry no PII."""
    report = load_draft(draft_cfg)
    length, digest = compare_digest(report)
    shuffled = {k: report[k] for k in reversed(list(report))}
    assert compare_digest(shuffled) == (length, digest), "digest depends on key order"

    canon = json.dumps(
        {k: report[k] for k in ("sub_games", "totals_by_group", "bonus_claim")},
        sort_keys=True,
        separators=(",", ":"),
    )
    assert "full_name" not in canon and "github" not in canon
    assert not __import__("re").search(r"\b\d{9}\b", canon), "national ID leaked into the digest subject"


def test_main_without_send_performs_no_egress(draft_cfg, capsys) -> None:
    """The default run reports only — the one-per-match email stays behind --send."""
    main(draft_cfg, agreed=False, send=False)
    out = capsys.readouterr().out
    assert "mutual_agreement=False" in out and "compare subject" in out


def test_sending_without_agreement_is_refused(draft_cfg) -> None:
    """REGRESSION: emailing before the byte-compare scores BOTH groups zero under §9.3."""
    from src.reporting.bonus_send import send_bonus_report  # noqa: PLC0415

    with pytest.raises(ValueError, match="mutual_agreement must be EXACTLY True"):
        send_bonus_report(draft_cfg, load_draft(draft_cfg), sender=object())
