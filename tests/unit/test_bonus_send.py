"""RED->GREEN tests for the §9 bonus send path — subject, redaction, gated idempotent send.

Mirrors test_report_send for the bonus variant: the PDF-exact subject shape, BOTH
student blocks + BOTH repo URLs masked in the tracked redacted copy, and the
one-valid-email-per-group rule (§9.3) enforced by the sha256 sentinel.
"""

from __future__ import annotations

from src.reporting.bonus import build_bonus_report
from src.reporting.bonus_send import format_bonus_subject, redact_bonus, send_bonus_report
from src.utils.config_loader import load_config

_G1, _G2 = "adrl-001", "team-beta"


def _report() -> dict:
    result = {
        "start": "2026-07-04T18:00:00+03:00",
        "end": "2026-07-04T18:02:00+03:00",
        "moves": 9,
        "winner": "cop",
        "scores": {"cop": 20, "thief": 5},
    }
    return build_bonus_report(
        groups=(_G1, _G2),
        repos=("https://github.com/example/ours", "https://github.com/example/theirs"),
        students=(
            [{"role": "A", "full_name": "Placeholder One", "id": "12345"}],
            [{"role": "A", "full_name": "Placeholder Two", "id": "67890"}],
        ),
        timezone="Asia/Jerusalem",
        results=[result] * 6,
        mutual_agreement=True,
    )


class _Sender:
    def __init__(self) -> None:
        self.sent: list[tuple] = []

    def send(self, subject: str, body: str, to: str) -> None:
        self.sent.append((subject, body, to))


def _cfg(tmp_path) -> dict:
    cfg = load_config()
    cfg["gmail"]["sentinel"] = str(tmp_path / ".bonus_sent")
    cfg["gmail"]["output_dir"] = str(tmp_path / "reports")
    return cfg


def test_bonus_subject_matches_the_pdf_example_shape():
    """§9.4: `[MARL Bonus Game] X vs Y - Final Report` (both group names, explicit bonus)."""
    subject = format_bonus_subject(load_config(), _report())
    assert subject == f"[MARL Bonus Game] {_G1} vs {_G2} – Final Report"  # noqa: RUF001 — the §9.4 example uses an EN DASH; subject is PDF-exact


def test_redact_bonus_masks_both_student_blocks_and_both_repos():
    redacted = redact_bonus(_report())
    assert redacted["students_group_1"] == [{"role": "A"}]
    assert redacted["students_group_2"] == [{"role": "A"}]
    assert "github.com" not in redacted["github_repo_group_1"]
    assert "github.com" not in redacted["github_repo_group_2"]
    assert redacted["totals_by_group"] == _report()["totals_by_group"]  # results stay visible


def test_send_bonus_report_sends_once_and_writes_redacted_copy(tmp_path):
    """First send goes out (to gmail.to); an identical resend is a sentinel no-op (§9.3)."""
    cfg = _cfg(tmp_path)
    sender = _Sender()
    first = send_bonus_report(cfg, _report(), sender)
    assert first["sent"] is True and len(sender.sent) == 1
    assert sender.sent[0][2] == cfg["gmail"]["to"]
    assert (tmp_path / "reports" / "bonus_report.redacted.json").exists()
    again = send_bonus_report(cfg, _report(), sender)
    assert again["sent"] is False and again["reason"] == "already_sent"
    assert len(sender.sent) == 1


def test_send_bonus_report_rejects_an_invalid_body(tmp_path):
    """A tampered bonus body never leaves the machine (validate_bonus gates the send)."""
    cfg = _cfg(tmp_path)
    sender = _Sender()
    report = _report()
    report["bonus_claim"] = {_G1: 10, _G2: 10}
    try:
        send_bonus_report(cfg, report, sender)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    assert sender.sent == []
