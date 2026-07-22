"""§9 bonus send path — subject, dual-block redaction, gated idempotent send.

Mirrors the §3.5 pipeline (``src/reporting/send.py``) for the bonus email: validate the
§9.4 body, format the PDF-exact subject, write a redacted tracked copy that masks BOTH
groups' students AND BOTH repo URLs, then send EXACTLY once via the §5 ApiGatekeeper —
the sha256 sentinel is written inside the egress thunk, enforcing §9.3's
one-valid-bonus-email-per-group mechanically. Recipient is unconditionally ``gmail.to``.
The send remains HARD-GATED behind an explicit human "send" (same rule as §3.5).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from src.reporting.bonus import validate_bonus
from src.reporting.send import gated_idempotent_send

_MASK = "<redacted — real URL only in the git-ignored copy + the email>"


def format_bonus_subject(cfg: dict, report: dict) -> str:
    """Render ``gmail.bonus_subject_template`` — the §9.4 example shape, both group names."""
    return cfg["gmail"]["bonus_subject_template"].format(
        group_1=report["groups"]["group_1"], group_2=report["groups"]["group_2"]
    )


def redact_bonus(report: dict) -> dict:
    """Mask ALL PII in a bonus body: both student blocks to roles + both repo URLs.

    The partner group's names/IDs are PII exactly like ours — the tracked redacted copy
    keeps only roles and the game results; real identities live solely in the
    git-ignored ``players*.local.yaml`` files and the outbound email itself.
    """
    redacted = copy.deepcopy(report)
    for key in ("students_group_1", "students_group_2"):
        redacted[key] = [{"role": s["role"]} for s in report[key]]
    redacted["github_repo_group_1"] = _MASK
    redacted["github_repo_group_2"] = _MASK
    return redacted


def write_redacted_bonus(cfg: dict, report: dict) -> Path:
    """Write the redacted bonus copy under ``gmail.output_dir``; return its path."""
    out_dir = Path(cfg["gmail"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "bonus_report.redacted.json"
    path.write_text(json.dumps(redact_bonus(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def send_bonus_report(cfg: dict, report: dict, sender: object, gatekeeper: object = None) -> dict:
    """Validate -> subject -> redact-copy -> idempotent gatekeeper send; return a result dict.

    Identical discipline to :func:`src.reporting.send.send_report`: the sentinel digest is
    recorded inside the egress thunk (a deferred send can never double-email), and an
    already-sent digest is a no-op — §9.3 allows exactly ONE valid bonus email per group.

    Raises:
        ValueError: If the bonus body fails :func:`validate_bonus` (nothing is sent).
    """
    validate_bonus(report)
    subject = format_bonus_subject(cfg, report)
    redacted_path = str(write_redacted_bonus(cfg, report))
    return gated_idempotent_send(cfg, report, sender, subject, redacted_path, gatekeeper)
