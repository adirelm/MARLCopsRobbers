"""Two-PROCESS sentinel race pin (wave-2 SENT): at most ONE email leaves the machine.

Before the inter-process ``send_lock``, two independently launched send processes
could both pass ``check_clear_to_send`` before either recorded ``intent`` — and both
email (reproduced with this exact harness). Now the second process is refused while
the first holds ``<sentinel>.lock``, and exactly one email is sent. Deterministic:
the hold child parks INSIDE the critical section until the parent releases it, and
the parent only releases AFTER the racing child has exited.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

_CHILD = Path(__file__).with_name("_sentinel_race_child.py")
_REPO = Path(__file__).resolve().parents[2]


def _spawn(args: list[str], **env_over) -> subprocess.Popen:
    env = {**os.environ, **{k: str(v) for k, v in env_over.items()}}
    return subprocess.Popen(
        [sys.executable, str(_CHILD), str(_REPO), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )


def test_second_process_cannot_double_email_through_the_intent_window(tmp_path):
    """hold parks between check and intent; fast races it as a SEPARATE process -> refused."""
    sentinel = str(tmp_path / "match.sha256")
    sent_log = tmp_path / "emails.log"
    ready, go = tmp_path / "ready", tmp_path / "go"

    hold = _spawn([sentinel, str(sent_log), "hold", str(ready), str(go)])
    try:
        deadline = time.monotonic() + 60
        while not ready.exists():  # hold is inside the critical section once this appears
            assert hold.poll() is None, hold.communicate()
            assert time.monotonic() < deadline, "hold child never reached the critical section"
            time.sleep(0.01)

        fast = _spawn(
            [sentinel, str(sent_log), "fast", str(tmp_path / "r2"), str(go)],
            SENTINEL_LOCK_TIMEOUT_S="0.4",
        )
        fast_out, _ = fast.communicate(timeout=60)
        assert fast.returncode == 3, fast_out  # refused — could NOT enter the locked window
        assert "held by another send process" in fast_out
    finally:
        go.touch()  # always release the parked child
    hold_out, _ = hold.communicate(timeout=60)
    assert hold.returncode == 0 and "RESULT True" in hold_out

    emails = sent_log.read_text(encoding="utf-8").splitlines() if sent_log.exists() else []
    assert emails == ["hold"]  # EXACTLY one email left the machine
    recorded = Path(sentinel).read_text(encoding="utf-8")
    assert recorded.count("intent ") == 1 and recorded.count("sent ") == 1
    assert not Path(sentinel + ".lock").exists()  # released on the way out
