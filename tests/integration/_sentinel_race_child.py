"""Child process for the two-PROCESS sentinel race pin (wave-2 SENT finding).

argv: repo_root sentinel_path sent_log_path mode ready_path go_path

mode "hold": patch ``mark_intent`` to park INSIDE the check->intent critical
section (touch ``ready``, wait for ``go``) — exactly the window two processes
could both occupy before the inter-process lock existed. mode "fast": run
straight through while hold is parked. Prints RESULT/REFUSED; exit 3 on refusal.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def main() -> None:
    repo, sentinel, sent_log, mode, ready, go = sys.argv[1:7]
    sys.path.insert(0, repo)
    from src.reporting import send as send_mod  # noqa: PLC0415 - must follow the sys.path insert

    if mode == "hold":
        original_mark_intent = send_mod.mark_intent

        def parked_mark_intent(path: str, digest: str) -> None:
            Path(ready).touch()  # signal: check passed, intent NOT yet written
            while not Path(go).exists():
                time.sleep(0.01)
            original_mark_intent(path, digest)

        send_mod.mark_intent = parked_mark_intent

    class _Sender:
        def send(self, subject: str, body: str, recipient: str) -> None:
            with open(sent_log, "a", encoding="utf-8") as handle:
                handle.write(mode + "\n")

    class _Gate:
        def execute(self, channel: str, thunk):
            return thunk()

    cfg = {"gmail": {"sentinel": sentinel, "to": "race-pin@example.com"}}
    try:
        out = send_mod.gated_idempotent_send(cfg, {"pin": "two-process-race"}, _Sender(), "s", "-", _Gate())
        print("RESULT", out.get("sent"), out.get("reason"))
    except (RuntimeError, ValueError) as exc:
        print("REFUSED", type(exc).__name__, str(exc)[:400])
        sys.exit(3)


if __name__ == "__main__":
    main()
