"""Install PID-1 termination handling before importing the application closure."""
# ruff: noqa: I001

from __future__ import annotations

import signal


def _terminate_before_server(_signum: int, _frame: object) -> None:
    """Exit cleanly while no simulator child can exist yet."""
    raise SystemExit(0)


signal.signal(signal.SIGTERM, _terminate_before_server)

# Publish a kernel-visible test/operator barrier only after PID 1 owns SIGTERM,
# and before importing any application closure that can create a runtime child.
import os

_ready_fd = os.open(
    "/tmp/.microduck-pid1-sigterm-ready",
    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
    0o600,
)
os.close(_ready_fd)

from mjlab_microduck.rom.main import main


if __name__ == "__main__":
    main()
