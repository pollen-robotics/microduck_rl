"""Parent-death harness: report the exact child PID, then await termination."""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

from mjlab_microduck.rom.process_supervisor import ChildLaunch, RuntimeProcessSupervisor

DIGEST = "sha256:" + "a" * 64


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-fd", type=int, required=True)
    args = parser.parse_args()

    report = socket.socket(fileno=args.report_fd)
    test_parent, test_child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    inherited = test_child.detach()

    def launch(control_fd: int) -> ChildLaunch:
        return ChildLaunch(
            (
                sys.executable,
                str(Path(__file__).with_name("fake_runtime_child.py")),
                "--socket-fd",
                str(control_fd),
                "--test-socket-fd",
                str(inherited),
                "--mode",
                "normal",
            ),
            pass_fds=(inherited,),
            close_after_spawn=(inherited,),
            env=os.environ.copy(),
        )

    supervisor = RuntimeProcessSupervisor(
        bundle_root="bundle",
        bundle_digest=DIGEST,
        launch_factory=launch,
        operation_timeout_s=1.0,
        terminate_timeout_s=0.1,
    )
    pid = supervisor.ensure_ready().pid
    assert pid is not None
    report.sendall(str(pid).encode("ascii"))
    report.recv(1)
    test_parent.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
