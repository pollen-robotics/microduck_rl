"""Spawn a stopped post-setpriv child for the pre-bootstrap orphan test."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-fd", type=int, required=True)
    parser.add_argument("--retained-peer-fd", type=int, required=True)
    args = parser.parse_args()
    report = socket.socket(fileno=args.report_fd)
    expected = os.getpid()
    child = subprocess.Popen(
        [
            "/usr/bin/setpriv",
            "--pdeathsig",
            "SIGTERM",
            sys.executable,
            "-P",
            "-c",
            (
                "import os,signal,time;"
                "os.kill(os.getpid(),signal.SIGSTOP);"
                f"os.getppid()=={expected} or os._exit(70);"
                "time.sleep(30)"
            ),
        ],
        pass_fds=(args.retained_peer_fd,),
    )
    report.sendall(str(child.pid).encode("ascii"))
    return child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
