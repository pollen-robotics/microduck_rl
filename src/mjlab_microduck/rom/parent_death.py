"""Linux-only process and inherited-socket safety primitives."""

from __future__ import annotations

import ctypes
import os
import signal
import socket

_PR_SET_PDEATHSIG = 1


def verify_seqpacket_socket(fd: int) -> socket.socket:
    """Adopt *fd* only when it is an inherited Unix ``SOCK_SEQPACKET`` socket."""
    if not isinstance(fd, int) or isinstance(fd, bool) or fd < 0:
        raise ValueError("runtime socket descriptor is invalid")
    try:
        control = socket.socket(fileno=fd)
    except OSError as exc:
        raise ValueError("runtime socket descriptor is invalid") from exc
    if control.family != socket.AF_UNIX or (control.type & 0xF) != socket.SOCK_SEQPACKET:
        control.detach()
        raise ValueError("runtime socket must be Unix SOCK_SEQPACKET")
    return control


def close_unrelated_fds(preserve: set[int]) -> None:
    """Close inherited descriptors other than stdio and explicitly owned IPC."""
    allowed = {0, 1, 2, *preserve}
    try:
        descriptors = [int(name) for name in os.listdir("/proc/self/fd")]
    except OSError:
        limit = os.sysconf("SC_OPEN_MAX")
        descriptors = list(range(3, min(limit, 65_536)))
    for descriptor in descriptors:
        if descriptor in allowed:
            continue
        try:
            os.close(descriptor)
        except OSError:
            continue


def install_parent_death_signal(expected_parent_pid: int) -> None:
    """Bind SIGTERM delivery to the parent identity captured before ``fork``."""
    if (
        not isinstance(expected_parent_pid, int)
        or isinstance(expected_parent_pid, bool)
        or expected_parent_pid <= 0
        or os.getppid() != expected_parent_pid
    ):
        os._exit(70)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, "unable to install parent-death signal")
    if os.getppid() != expected_parent_pid:
        os._exit(70)
