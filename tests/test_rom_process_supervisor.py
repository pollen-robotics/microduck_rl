from __future__ import annotations

import os
import socket
import sys
import threading
from pathlib import Path

import pytest

from mjlab_microduck.rom.contracts import TaskCommandRequest, TaskCreateRequest
from mjlab_microduck.rom.process_supervisor import (
    ChildLaunch,
    RuntimeProcessSupervisor,
    SupervisorOperationError,
)


def test_supervisor_module_exposes_process_owner() -> None:
    assert RuntimeProcessSupervisor is not None


DIGEST = "sha256:" + "a" * 64
TASK_ID = "1" * 32


def _request() -> TaskCreateRequest:
    return TaskCreateRequest(
        schema="MICRODUCK_SIM_TASK_V1",
        taskId=TASK_ID,
        actionCode="WALK_VELOCITY",
        bundleVersion="1.0.0",
        bundleDigest=DIGEST,
        parameters={"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
        scenario={"terrain": "flat", "seed": 7},
        leaseMs=500,
        requestedBy="test",
    )


class FakeLaunch:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.test_peer: socket.socket | None = None

    def __call__(self, control_fd: int) -> ChildLaunch:
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.test_peer = parent
        inherited = child.detach()
        return ChildLaunch(
            (
                sys.executable,
                str(Path(__file__).parent / "fakes" / "fake_runtime_child.py"),
                "--socket-fd",
                str(control_fd),
                "--test-socket-fd",
                str(inherited),
                "--mode",
                self.mode,
            ),
            pass_fds=(inherited,),
            close_after_spawn=(inherited,),
            env=os.environ.copy(),
        )


def _supervisor(
    mode: str = "normal", **kwargs: object
) -> tuple[RuntimeProcessSupervisor, FakeLaunch]:
    launch = FakeLaunch(mode)
    supervisor = RuntimeProcessSupervisor(
        bundle_root="bundle",
        bundle_digest=DIGEST,
        launch_factory=launch,
        operation_timeout_s=2.0,
        terminate_timeout_s=0.1,
        **kwargs,
    )
    return supervisor, launch


def test_start_command_stop_reuses_exact_healthy_child_pid() -> None:
    supervisor, _launch = _supervisor()
    try:
        first = supervisor.ensure_ready().pid
        supervisor.start(_request())
        supervisor.command(
            TASK_ID,
            TaskCommandRequest(
                commandSequence=1,
                parameters={"vxMps": 0.1, "vyMps": 0.0, "yawRateRadps": 0.0},
                leaseMs=500,
            ),
        )
        assert supervisor.status(TASK_ID).health["healthy"] is True
        terminal = supervisor.stop(TASK_ID, "CANCELLED")
        assert terminal.evidence.stopReason == "CANCELLED"
        assert supervisor.snapshot().slot_releasable is True
        assert supervisor.ensure_ready().pid == first
        supervisor.start(_request())
        supervisor.stop(TASK_ID, "CANCELLED")
    finally:
        supervisor.close()


@pytest.mark.parametrize(
    "mode", ["block-load", "malformed-response", "exit-before-ack"]
)
def test_readiness_failure_reaps_exact_child_before_releasing_slot(mode: str) -> None:
    supervisor, launch = _supervisor(mode)
    with pytest.raises(SupervisorOperationError):
        supervisor.ensure_ready()
    snapshot = supervisor.snapshot()
    assert snapshot.pid is None
    assert snapshot.slot_releasable is True
    assert "CHILD_REAPED" in supervisor.trace
    supervisor.close()
    if launch.test_peer is not None:
        launch.test_peer.close()


@pytest.mark.parametrize("mode", ["block-start", "late-response"])
def test_start_failure_is_quarantined_and_reaped(mode: str) -> None:
    supervisor, launch = _supervisor(mode)
    if mode == "late-response":
        with pytest.raises(SupervisorOperationError):
            supervisor.ensure_ready()
    else:
        supervisor.ensure_ready()
        with pytest.raises(SupervisorOperationError):
            supervisor.start(_request())
    assert supervisor.snapshot().pid is None
    assert supervisor.snapshot().slot_releasable
    supervisor.close()
    if launch.test_peer is not None:
        launch.test_peer.close()


@pytest.mark.parametrize(
    "mode,operation", [("block-status", "status"), ("block-stop", "stop")]
)
def test_running_operation_failure_requires_exact_reap(
    mode: str, operation: str
) -> None:
    supervisor, launch = _supervisor(mode)
    supervisor.ensure_ready()
    supervisor.start(_request())
    with pytest.raises(SupervisorOperationError):
        if operation == "status":
            supervisor.status(TASK_ID)
        else:
            supervisor.stop(TASK_ID, "CANCELLED")
    assert supervisor.trace.index("QUARANTINED") < supervisor.trace.index(
        "CHILD_REAPED"
    )
    assert supervisor.snapshot().slot_releasable
    supervisor.close()
    assert launch.test_peer is not None
    launch.test_peer.close()


def test_close_escalates_sigterm_ignoring_exact_child() -> None:
    supervisor, launch = _supervisor("ignore-sigterm")
    pid = supervisor.ensure_ready().pid
    assert pid is not None
    supervisor.close()
    assert "SIGKILL_SENT" in supervisor.trace
    assert supervisor.snapshot().pid is None
    assert supervisor.snapshot().slot_releasable
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert launch.test_peer is not None
    launch.test_peer.close()


def test_blocked_command_quarantines_and_reaps_before_slot_release() -> None:
    supervisor, launch = _supervisor("block-command")
    supervisor.ensure_ready()
    supervisor.start(_request())
    with pytest.raises(SupervisorOperationError):
        supervisor.command(TASK_ID, {"vxMps": 0.0}, 500)
    assert supervisor.snapshot().pid is None
    assert supervisor.snapshot().slot_releasable
    assert supervisor.trace.index("QUARANTINED") < supervisor.trace.index(
        "CHILD_REAPED"
    )
    assert supervisor.trace[-1] == "NO_CHILD"
    supervisor.close()
    assert launch.test_peer is not None
    launch.test_peer.close()


def test_24_callers_share_one_owner_thread_and_one_child() -> None:
    supervisor, launch = _supervisor("block-status", queue_size=2)
    supervisor.ensure_ready()
    supervisor.start(_request())
    before = {thread.name for thread in threading.enumerate()}
    outcomes: list[type[BaseException] | None] = []

    def call() -> None:
        try:
            supervisor.status(TASK_ID)
        except Exception as exc:  # noqa: BLE001 - all bounded failures are expected
            outcomes.append(type(exc))
        else:
            outcomes.append(None)

    callers = [threading.Thread(target=call) for _ in range(24)]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(timeout=3)
    assert all(not caller.is_alive() for caller in callers)
    assert len(outcomes) == 24
    assert (
        sum(t.name == "microduck-runtime-supervisor" for t in threading.enumerate())
        == 1
    )
    assert not (
        {thread.name for thread in threading.enumerate()}
        - before
        - {c.name for c in callers}
    )
    assert supervisor.snapshot().slot_releasable
    supervisor.close()
    assert launch.test_peer is not None
    launch.test_peer.close()
