from __future__ import annotations

import os
import select
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from mjlab_microduck.rom.contracts import (
    TaskCommandRequest,
    TaskCreateRequest,
    TaskEvidence,
)
from mjlab_microduck.rom.process_protocol import TerminalPayload
from mjlab_microduck.rom.process_supervisor import (
    ChildLaunch,
    RuntimeProcessSupervisor,
    SupervisorOperationError,
    SupervisorTaskTerminalized,
    SupervisorUnavailable,
)
from mjlab_microduck.rom.supervisor_state import SupervisorState


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
        self.test_peers: list[socket.socket] = []
        self.launched = threading.Event()

    def __call__(self, control_fd: int) -> ChildLaunch:
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.test_peer = parent
        self.test_peers.append(parent)
        self.launched.set()
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
    mode: str = "normal", *, operation_timeout_s: float = 0.75, **kwargs: object
) -> tuple[RuntimeProcessSupervisor, FakeLaunch]:
    launch = FakeLaunch(mode)
    supervisor = RuntimeProcessSupervisor(
        bundle_root="bundle",
        bundle_digest=DIGEST,
        launch_factory=launch,
        operation_timeout_s=operation_timeout_s,
        terminate_timeout_s=0.1,
        **kwargs,
    )
    return supervisor, launch


def _receive_gate(launch: FakeLaunch, expected: bytes) -> socket.socket:
    assert launch.launched.wait(timeout=2)
    peer = launch.test_peer
    assert peer is not None
    peer.settimeout(2)
    assert peer.recv(64) == expected
    return peer


def _assert_pidfd_dead(pidfd: int) -> None:
    try:
        readable, _, _ = select.select([pidfd], [], [], 0)
        assert readable == [pidfd]
    finally:
        os.close(pidfd)


def _assert_containment_trace(
    supervisor: RuntimeProcessSupervisor,
    *,
    leading: str = "OPERATION_TIMEOUT",
    killed: bool = False,
) -> None:
    expected = [leading, "QUARANTINED", "SIGTERM_SENT"]
    if killed:
        expected += ["TERM_TIMEOUT", "SIGKILL_SENT"]
    expected += ["CHILD_REAPED", "NO_CHILD"]
    assert supervisor.trace == tuple(expected)
    assert supervisor.snapshot().state is SupervisorState.NO_CHILD
    assert supervisor.snapshot().pid is None
    assert supervisor.snapshot().slot_releasable is True


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


def test_idle_owner_consumes_unsolicited_terminal_and_releases_slot() -> None:
    delivered = threading.Event()
    supervisor, launch = _supervisor(
        "terminal-event", terminal_callback=lambda _payload: delivered.set()
    )
    try:
        supervisor.ensure_ready()
        supervisor.start(_request())
        peer = _receive_gate(launch, b"STARTED")
        peer.sendall(b"EMIT")
        assert delivered.wait(timeout=2)
        snapshot = supervisor.snapshot()
        assert snapshot.cached_terminal is not None
        assert snapshot.cached_terminal.outcome == "SUCCEEDED"
        assert snapshot.state is SupervisorState.IDLE
        assert snapshot.slot_releasable is True
    finally:
        supervisor.close()


def test_terminal_event_interleaved_with_status_is_not_consumed_as_response() -> None:
    delivered = threading.Event()
    supervisor, _launch = _supervisor(
        "event-before-status", terminal_callback=lambda _payload: delivered.set()
    )
    try:
        supervisor.ensure_ready()
        supervisor.start(_request())
        with pytest.raises(SupervisorTaskTerminalized):
            supervisor.status(TASK_ID)
        assert delivered.wait(timeout=2)
        assert supervisor.snapshot().cached_terminal is not None
    finally:
        supervisor.close()


def test_terminal_event_interleaved_with_command_is_not_consumed_as_response() -> None:
    delivered = threading.Event()
    supervisor, _launch = _supervisor(
        "event-before-command", terminal_callback=lambda _payload: delivered.set()
    )
    try:
        supervisor.ensure_ready()
        supervisor.start(_request())
        with pytest.raises(SupervisorTaskTerminalized):
            supervisor.command(
                TASK_ID,
                TaskCommandRequest(
                    commandSequence=1,
                    parameters={"vxMps": 0.1, "vyMps": 0.0, "yawRateRadps": 0.0},
                    leaseMs=500,
                ),
            )
        assert delivered.wait(timeout=2)
        assert supervisor.snapshot().cached_terminal is not None
    finally:
        supervisor.close()


def test_terminal_callback_observes_published_idle_snapshot() -> None:
    observed: list[object] = []
    delivered = threading.Event()
    supervisor: RuntimeProcessSupervisor

    def callback(terminal: object) -> None:
        observed.extend((terminal, supervisor.snapshot()))
        delivered.set()

    supervisor, launch = _supervisor("terminal-event", terminal_callback=callback)
    try:
        supervisor.ensure_ready()
        supervisor.start(_request())
        peer = _receive_gate(launch, b"STARTED")
        peer.sendall(b"EMIT")
        assert delivered.wait(timeout=2)
        terminal, snapshot = observed
        assert snapshot.state is SupervisorState.IDLE
        assert snapshot.cached_terminal == terminal
        assert snapshot.slot_releasable is True
    finally:
        supervisor.close()


def test_terminal_callback_backpressure_preserves_truthful_idle_cache() -> None:
    blocked = threading.Event()
    release = threading.Event()
    supervisor, launch = _supervisor(
        "terminal-event",
        terminal_callback=lambda _payload: (blocked.set(), release.wait(timeout=2)),
    )
    try:
        supervisor.ensure_ready()
        supervisor.start(_request())
        assert supervisor._terminal_queue is not None
        supervisor._terminal_queue.put_nowait(TerminalPayload(
            outcome="CANCELLED",
            evidence=TaskEvidence(
                bundleDigest=DIGEST, policyDigest="sha256:" + "b" * 64,
                modelDigest="sha256:" + "c" * 64, stopReason="CANCELLED",
            ),
        ))
        assert blocked.wait(timeout=1)
        supervisor._terminal_queue.put_nowait(TerminalPayload(
            outcome="CANCELLED",
            evidence=TaskEvidence(
                bundleDigest=DIGEST, policyDigest="sha256:" + "b" * 64,
                modelDigest="sha256:" + "c" * 64, stopReason="CANCELLED",
            ),
        ))
        peer = _receive_gate(launch, b"STARTED")
        peer.sendall(b"EMIT")
        deadline = time.monotonic() + 1
        while supervisor.snapshot().state is not SupervisorState.IDLE and time.monotonic() < deadline:
            time.sleep(0.01)
        assert supervisor.snapshot().state is SupervisorState.IDLE
        assert supervisor.snapshot().cached_terminal is not None
        assert "TERMINAL_DELIVERY_BACKPRESSURE" in supervisor.trace
        assert supervisor.snapshot().pid is not None
    finally:
        release.set()
        supervisor.close()


@pytest.mark.parametrize("mode", ["duplicate-event", "stale-event", "malformed-event"])
def test_replayed_stale_or_malformed_terminal_event_quarantines(mode: str) -> None:
    supervisor, _launch = _supervisor(mode)
    supervisor.ensure_ready()
    supervisor.start(_request())
    deadline = time.monotonic() + 2
    while supervisor.snapshot().pid is not None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert supervisor.snapshot().state is SupervisorState.NO_CHILD
    assert "QUARANTINED" in supervisor.trace
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
    "mode,gate,operation",
    [
        ("block-load", b"LOAD", "ready"),
        ("block-start", b"START", "start"),
        ("block-command", b"COMMAND", "command"),
        ("block-status", b"STATUS", "status"),
        ("block-stop", b"ZERO_AND_STOP", "stop"),
    ],
)
def test_each_block_mode_has_ordered_reap_barrier(
    mode: str, gate: bytes, operation: str
) -> None:
    supervisor, launch = _supervisor(mode)
    if operation != "ready":
        supervisor.ensure_ready()
    if operation not in {"ready", "start"}:
        supervisor.start(_request())
    errors: list[BaseException] = []

    def invoke() -> None:
        try:
            if operation == "ready":
                supervisor.ensure_ready()
            elif operation == "start":
                supervisor.start(_request())
            elif operation == "command":
                supervisor.command(TASK_ID, {"vxMps": 0.0}, 500)
            elif operation == "status":
                supervisor.status(TASK_ID)
            else:
                supervisor.stop(TASK_ID, "CANCELLED")
        except BaseException as exc:  # noqa: BLE001 - expected fail-closed result
            errors.append(exc)

    caller = threading.Thread(target=invoke)
    caller.start()
    _receive_gate(launch, gate)
    pid = supervisor.snapshot().pid
    assert pid is not None
    pidfd = os.pidfd_open(pid)
    caller.join(timeout=2)
    assert not caller.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], SupervisorOperationError)
    _assert_pidfd_dead(pidfd)
    _assert_containment_trace(supervisor)
    supervisor.close()
    for peer in launch.test_peers:
        peer.close()


def test_late_packet_is_released_only_by_post_deadline_sigterm() -> None:
    supervisor, launch = _supervisor("late-response")
    errors: list[BaseException] = []

    def ready() -> None:
        try:
            supervisor.ensure_ready()
        except BaseException as exc:  # noqa: BLE001 - expected timeout containment
            errors.append(exc)

    caller = threading.Thread(target=ready)
    caller.start()
    peer = _receive_gate(launch, b"HELLO")
    pid = supervisor.snapshot().pid
    assert pid is not None
    pidfd = os.pidfd_open(pid)
    # No timer releases the fake. Its SIGTERM handler sends the late response,
    # proving the packet was emitted only after the supervisor's deadline path.
    assert peer.recv(64) == b"LATE_SENT"
    caller.join(timeout=2)
    assert not caller.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], SupervisorOperationError)
    assert supervisor.trace.index("OPERATION_TIMEOUT") < supervisor.trace.index(
        "SIGTERM_SENT"
    )
    # Exact PID death is the release barrier; inspect it before availability.
    _assert_pidfd_dead(pidfd)
    _assert_containment_trace(supervisor, killed=True)
    supervisor.close()
    peer.close()


def test_old_generation_packet_after_reap_cannot_claim_replacement() -> None:
    supervisor, launch = _supervisor("exit-before-ack")
    with pytest.raises(SupervisorOperationError):
        supervisor.ensure_ready()
    assert supervisor.snapshot().generation == 1
    first_peer = launch.test_peer
    launch.mode = "stale-generation"
    launch.launched.clear()
    second_pid = supervisor.ensure_ready().pid
    assert second_pid is not None
    second_pidfd = os.pidfd_open(second_pid)
    assert supervisor.snapshot().generation == 2

    errors: list[BaseException] = []

    def start() -> None:
        try:
            supervisor.start(_request())
        except BaseException as exc:  # noqa: BLE001 - stale response must fail closed
            errors.append(exc)

    caller = threading.Thread(target=start)
    caller.start()
    second_peer = _receive_gate(launch, b"START")
    second_peer.sendall(b"1")
    caller.join(timeout=2)
    assert not caller.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], SupervisorOperationError)
    assert supervisor.snapshot().generation == 2
    _assert_pidfd_dead(second_pidfd)
    supervisor.close()
    assert first_peer is not None
    first_peer.close()
    second_peer.close()


@pytest.mark.parametrize(
    "mode,operation",
    [
        ("wrong-hello-ack", "ready"),
        ("wrong-start-ack", "start"),
        ("wrong-command-ack", "command"),
        ("wrong-shutdown-ack", "close"),
    ],
)
def test_every_ack_exchange_rejects_mismatched_ack(
    mode: str, operation: str
) -> None:
    supervisor, launch = _supervisor(mode)
    if operation != "ready":
        supervisor.ensure_ready()
    if operation in {"command"}:
        supervisor.start(_request())
    pidfd: int | None = None
    if operation != "ready":
        pid = supervisor.snapshot().pid
        assert pid is not None
        pidfd = os.pidfd_open(pid)
    if operation == "close":
        supervisor.close()
        assert any(item.startswith("SHUTDOWN_FAILED") for item in supervisor.trace)
    else:
        with pytest.raises(SupervisorOperationError):
            if operation == "ready":
                supervisor.ensure_ready()
            elif operation == "start":
                supervisor.start(_request())
            else:
                supervisor.command(TASK_ID, {"vxMps": 0.0}, 500)
        supervisor.close()
    if pidfd is not None:
        _assert_pidfd_dead(pidfd)
    for peer in launch.test_peers:
        peer.close()


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


def test_blocking_terminal_callback_cannot_block_owner_or_close() -> None:
    entered = threading.Event()
    release = threading.Event()

    def callback(_terminal: object) -> None:
        entered.set()
        release.wait()

    supervisor, launch = _supervisor(
        "normal", operation_timeout_s=2.0, terminal_callback=callback
    )
    supervisor.start(_request())
    supervisor.stop(TASK_ID, "CANCELLED")
    assert entered.wait(timeout=1)
    with pytest.raises(SupervisorUnavailable, match="terminal delivery worker"):
        supervisor.close()
    assert supervisor.snapshot().pid is None
    assert supervisor.snapshot().state is SupervisorState.NO_CHILD
    assert supervisor.terminal_delivery_alive
    assert "TERMINAL_WORKER_ABANDONED" in supervisor.trace
    release.set()
    supervisor.close()
    assert not supervisor.terminal_delivery_alive
    assert supervisor.trace[-1] == "TERMINAL_WORKER_TERMINATED"
    assert launch.test_peer is not None
    launch.test_peer.close()


def test_throwing_terminal_callback_isolated_from_acknowledged_stop() -> None:
    called = threading.Event()

    def callback(_terminal: object) -> None:
        called.set()
        raise RuntimeError("test callback failure")

    supervisor, launch = _supervisor(
        "normal", operation_timeout_s=2.0, terminal_callback=callback
    )
    supervisor.start(_request())
    terminal = supervisor.stop(TASK_ID, "CANCELLED")
    assert terminal.evidence.stopReason == "CANCELLED"
    assert called.wait(timeout=1)
    supervisor.close()
    assert any(item.startswith("TERMINAL_DELIVERY_FAILED") for item in supervisor.trace)
    assert not supervisor.terminal_delivery_alive
    assert supervisor.trace[-1] == "TERMINAL_WORKER_TERMINATED"
    assert launch.test_peer is not None
    launch.test_peer.close()


def test_completed_terminal_callback_worker_is_joined_on_close() -> None:
    called = threading.Event()

    def callback(_terminal: object) -> None:
        called.set()

    supervisor, launch = _supervisor(
        "normal", operation_timeout_s=2.0, terminal_callback=callback
    )
    supervisor.start(_request())
    supervisor.stop(TASK_ID, "CANCELLED")
    assert called.wait(timeout=1)
    supervisor.close()
    assert not supervisor.terminal_delivery_alive
    assert supervisor.trace[-1] == "TERMINAL_WORKER_TERMINATED"
    assert launch.test_peer is not None
    launch.test_peer.close()


@pytest.mark.parametrize("mode", ["gate-malformed", "gate-exit"])
def test_protocol_failure_and_unexpected_exit_reap_captured_exact_pid(mode: str) -> None:
    supervisor, launch = _supervisor(mode, operation_timeout_s=2.0)
    errors: list[BaseException] = []

    def ready() -> None:
        try:
            supervisor.ensure_ready()
        except BaseException as exc:  # noqa: BLE001 - expected child failure
            errors.append(exc)

    caller = threading.Thread(target=ready)
    caller.start()
    peer = _receive_gate(launch, b"HELLO")
    pid = supervisor.snapshot().pid
    assert pid is not None
    pidfd = os.pidfd_open(pid)
    peer.sendall(b"1")
    caller.join(timeout=2)
    assert not caller.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], SupervisorOperationError)
    _assert_pidfd_dead(pidfd)
    _assert_containment_trace(supervisor, leading="OPERATION_FAILED")
    supervisor.close()
    peer.close()


@pytest.mark.parametrize("state", ["IDLE", "RUNNING", "STARTING", "STOPPING"])
def test_close_is_bounded_and_exactly_reaps_from_owned_lifecycle_state(
    state: str,
) -> None:
    mode = {"STARTING": "block-start", "STOPPING": "block-stop"}.get(
        state, "normal"
    )
    supervisor, launch = _supervisor(mode)
    supervisor.ensure_ready()
    if state in {"RUNNING", "STOPPING"}:
        supervisor.start(_request())
    pid = supervisor.snapshot().pid
    assert pid is not None
    pidfd = os.pidfd_open(pid)
    operation_errors: list[BaseException] = []
    operation_thread: threading.Thread | None = None
    if state in {"STARTING", "STOPPING"}:

        def operation() -> None:
            try:
                if state == "STARTING":
                    supervisor.start(_request())
                else:
                    supervisor.stop(TASK_ID, "CANCELLED")
            except BaseException as exc:  # noqa: BLE001 - close races fail closed
                operation_errors.append(exc)

        operation_thread = threading.Thread(target=operation)
        operation_thread.start()
        _receive_gate(
            launch, b"START" if state == "STARTING" else b"ZERO_AND_STOP"
        )
        assert supervisor.snapshot().state.value == state

    close_errors: list[BaseException] = []

    def close() -> None:
        try:
            supervisor.close()
        except BaseException as exc:  # noqa: BLE001 - asserted below
            close_errors.append(exc)

    closer = threading.Thread(target=close)
    closer.start()
    closer.join(timeout=2)
    assert not closer.is_alive()
    assert close_errors == []
    if operation_thread is not None:
        operation_thread.join(timeout=2)
        assert not operation_thread.is_alive()
        assert len(operation_errors) == 1
    _assert_pidfd_dead(pidfd)
    assert supervisor.snapshot().pid is None
    assert supervisor.snapshot().slot_releasable
    assert supervisor.snapshot().state is SupervisorState.NO_CHILD
    assert supervisor.trace[-1] == "NO_CHILD"
    supervisor.close()
    for peer in launch.test_peers:
        peer.close()


def test_close_queued_during_fault_is_bounded_through_quarantine_and_reap() -> None:
    supervisor, launch = _supervisor("block-command")
    supervisor.ensure_ready()
    supervisor.start(_request())
    pid = supervisor.snapshot().pid
    assert pid is not None
    pidfd = os.pidfd_open(pid)
    operation_done = threading.Event()

    def blocked_command() -> None:
        try:
            supervisor.command(TASK_ID, {"vxMps": 0.0}, 500)
        except SupervisorOperationError:
            pass
        finally:
            operation_done.set()

    caller = threading.Thread(target=blocked_command)
    caller.start()
    _receive_gate(launch, b"COMMAND")
    closer = threading.Thread(target=supervisor.close)
    closer.start()
    assert operation_done.wait(timeout=2)
    closer.join(timeout=2)
    caller.join(timeout=2)
    assert not closer.is_alive() and not caller.is_alive()
    _assert_pidfd_dead(pidfd)
    _assert_containment_trace(supervisor)
    assert supervisor.snapshot().state is SupervisorState.NO_CHILD
    supervisor.close()
    for peer in launch.test_peers:
        peer.close()


def test_killed_parent_harness_leaves_no_orphan_exact_child() -> None:
    report_parent, report_child = socket.socketpair(
        socket.AF_UNIX, socket.SOCK_SEQPACKET
    )
    inherited = report_child.detach()
    harness = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).parent / "fakes" / "supervisor_parent_harness.py"),
            "--report-fd",
            str(inherited),
        ],
        pass_fds=(inherited,),
        close_fds=True,
        env=os.environ.copy(),
    )
    os.close(inherited)
    try:
        report_parent.settimeout(3)
        child_pid = int(report_parent.recv(32).decode("ascii"))
        child_pidfd = os.pidfd_open(child_pid)
        harness.send_signal(signal.SIGKILL)
        harness.wait(timeout=2)
        readable, _, _ = select.select([child_pidfd], [], [], 2)
        assert readable == [child_pidfd]
        os.close(child_pidfd)
    finally:
        if harness.poll() is None:
            harness.kill()
            harness.wait(timeout=2)
        report_parent.close()
