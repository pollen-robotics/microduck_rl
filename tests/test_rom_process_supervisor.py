from __future__ import annotations

import os
import select
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
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
    SupervisorSnapshot,
    SupervisorTaskTerminalized,
    SupervisorUnavailable,
    _Intent,
    _TerminalDelivery,
)
from mjlab_microduck.rom.supervisor_state import SupervisorState


def test_supervisor_module_exposes_process_owner() -> None:
    assert RuntimeProcessSupervisor is not None


def test_production_child_launch_uses_safe_module_path_and_null_stdio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Writable service state and inherited streams must not reach the native child."""
    monkeypatch.setenv("PYTHONPATH", "/state/attacker-controlled")
    supervisor = RuntimeProcessSupervisor(bundle_root="/bundle", bundle_digest=DIGEST)
    try:
        launch = supervisor._default_launch(17)
        assert launch.argv[:3] == (sys.executable, "-P", "-m")
        assert launch.env is not None and "PYTHONPATH" not in launch.env
        assert launch.stdin == subprocess.DEVNULL
        assert launch.stdout == subprocess.DEVNULL
        assert launch.stderr == subprocess.DEVNULL
    finally:
        supervisor.close()


def test_qualification_launch_uses_the_same_runtime_child_with_bounded_mode() -> None:
    supervisor = RuntimeProcessSupervisor(
        bundle_root="/candidate",
        bundle_digest=DIGEST,
        qualification_max_steps=100,
    )
    try:
        launch = supervisor._default_launch(17)
        assert launch.argv[-2:] == ("--qualification-max-steps", "100")
        assert "mjlab_microduck.rom.runtime_child" in launch.argv
        assert "qualification_worker" not in " ".join(launch.argv)
    finally:
        supervisor.close()


@pytest.mark.parametrize(("mode", "operation"), [("block-start", "start"), ("block-stop", "stop")])
def test_dispatch_callback_proves_request_was_sent_before_blocked_reply(
    mode: str, operation: str
) -> None:
    supervisor, _launch = _supervisor(mode, operation_timeout_s=2.0)
    supervisor.ensure_ready()
    if operation == "stop":
        supervisor.start(_request())
    dispatched = threading.Event()
    result: list[BaseException] = []

    def invoke() -> None:
        try:
            if operation == "start":
                supervisor.start(_request(), register_dispatch=dispatched.set)
            else:
                supervisor.stop(TASK_ID, "CANCELLED", dispatched.set)
        except BaseException as exc:  # noqa: BLE001 - close deliberately severs IPC
            result.append(exc)

    worker = threading.Thread(target=invoke, daemon=True)
    worker.start()
    assert dispatched.wait(timeout=1)
    assert worker.is_alive()
    supervisor.close()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert result


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


def _gate_next_unsolicited_poll(
    supervisor: RuntimeProcessSupervisor, *, drain: bool = True
) -> threading.Event:
    """Hold the owner before poll so the exact child can become a confirmed zombie."""
    entered = threading.Event()
    release = threading.Event()
    original = supervisor._poll_unsolicited

    def gated_poll() -> None:
        entered.set()
        assert release.wait(timeout=2)
        supervisor._poll_unsolicited = original  # type: ignore[method-assign]
        if drain:
            original()

    supervisor._poll_unsolicited = gated_poll  # type: ignore[method-assign]
    assert entered.wait(timeout=2)
    return release


def _capture_exception(
    outcome: dict[str, BaseException], key: str, operation: Callable[[], object]
) -> None:
    try:
        operation()
    except BaseException as exc:  # noqa: BLE001 - test records exact caller outcome.
        outcome[key] = exc


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


def test_start_result_carries_exact_acknowledged_generation_and_task_identity() -> None:
    supervisor, _launch = _supervisor()
    try:
        supervisor.ensure_ready()

        result = supervisor.start(_request())

        assert result.generation == 1
        assert result.task_id == TASK_ID
        assert result.acknowledgement.acknowledgedKind.value == "START"
        supervisor.stop(TASK_ID, "CANCELLED")
    finally:
        supervisor.close()


def test_start_registration_ambiguity_contains_exact_acknowledged_child() -> None:
    supervisor, _launch = _supervisor()
    supervisor.ensure_ready()
    pid = supervisor.snapshot().pid
    assert pid is not None
    pidfd = os.pidfd_open(pid)

    def reject_registration(_result) -> None:
        raise RuntimeError("ambiguous service identity")

    try:
        with pytest.raises(
            SupervisorOperationError,
            match="START acknowledgement registration failed closed",
        ):
            supervisor.start(_request(), reject_registration)

        _assert_pidfd_dead(pidfd)
        assert "START_REGISTRATION_FAILED" in supervisor.trace
        assert "BEST_EFFORT_STOP_ACK" in supervisor.trace
        assert "CHILD_REAPED" in supervisor.trace
        assert supervisor.snapshot().slot_releasable is True
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
        deadline = time.monotonic() + 1
        while not supervisor.snapshot().slot_releasable and time.monotonic() < deadline:
            time.sleep(0.01)
        snapshot = supervisor.snapshot()
        assert snapshot.cached_terminal is not None
        assert snapshot.cached_terminal.outcome == "SUCCEEDED"
        assert snapshot.state is SupervisorState.IDLE
        assert snapshot.slot_releasable is True
    finally:
        supervisor.close()


def test_owner_autonomously_reaps_child_exit_after_start_ack_without_terminal() -> None:
    """Removing idle exit polling would leave the exact acknowledged task RUNNING."""
    supervisor, launch = _supervisor("exit-after-start-ack")
    try:
        supervisor.ensure_ready()
        supervisor.start(_request())
        peer = _receive_gate(launch, b"STARTED")
        pid = supervisor.snapshot().pid
        assert pid is not None
        pidfd = os.pidfd_open(pid)
        release_poll = _gate_next_unsolicited_poll(supervisor)
        peer.sendall(b"EXIT")
        assert select.select([pidfd], [], [], 2)[0] == [pidfd]
        release_poll.set()

        deadline = time.monotonic() + 2
        while supervisor.snapshot().pid is not None and time.monotonic() < deadline:
            time.sleep(0.005)
        _assert_pidfd_dead(pidfd)
        assert supervisor.snapshot().state is SupervisorState.NO_CHILD
        assert supervisor.snapshot().slot_releasable is True
        assert "CHILD_REAPED" in supervisor.trace
    finally:
        supervisor.close()


def test_terminal_packet_is_consumed_before_immediately_exited_child_is_reaped() -> None:
    """Checking poll first would discard a truthful queued terminal packet."""
    delivered: list[TerminalPayload] = []
    supervisor, launch = _supervisor(
        "terminal-event-exit",
        terminal_callback=lambda delivery: delivered.append(delivery.terminal),
    )
    try:
        supervisor.ensure_ready()
        supervisor.start(_request())
        peer = _receive_gate(launch, b"STARTED")
        pid = supervisor.snapshot().pid
        assert pid is not None
        pidfd = os.pidfd_open(pid)
        release_poll = _gate_next_unsolicited_poll(supervisor)
        peer.sendall(b"EMIT")
        assert select.select([pidfd], [], [], 2)[0] == [pidfd]
        os.close(pidfd)
        release_poll.set()

        deadline = time.monotonic() + 2
        while (
            (not delivered or supervisor.snapshot().pid is not None)
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        assert [item.outcome for item in delivered] == ["SUCCEEDED"]
        assert supervisor.snapshot().cached_terminal is not None
        assert supervisor.snapshot().cached_terminal.outcome == "SUCCEEDED"
        assert supervisor.snapshot().state is SupervisorState.NO_CHILD
        assert supervisor.snapshot().slot_releasable is True
    finally:
        supervisor.close()


def test_prequeued_terminal_from_exited_child_wins_over_next_public_operation() -> None:
    """A send to EOF must not discard the terminal already queued by the kernel."""
    delivered: list[TerminalPayload] = []
    supervisor, launch = _supervisor(
        "terminal-event-exit",
        terminal_callback=lambda delivery: delivered.append(delivery.terminal),
    )
    outcome: dict[str, BaseException] = {}
    try:
        supervisor.ensure_ready()
        supervisor.start(_request())
        peer = _receive_gate(launch, b"STARTED")
        release_poll = _gate_next_unsolicited_poll(supervisor, drain=False)
        pid = supervisor.snapshot().pid
        assert pid is not None
        pidfd = os.pidfd_open(pid)
        peer.sendall(b"EMIT")
        assert select.select([pidfd], [], [], 2)[0] == [pidfd]
        os.close(pidfd)

        caller = threading.Thread(
            target=lambda: _capture_exception(
                outcome, "status", lambda: supervisor.status(TASK_ID)
            ),
            daemon=True,
        )
        caller.start()
        release_poll.set()
        caller.join(timeout=2)
        assert not caller.is_alive()
        assert isinstance(outcome.get("status"), SupervisorTaskTerminalized)
        deadline = time.monotonic() + 2
        while (
            (not delivered or supervisor.snapshot().pid is not None)
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        assert [terminal.outcome for terminal in delivered] == ["SUCCEEDED"]
        assert supervisor.snapshot().state is SupervisorState.NO_CHILD
    finally:
        supervisor.close()


def test_lease_null_cleanup_failure_eof_is_reaped_before_slot_release() -> None:
    delivered = threading.Event()
    supervisor, launch = _supervisor(
        "lease-null-cleanup-failure",
        terminal_callback=lambda _payload: delivered.set(),
    )
    try:
        supervisor.ensure_ready()
        supervisor.start(_request())
        peer = _receive_gate(launch, b"STARTED")
        pid = supervisor.snapshot().pid
        assert pid is not None
        pidfd = os.pidfd_open(pid)
        peer.sendall(b"EMIT")
        deadline = time.monotonic() + 2
        while supervisor.snapshot().pid is not None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not delivered.is_set()
        assert "QUARANTINED" in supervisor.trace
        assert "CHILD_REAPED" in supervisor.trace
        _assert_pidfd_dead(pidfd)
        assert supervisor.snapshot().slot_releasable is True
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
        assert snapshot.slot_releasable is False
        assert snapshot.terminal_delivery_outstanding is True
    finally:
        supervisor.close()


def test_terminal_callback_saturation_blocks_next_task_until_delivery_ack() -> None:
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
        dummy = TerminalPayload(
            outcome="CANCELLED",
            evidence=TaskEvidence(
                bundleDigest=DIGEST, policyDigest="sha256:" + "b" * 64,
                modelDigest="sha256:" + "c" * 64, stopReason="CANCELLED",
            ),
        )
        supervisor._terminal_queue.put_nowait(_TerminalDelivery(0, dummy))
        assert blocked.wait(timeout=1)
        supervisor._terminal_queue.put_nowait(_TerminalDelivery(0, dummy))
        peer = _receive_gate(launch, b"STARTED")
        peer.sendall(b"EMIT")
        deadline = time.monotonic() + 1
        while supervisor.snapshot().state is not SupervisorState.IDLE and time.monotonic() < deadline:
            time.sleep(0.01)
        assert supervisor.snapshot().state is SupervisorState.IDLE
        assert supervisor.snapshot().cached_terminal is not None
        assert supervisor.snapshot().terminal_delivery_outstanding is True
        assert supervisor.snapshot().slot_releasable is False
        assert supervisor.readiness() is False
        with pytest.raises(SupervisorUnavailable):
            supervisor.start(_request())
        assert supervisor.snapshot().pid is not None
        release.set()
        deadline = time.monotonic() + 2
        while not supervisor.snapshot().slot_releasable and time.monotonic() < deadline:
            time.sleep(0.01)
        assert supervisor.snapshot().terminal_delivery_outstanding is False
        assert supervisor.snapshot().slot_releasable is True
        assert supervisor.readiness() is True
    finally:
        release.set()
        supervisor.close()


def test_throw_once_terminal_callback_retries_identical_payload_before_reuse() -> None:
    received: list[TerminalPayload] = []
    first_failed = threading.Event()

    def callback(payload: TerminalPayload) -> None:
        received.append(payload)
        if len(received) == 1:
            first_failed.set()
            raise RuntimeError("transient")

    supervisor, launch = _supervisor(
        "terminal-event", terminal_callback=callback, terminal_retry_delay_s=0.05
    )
    try:
        supervisor.ensure_ready()
        supervisor.start(_request())
        peer = _receive_gate(launch, b"STARTED")
        peer.sendall(b"EMIT")
        assert first_failed.wait(timeout=1)
        assert supervisor.readiness() is False
        with pytest.raises(SupervisorUnavailable):
            supervisor.start(_request())
        deadline = time.monotonic() + 2
        while not supervisor.readiness() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(received) == 2
        assert received[0] == received[1] == supervisor.snapshot().cached_terminal
        assert supervisor.snapshot().terminal_delivery_outstanding is False
        assert supervisor.readiness() is True
    finally:
        supervisor.close()


def test_permanent_terminal_callback_failure_remains_bounded_and_fail_closed() -> None:
    received: list[TerminalPayload] = []

    def callback(payload: TerminalPayload) -> None:
        received.append(payload)
        raise RuntimeError("permanent")

    supervisor, launch = _supervisor(
        "terminal-event",
        terminal_callback=callback,
        terminal_retry_delay_s=0.02,
        terminal_retry_limit=2,
    )
    try:
        supervisor.ensure_ready()
        supervisor.start(_request())
        peer = _receive_gate(launch, b"STARTED")
        peer.sendall(b"EMIT")
        deadline = time.monotonic() + 2
        while "TERMINAL_DELIVERY_PERMANENT_FAILURE" not in supervisor.trace and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(received) == 2
        assert received[0] == received[1] == supervisor.snapshot().cached_terminal
        assert supervisor.snapshot().terminal_delivery_outstanding is True
        assert supervisor.readiness() is False
        with pytest.raises(SupervisorUnavailable):
            supervisor.start(_request())
    finally:
        supervisor.close()


def test_terminal_retry_has_priority_under_continuously_populated_intent_queue() -> None:
    first_failed = threading.Event()
    redelivered = threading.Event()
    dispatch_count = 0
    count_lock = threading.Lock()

    def callback(_payload: TerminalPayload) -> None:
        if not first_failed.is_set():
            first_failed.set()
            raise RuntimeError("transient")
        redelivered.set()

    supervisor, launch = _supervisor(
        "terminal-event",
        queue_size=64,
        terminal_callback=callback,
        terminal_retry_delay_s=0.03,
    )
    try:
        supervisor.ensure_ready()
        supervisor.start(_request())
        peer = _receive_gate(launch, b"STARTED")
        peer.sendall(b"EMIT")
        assert first_failed.wait(timeout=1)
        original = supervisor._dispatch

        def counted(intent: _Intent) -> object:
            nonlocal dispatch_count
            with count_lock:
                dispatch_count += 1
            time.sleep(0.005)
            return original(intent)

        supervisor._dispatch = counted  # type: ignore[method-assign]
        queued = [_Intent(kind="ready") for _ in range(40)]
        for intent in queued:
            supervisor._queue.put_nowait(intent)
        assert redelivered.wait(timeout=0.25)
        with count_lock:
            assert dispatch_count <= 10
        deadline = time.monotonic() + 2
        while not all(intent.done.is_set() for intent in queued) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert all(intent.done.is_set() for intent in queued)
        assert all(
            isinstance(intent.error, SupervisorUnavailable)
            or isinstance(intent.result, SupervisorSnapshot)
            for intent in queued
        )
        assert supervisor.readiness() is True
    finally:
        supervisor.close()


def test_full_intent_queue_concurrent_close_cannot_strand_callback_ack() -> None:
    callback_entered = threading.Event()
    callback_release = threading.Event()
    owner_blocked = threading.Event()
    owner_release = threading.Event()

    def callback(_payload: TerminalPayload) -> None:
        callback_entered.set()
        callback_release.wait(timeout=2)

    supervisor, launch = _supervisor(
        "normal", queue_size=1, operation_timeout_s=2.0, terminal_callback=callback
    )
    supervisor.start(_request())
    supervisor.stop(TASK_ID, "CANCELLED")
    assert callback_entered.wait(timeout=1)
    original = supervisor._complete_terminal_delivery

    def gated(sequence: int, success: bool) -> None:
        if sequence == 999:
            owner_blocked.set()
            owner_release.wait(timeout=2)
            return
        original(sequence, success)

    supervisor._complete_terminal_delivery = gated  # type: ignore[method-assign]
    blocker = _Intent(kind="delivery", args=(999, True))
    supervisor._queue.put_nowait(blocker)
    assert owner_blocked.wait(timeout=1)
    queued = _Intent(kind="ready")
    supervisor._queue.put_nowait(queued)
    errors: list[BaseException] = []

    def close() -> None:
        try:
            supervisor.close()
        except BaseException as exc:  # noqa: BLE001 - test captures close failure
            errors.append(exc)

    closer = threading.Thread(target=close)
    closer.start()
    callback_release.set()
    owner_release.set()
    closer.join(timeout=4)
    assert not closer.is_alive()
    assert errors == []
    assert queued.done.wait(timeout=1)
    assert isinstance(queued.error, SupervisorUnavailable)
    assert not supervisor.terminal_delivery_alive
    assert supervisor.snapshot().pid is None
    assert launch.test_peer is not None
    launch.test_peer.close()


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
    if operation != "ready":
        assert supervisor.snapshot().quarantine_reason == "OPERATION_TIMEOUT"
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
