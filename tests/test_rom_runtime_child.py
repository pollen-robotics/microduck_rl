from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mjlab_microduck.rom.parent_death import verify_seqpacket_socket
from mjlab_microduck.rom.process_protocol import (
    CommandPayload,
    HelloPayload,
    LoadPayload,
    RuntimeMessage,
    RuntimeMessageKind,
    StartPayload,
    ZeroAndStopPayload,
    decode_packet,
    encode_packet,
)
from mjlab_microduck.rom.runtime import RuntimeEvidence, RuntimeHandle, RuntimeSample
from mjlab_microduck.rom.runtime_child import (
    RuntimeChildHost,
    _cleanup_evidence_is_truthful,
)
from mjlab_microduck.rom.runtime_identity import runtime_revision
from tests.fakes.fake_microduck_runtime import FakeMicroduckRuntime
from tests.fakes.fake_runtime_child import MODES


def test_runtime_child_requires_unix_seqpacket_descriptor() -> None:
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        with pytest.raises(ValueError, match="SOCK_SEQPACKET"):
            verify_seqpacket_socket(left.fileno())
    finally:
        left.close()
        right.close()


def test_runtime_child_has_bounded_run_interface() -> None:
    assert callable(RuntimeChildHost.run)


@pytest.mark.parametrize(
    "failure",
    [
        "EMERGENCY_STOP_FAILED",
        "RUNTIME_UNRESPONSIVE",
        "SAFE_STOP_FAILED",
        "WATCHDOG_FAILURE",
        "ZERO_COMMAND_FAILED",
        "UNKNOWN_FUTURE_FAILURE",
        None,
        True,
        7,
        0.5,
    ],
)
def test_cleanup_evidence_semantics_fail_closed(failure: object) -> None:
    assert not _cleanup_evidence_is_truthful(
        RuntimeEvidence(metrics={"safetyFailure": failure})
    )


def test_cleanup_evidence_without_safety_failure_is_truthful() -> None:
    assert _cleanup_evidence_is_truthful(RuntimeEvidence(metrics={"safeStop": True}))


def _exchange(peer: socket.socket, message: RuntimeMessage) -> RuntimeMessage:
    peer.sendall(encode_packet(message))
    return decode_packet(peer.recv(65_537))


def test_handshake_and_load_echo_exact_runtime_and_bundle_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    digest = "sha256:" + "a" * 64
    bundle = SimpleNamespace(bundleDigest=digest)
    monkeypatch.setattr(
        "mjlab_microduck.rom.runtime_child.load_qualified_bundle", lambda _root: bundle
    )
    host = RuntimeChildHost(
        child, runtime_factory=lambda _root, _bundle: FakeMicroduckRuntime()
    )
    thread = threading.Thread(target=host.run, daemon=True)
    thread.start()
    hello = RuntimeMessage(
        kind="HELLO",
        generation=4,
        operationSequence=1,
        taskId=None,
        payload=HelloPayload(runtimeRevision=runtime_revision()),
    )
    hello_reply = _exchange(parent, hello)
    assert hello_reply.kind is RuntimeMessageKind.ACK
    assert (hello_reply.generation, hello_reply.operationSequence) == (4, 1)
    load = RuntimeMessage(
        kind="LOAD",
        generation=4,
        operationSequence=2,
        taskId=None,
        payload=LoadPayload(bundleDigest=digest, bundleRoot="bundle"),
    )
    ready = _exchange(parent, load)
    assert ready.kind is RuntimeMessageKind.READY
    assert ready.payload.runtimeRevision == runtime_revision()
    assert ready.payload.bundleDigest == digest
    assert (ready.generation, ready.operationSequence) == (4, 2)
    parent.close()
    thread.join(timeout=1)


def test_wrong_runtime_revision_returns_bounded_error_then_exits() -> None:
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mjlab_microduck.rom.runtime_child",
                "--socket-fd",
                str(child.fileno()),
                "--expected-parent-pid",
                str(os.getpid()),
        ],
        pass_fds=(child.fileno(),),
    )
    child.close()
    request = RuntimeMessage(
        kind="HELLO",
        generation=1,
        operationSequence=1,
        taskId=None,
        payload=HelloPayload(runtimeRevision="wrong-revision"),
    )
    response = _exchange(parent, request)
    assert response.kind is RuntimeMessageKind.ERROR
    assert response.payload.code == "PROTOCOL_INCOMPATIBLE"
    assert response.payload.detail.retryable is False
    assert process.wait(timeout=5) == 0
    parent.close()


def test_fake_child_exposes_every_required_environment_free_mode() -> None:
    assert MODES == (
        "normal",
        "block-load",
        "block-start",
        "block-command",
        "block-status",
        "block-stop",
        "ignore-sigterm",
        "malformed-response",
        "late-response",
        "exit-before-ack",
        "exit-start",
        "terminal-event",
        "terminal-fallen",
        "terminal-overrun",
        "terminal-nonfinite",
        "terminal-runtime-exception",
        "event-before-status",
        "event-before-command",
        "duplicate-event",
        "stale-event",
        "malformed-event",
        "lease-null-cleanup-failure",
        "exit-after-ready",
        "exit-after-start-ack",
        "terminal-event-exit",
    )


@pytest.mark.parametrize(
    ("state", "reason", "outcome"),
    [("SUCCEEDED", "TASK_COMPLETE", "SUCCEEDED"), ("FAILED", "FALLEN", "FAILED")],
)
def test_discrete_sample_is_safely_stopped_then_emitted_with_metrics(
    state: str, reason: str, outcome: str
) -> None:
    host, runtime, parent, thread = _active_host()
    host._active_action_code = "STAND"
    host._bundle.actions[0].actionCode = "STAND"
    runtime.complete_next(
        state=state, metrics={"upright": state == "SUCCEEDED"}, stop_reason=reason
    )
    host._start_discrete_monitor()
    terminal = decode_packet(parent.recv(65_537))
    assert terminal.kind is RuntimeMessageKind.TERMINAL_EVENT
    assert terminal.operationSequence == 0
    assert terminal.payload.eventSequence == 1
    assert terminal.payload.terminal.outcome == outcome
    assert terminal.payload.terminal.evidence.metrics == {
        "safeStop": True,
        "upright": state == "SUCCEEDED",
    }
    assert runtime.safe_stop_calls[-1][1] == reason
    host._stop.set()
    host._put_message(None)
    thread.join(timeout=1)
    parent.close()


@pytest.mark.parametrize("reason", ["FALLEN", "CONTROL_LOOP_OVERRUN"])
def test_continuous_runtime_fault_is_safely_stopped_then_emitted_before_lease(
    reason: str,
) -> None:
    host, runtime, parent, thread = _active_host()
    with host._state_lock:
        host._lease_deadline = time.monotonic() + 10
    runtime.complete_next(state="FAILED", metrics={"fault": reason}, stop_reason=reason)

    host._start_runtime_monitor()

    terminal = decode_packet(parent.recv(65_537))
    assert terminal.kind is RuntimeMessageKind.TERMINAL_EVENT
    assert terminal.payload.eventSequence == 1
    assert terminal.payload.terminal.outcome == "FAILED"
    assert terminal.payload.terminal.evidence.stopReason == reason
    assert terminal.payload.terminal.evidence.metrics == {
        "fault": reason,
        "safeStop": True,
    }
    assert runtime.safe_stop_calls == [(RuntimeHandle(taskId="1" * 32), reason)]
    assert not host.sample_monitor_alive
    host._stop.set()
    host._put_message(None)
    thread.join(timeout=1)
    parent.close()


def test_continuous_running_samples_allow_commands_until_terminal_fault() -> None:
    host, runtime, parent, thread = _active_host()
    with host._state_lock:
        host._lease_deadline = time.monotonic() + 10
    host._start_runtime_monitor()
    assert runtime.sample_started.wait(timeout=1)

    command = RuntimeMessage(
        kind="COMMAND",
        generation=7,
        operationSequence=1,
        taskId="1" * 32,
        payload=CommandPayload(
            parameters={"vxMps": 0.1, "vyMps": 0.0, "yawRateRadps": 0.0},
            leaseMs=1000,
        ),
    )
    assert _exchange(parent, command).kind is RuntimeMessageKind.ACK
    assert runtime.command_calls[-1]["vxMps"] == 0.1

    runtime.complete_next(state="FAILED", metrics={}, stop_reason="FALLEN")
    terminal = decode_packet(parent.recv(65_537))
    assert terminal.kind is RuntimeMessageKind.TERMINAL_EVENT
    assert terminal.payload.terminal.evidence.stopReason == "FALLEN"
    host._stop.set()
    host._put_message(None)
    thread.join(timeout=1)
    parent.close()


def test_continuous_normal_sample_is_bounded_evidence_on_operator_stop() -> None:
    host, runtime, parent, thread = _active_host()
    # First prove the monitor has observed a normal sample without emitting high-rate IPC.
    runtime._samples.append(RuntimeSample(running=True, metrics={"tiltRad": 0.1}))
    host._start_runtime_monitor()
    assert runtime.sample_started.wait(timeout=1)
    deadline = time.monotonic() + 1
    while (
        host._latest_sample_metrics != {"tiltRad": 0.1} and time.monotonic() < deadline
    ):
        time.sleep(0.001)
    assert host._latest_sample_metrics == {"tiltRad": 0.1}
    stop = RuntimeMessage(
        kind="ZERO_AND_STOP",
        generation=7,
        operationSequence=1,
        taskId="1" * 32,
        payload=ZeroAndStopPayload(reason="OPERATOR_CANCELLED"),
    )
    terminal = _exchange(parent, stop)
    assert terminal.kind is RuntimeMessageKind.TERMINAL
    assert terminal.payload.evidence.metrics["tiltRad"] == 0.1
    host._stop.set()
    host._put_message(None)
    thread.join(timeout=1)
    parent.close()


def test_qualification_horizon_uses_exact_samples_and_terminal_event() -> None:
    host, runtime, parent, thread = _active_host()
    host._qualification_max_steps = 3

    host._start_runtime_monitor()

    terminal = decode_packet(parent.recv(65_537))
    assert terminal.kind is RuntimeMessageKind.TERMINAL_EVENT
    assert terminal.payload.terminal.outcome == "SUCCEEDED"
    assert terminal.payload.terminal.evidence.stopReason == "MAX_STEPS_REACHED"
    assert runtime.sample_call_count == 3
    assert runtime.safe_stop_calls == [
        (RuntimeHandle(taskId="1" * 32), "MAX_STEPS_REACHED")
    ]
    host._stop.set()
    host._put_message(None)
    thread.join(timeout=1)
    parent.close()


def test_continuous_qualification_waits_for_status_and_command_barrier() -> None:
    host, runtime, parent, thread = _active_host()
    host._qualification_max_steps = 3
    host._handle = None
    runtime.active_handle = None

    start = RuntimeMessage(
        kind="START",
        generation=8,
        operationSequence=1,
        taskId="2" * 32,
        payload=StartPayload(
            actionCode="WALK_VELOCITY",
            bundleDigest=host._bundle.bundleDigest,
            parameters={"vxMps": 0.1, "vyMps": 0.0, "yawRateRadps": 0.0},
            scenario={"terrain": "flat", "seed": 8},
            leaseMs=1000,
        ),
    )
    assert _exchange(parent, start).kind is RuntimeMessageKind.ACK
    assert not runtime.sample_started.wait(timeout=0.03)

    status = RuntimeMessage(
        kind="STATUS",
        generation=8,
        operationSequence=2,
        taskId="2" * 32,
        payload={},
    )
    assert _exchange(parent, status).kind is RuntimeMessageKind.STATUS
    assert not runtime.sample_started.wait(timeout=0.03)

    command = RuntimeMessage(
        kind="COMMAND",
        generation=8,
        operationSequence=3,
        taskId="2" * 32,
        payload=CommandPayload(
            parameters={"vxMps": 0.1, "vyMps": 0.0, "yawRateRadps": 0.0},
            leaseMs=1000,
        ),
    )
    assert _exchange(parent, command).kind is RuntimeMessageKind.ACK
    terminal = decode_packet(parent.recv(65_537))
    assert terminal.kind is RuntimeMessageKind.TERMINAL_EVENT
    assert runtime.sample_call_count == 3
    parent.close()
    thread.join(timeout=1)


def test_blocked_continuous_sample_retires_transport_on_normal_stop() -> None:
    host, runtime, parent, thread = _active_host()
    runtime.sample_release.clear()
    host._start_runtime_monitor()
    assert runtime.sample_started.wait(timeout=1)
    parent.sendall(
        encode_packet(
            RuntimeMessage(
                kind="ZERO_AND_STOP",
                generation=7,
                operationSequence=1,
                taskId="1" * 32,
                payload=ZeroAndStopPayload(reason="OPERATOR_CANCELLED"),
            )
        )
    )
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert host._cleanup_timed_out.is_set()
    assert runtime.emergency_stop_calls == ["RUNTIME_UNRESPONSIVE"]
    parent.settimeout(1)
    assert parent.recv(65_537) == b""
    runtime.sample_release.set()
    parent.close()


def test_continuous_fault_wins_stop_race_without_poisoning_same_child_reuse() -> None:
    host, runtime, parent, thread = _active_host()
    runtime.complete_next(state="FAILED", metrics={}, stop_reason="FALLEN")
    host._completion_claim.acquire()
    try:
        host._start_runtime_monitor()
        assert runtime.sample_started.wait(timeout=1)
        parent.sendall(
            encode_packet(
                RuntimeMessage(
                    kind="ZERO_AND_STOP",
                    generation=7,
                    operationSequence=1,
                    taskId="1" * 32,
                    payload=ZeroAndStopPayload(reason="OPERATOR_CANCELLED"),
                )
            )
        )
    finally:
        host._completion_claim.release()
    terminal = decode_packet(parent.recv(65_537))
    assert terminal.kind is RuntimeMessageKind.TERMINAL_EVENT

    start = RuntimeMessage(
        kind="START",
        generation=8,
        operationSequence=2,
        taskId="2" * 32,
        payload=StartPayload(
            actionCode="WALK_VELOCITY",
            bundleDigest="sha256:" + "a" * 64,
            parameters={"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
            scenario={"terrain": "flat", "seed": 8},
            leaseMs=1000,
        ),
    )
    assert _exchange(parent, start).kind is RuntimeMessageKind.ACK
    deadline = time.monotonic() + 1
    while not host.sample_monitor_alive and time.monotonic() < deadline:
        time.sleep(0.001)
    assert host.sample_monitor_alive
    assert thread.is_alive()
    parent.close()
    thread.join(timeout=1)


def test_continuous_stop_wins_queued_completion_and_reuses_same_child() -> None:
    host, runtime, parent, thread = _active_host()
    runtime.sample_release.clear()
    host._start_runtime_monitor()
    assert runtime.sample_started.wait(timeout=1)
    parent.sendall(
        encode_packet(
            RuntimeMessage(
                kind="ZERO_AND_STOP",
                generation=7,
                operationSequence=1,
                taskId="1" * 32,
                payload=ZeroAndStopPayload(reason="OPERATOR_CANCELLED"),
            )
        )
    )
    deadline = time.monotonic() + 1
    while not host._sample_stop.is_set() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert host._sample_stop.is_set()
    runtime.complete_next(state="FAILED", metrics={}, stop_reason="FALLEN")
    runtime.sample_release.set()
    terminal = decode_packet(parent.recv(65_537))
    assert terminal.kind is RuntimeMessageKind.TERMINAL
    assert terminal.payload.evidence.stopReason == "OPERATOR_CANCELLED"

    start = RuntimeMessage(
        kind="START",
        generation=8,
        operationSequence=2,
        taskId="2" * 32,
        payload=StartPayload(
            actionCode="WALK_VELOCITY",
            bundleDigest="sha256:" + "a" * 64,
            parameters={"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
            scenario={"terrain": "flat", "seed": 8},
            leaseMs=1000,
        ),
    )
    assert _exchange(parent, start).kind is RuntimeMessageKind.ACK
    assert thread.is_alive()
    parent.close()
    thread.join(timeout=1)


def test_discrete_safe_stop_failure_withholds_terminal_and_exits_transport() -> None:
    host, runtime, parent, thread = _active_host()
    host._active_action_code = "STAND"
    host._bundle.actions[0].actionCode = "STAND"
    runtime.safe_stop_error = RuntimeError("uncertain cleanup")
    runtime.complete_next(state="SUCCEEDED", metrics={"upright": True})
    host._start_discrete_monitor()
    thread.join(timeout=1)
    assert not thread.is_alive()
    parent.settimeout(1)
    assert parent.recv(65_537) == b""
    assert not host._safety_complete.is_set()
    parent.close()


def test_discrete_unbounded_safe_stop_evidence_withholds_terminal() -> None:
    host, runtime, parent, thread = _active_host()
    host._active_action_code = "STAND"
    host._bundle.actions[0].actionCode = "STAND"
    runtime.safe_stop_metrics = {f"metric-{index}": index for index in range(33)}
    runtime.complete_next(state="SUCCEEDED", metrics={})
    host._start_discrete_monitor()
    thread.join(timeout=1)
    assert not thread.is_alive()
    parent.settimeout(1)
    assert parent.recv(65_537) == b""
    parent.close()


def test_blocked_discrete_sample_forces_process_retirement_on_cleanup() -> None:
    host, runtime, parent, thread = _active_host()
    host._active_action_code = "STAND"
    host._bundle.actions[0].actionCode = "STAND"
    runtime.sample_release.clear()
    host._start_discrete_monitor()
    assert runtime.sample_started.wait(timeout=1)
    host._last_request = RuntimeMessage(
        kind="STATUS", generation=7, operationSequence=1, taskId="1" * 32, payload={}
    )
    host._request_safety("LEASE_EXPIRED")
    assert runtime.safe_stopped.wait(timeout=1)
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert host.sample_monitor_alive
    parent.settimeout(1)
    assert parent.recv(65_537) == b""
    with pytest.raises(BrokenPipeError):
        parent.sendall(
            encode_packet(
                RuntimeMessage(
                    kind="START",
                    generation=8,
                    operationSequence=2,
                    taskId="2" * 32,
                    payload=StartPayload(
                        actionCode="STAND",
                        bundleDigest="sha256:" + "a" * 64,
                        parameters={},
                        scenario={"terrain": "flat", "seed": 8},
                        leaseMs=1000,
                    ),
                )
            )
        )
    runtime.sample_release.set()
    parent.close()


def test_discrete_deadline_requests_safety_while_sample_is_blocked() -> None:
    now = [0.0]
    host, runtime, parent, thread = _active_host(clock=lambda: now[0])
    host._active_action_code = "STAND"
    host._bundle.actions[0].actionCode = "STAND"
    runtime.sample_release.clear()
    host._start_discrete_monitor()
    assert runtime.sample_started.wait(timeout=1)
    now[0] = 60.0
    assert host._safety_started.wait(timeout=1)
    thread.join(timeout=1)
    assert not thread.is_alive()
    parent.settimeout(1)
    assert parent.recv(65_537) == b""
    runtime.sample_release.set()
    parent.close()


def test_completion_cleanup_deadline_retires_transport_when_safe_stop_blocks() -> None:
    host, runtime, parent, thread = _active_host(fatal_cleanup_timeout_s=0.05)
    host._active_action_code = "STAND"
    host._bundle.actions[0].actionCode = "STAND"
    runtime.safe_stop_release.clear()
    runtime.complete_next(state="SUCCEEDED", metrics={})
    host._start_discrete_monitor()
    assert runtime.safe_stop_started.wait(timeout=1)
    assert host._safety_started.wait(timeout=1)
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert host._cleanup_timed_out.is_set()
    parent.settimeout(1)
    assert parent.recv(65_537) == b""
    runtime.safe_stop_release.set()
    parent.close()


def test_failed_completion_terminal_send_retires_transport(monkeypatch) -> None:
    host, runtime, parent, thread = _active_host()
    host._active_action_code = "STAND"
    host._bundle.actions[0].actionCode = "STAND"
    original_send = host._send

    def fail_terminal(message):
        if message.kind is RuntimeMessageKind.TERMINAL_EVENT:
            return False
        return original_send(message)

    monkeypatch.setattr(host, "_send", fail_terminal)
    runtime.complete_next(state="SUCCEEDED", metrics={})
    host._start_discrete_monitor()
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert host._cleanup_timed_out.is_set()
    assert host._task_id == "1" * 32
    parent.settimeout(1)
    assert parent.recv(65_537) == b""
    parent.close()


@pytest.mark.parametrize(
    "mode", ["blocked-sample", "blocked-completion-cleanup"]
)
def test_real_child_retires_blocked_fake_native_work_without_injected_lease(
    mode: str,
) -> None:
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).parent / "fakes" / "runtime_host_harness.py"),
            "--control-fd",
            str(child.fileno()),
            "--mode",
            mode,
        ],
        pass_fds=(child.fileno(),),
    )
    child.close()
    try:
        parent.settimeout(2)
        assert parent.recv(65_537) == b""
        assert process.wait(timeout=2) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        parent.close()


def _active_host(**host_kwargs) -> tuple[
    RuntimeChildHost, FakeMicroduckRuntime, socket.socket, threading.Thread
]:
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    runtime = FakeMicroduckRuntime()
    host = RuntimeChildHost(child, **host_kwargs)
    host._runtime = runtime
    host._bundle = SimpleNamespace(
        bundleVersion="1.0.0",
        bundleDigest="sha256:" + "a" * 64,
        model=SimpleNamespace(digest="sha256:" + "b" * 64),
        actions=[
            SimpleNamespace(
                actionCode="WALK_VELOCITY", policyRef="walk", availability="AVAILABLE"
            )
        ],
        policies=[SimpleNamespace(policyRef="walk", digest="sha256:" + "c" * 64)],
    )
    host._handle = RuntimeHandle(taskId="1" * 32)
    runtime.active_handle = host._handle
    host._generation = 7
    host._task_id = "1" * 32
    host._active_action_code = "WALK_VELOCITY"
    thread = threading.Thread(target=host.run, daemon=True)
    thread.start()
    return host, runtime, parent, thread


def test_lease_expiry_initiates_zero_stop_without_parent_watchdog() -> None:
    host, runtime, parent, thread = _active_host()
    # The real runtime rejects public commands after emergency_stop has already
    # zeroed and disabled actuators; safe_stop remains the cleanup proof.
    runtime.zero_command_error = RuntimeError("runtime is emergency-stopped")
    host._start_runtime_monitor()
    host._last_request = RuntimeMessage(
        kind="COMMAND",
        generation=7,
        operationSequence=1,
        taskId="1" * 32,
        payload=CommandPayload(
            parameters={"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
            leaseMs=100,
        ),
    )
    with host._state_lock:
        host._lease_deadline = time.monotonic() + 0.03
    assert runtime.emergency_stopped.wait(timeout=1)
    assert runtime.safe_stopped.wait(timeout=1)
    assert runtime.command_calls[-1] == {
        "vxMps": 0.0,
        "vyMps": 0.0,
        "yawRateRadps": 0.0,
    }
    assert runtime.safe_stop_calls[-1][1] == "LEASE_EXPIRED"
    parent.settimeout(1)
    terminal = decode_packet(parent.recv(65_537))
    assert terminal.kind is RuntimeMessageKind.TERMINAL_EVENT
    assert terminal.operationSequence == 0
    assert terminal.payload.eventSequence == 1
    assert terminal.payload.terminal.outcome == "TIMED_OUT"
    assert not host.sample_monitor_alive
    assert host._safety_complete.wait(timeout=1)
    parent.close()
    thread.join(timeout=1)


def test_safety_completion_waits_until_terminal_packet_is_published() -> None:
    """The main loop must not close transport while safety terminal send is paused."""
    host, runtime, parent, thread = _active_host()
    host._last_request = RuntimeMessage(
        kind="COMMAND",
        generation=7,
        operationSequence=1,
        taskId="1" * 32,
        payload=CommandPayload(
            parameters={"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
            leaseMs=100,
        ),
    )
    send_entered, send_release = threading.Event(), threading.Event()
    original_send = host._send

    def paused_terminal_send(message: RuntimeMessage) -> bool:
        if message.kind is RuntimeMessageKind.TERMINAL_EVENT:
            send_entered.set()
            assert send_release.wait(timeout=1)
        return original_send(message)

    host._send = paused_terminal_send  # type: ignore[method-assign]
    host._request_safety("LEASE_EXPIRED")
    assert runtime.safe_stopped.wait(timeout=1)
    assert send_entered.wait(timeout=1)
    time.sleep(0.02)
    assert not host._safety_complete.is_set()
    assert thread.is_alive()
    parent.setblocking(False)
    with pytest.raises(BlockingIOError):
        parent.recv(65_537)
    parent.setblocking(True)

    send_release.set()
    parent.settimeout(1)
    terminal = decode_packet(parent.recv(65_537))
    assert terminal.kind is RuntimeMessageKind.TERMINAL_EVENT
    assert host._safety_complete.wait(timeout=1)
    thread.join(timeout=1)
    assert not thread.is_alive()
    parent.close()


@pytest.mark.parametrize(
    "failure",
    [
        "exception",
        "invalid-evidence",
        "runtime-unresponsive",
        "unknown-safety-failure",
        "null-safety-failure",
        "bool-safety-failure",
        "numeric-safety-failure",
    ],
)
def test_lease_expiry_safe_stop_failure_withholds_event_and_retires_transport(
    failure: str,
) -> None:
    host, runtime, parent, thread = _active_host()
    host._start_runtime_monitor()
    if failure == "exception":
        runtime.safe_stop_error = RuntimeError("uncertain cleanup")
    elif failure == "invalid-evidence":
        runtime.safe_stop_metrics = {f"metric-{index}": index for index in range(33)}
    elif failure == "runtime-unresponsive":
        runtime.safe_stop_metrics = {"safetyFailure": "RUNTIME_UNRESPONSIVE"}
    elif failure == "unknown-safety-failure":
        runtime.safe_stop_metrics = {"safetyFailure": "NEW_UNKNOWN_FAILURE"}
    elif failure == "null-safety-failure":
        runtime.safe_stop_metrics = {"safetyFailure": None}
    elif failure == "bool-safety-failure":
        runtime.safe_stop_metrics = {"safetyFailure": True}
    else:
        runtime.safe_stop_metrics = {"safetyFailure": 7}
    host._last_request = RuntimeMessage(
        kind="COMMAND",
        generation=7,
        operationSequence=1,
        taskId="1" * 32,
        payload=CommandPayload(
            parameters={"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
            leaseMs=100,
        ),
    )
    with host._state_lock:
        host._lease_deadline = time.monotonic() + 0.03
    assert runtime.safe_stopped.wait(timeout=1)
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert host._cleanup_timed_out.is_set()
    parent.settimeout(1)
    assert parent.recv(65_537) == b""
    parent.close()


def test_continuous_completion_unresponsive_evidence_withholds_event() -> None:
    host, runtime, parent, thread = _active_host()
    runtime.safe_stop_metrics = {"safetyFailure": "RUNTIME_UNRESPONSIVE"}
    runtime.complete_next(state="FAILED", metrics={}, stop_reason="FALLEN")
    host._start_runtime_monitor()
    assert runtime.safe_stopped.wait(timeout=1)
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert host._cleanup_timed_out.is_set()
    parent.settimeout(1)
    assert parent.recv(65_537) == b""
    parent.close()


def test_normal_stop_unresponsive_evidence_withholds_correlated_terminal() -> None:
    host, runtime, parent, thread = _active_host()
    runtime.safe_stop_metrics = {"safetyFailure": "RUNTIME_UNRESPONSIVE"}
    host._start_runtime_monitor()
    parent.sendall(
        encode_packet(
            RuntimeMessage(
                kind="ZERO_AND_STOP",
                generation=7,
                operationSequence=1,
                taskId="1" * 32,
                payload=ZeroAndStopPayload(reason="OPERATOR_CANCELLED"),
            )
        )
    )
    assert runtime.safe_stopped.wait(timeout=1)
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert host._cleanup_timed_out.is_set()
    parent.settimeout(1)
    assert parent.recv(65_537) == b""
    parent.close()


def test_blocked_continuous_monitor_on_lease_expiry_retires_without_event() -> None:
    host, runtime, parent, thread = _active_host()
    runtime.sample_release.clear()
    host._start_runtime_monitor()
    assert runtime.sample_started.wait(timeout=1)
    host._last_request = RuntimeMessage(
        kind="COMMAND",
        generation=7,
        operationSequence=1,
        taskId="1" * 32,
        payload=CommandPayload(
            parameters={"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
            leaseMs=100,
        ),
    )
    with host._state_lock:
        host._lease_deadline = time.monotonic() + 0.03
    assert runtime.safe_stopped.wait(timeout=1)
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert host._cleanup_timed_out.is_set()
    parent.settimeout(1)
    assert parent.recv(65_537) == b""
    assert len(runtime.safe_stop_calls) == 1
    runtime.sample_release.set()
    parent.close()


def test_parent_eof_initiates_local_zero_stop() -> None:
    _host, runtime, parent, thread = _active_host()
    parent.close()
    assert runtime.emergency_stopped.wait(timeout=1)
    assert runtime.safe_stopped.wait(timeout=1)
    assert runtime.safe_stop_calls[-1][1] == "PARENT_EOF"
    thread.join(timeout=1)


def test_deadman_initiates_emergency_zero_while_command_call_is_blocked() -> None:
    host, runtime, parent, thread = _active_host()
    runtime.command_release.clear()
    command = RuntimeMessage(
        kind="COMMAND",
        generation=7,
        operationSequence=1,
        taskId="1" * 32,
        payload=CommandPayload(
            parameters={"vxMps": 0.1, "vyMps": 0.0, "yawRateRadps": 0.0},
            leaseMs=100,
        ),
    )
    parent.sendall(encode_packet(command))
    assert runtime.command_started.wait(timeout=1)
    assert runtime.emergency_stopped.wait(timeout=1)
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert not host._safety_complete.is_set()
    parent.settimeout(0.1)
    with pytest.raises((TimeoutError, ConnectionError, OSError)):
        packet = parent.recv(65_537)
        if not packet:
            raise ConnectionError
        assert decode_packet(packet).kind is not RuntimeMessageKind.TERMINAL
    runtime.command_release.set()
    parent.close()


def test_normal_stop_returns_child_to_idle_for_next_generation() -> None:
    host, runtime, parent, thread = _active_host()
    stop = RuntimeMessage(
        kind="ZERO_AND_STOP",
        generation=7,
        operationSequence=1,
        taskId="1" * 32,
        payload=ZeroAndStopPayload(reason="OPERATOR_CANCELLED"),
    )
    terminal = _exchange(parent, stop)
    assert terminal.kind is RuntimeMessageKind.TERMINAL
    assert terminal.payload.outcome == "CANCELLED"
    assert thread.is_alive()
    start_request = RuntimeMessage(
        kind="START",
        generation=8,
        operationSequence=2,
        taskId="2" * 32,
        payload=StartPayload(
            actionCode="WALK_VELOCITY",
            bundleDigest=host._bundle.bundleDigest,
            parameters={"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
            scenario={"terrain": "flat", "seed": 2},
            leaseMs=500,
        ),
    )
    assert _exchange(parent, start_request).kind is RuntimeMessageKind.ACK
    assert runtime.active_handle == RuntimeHandle(taskId="2" * 32)
    status_request = RuntimeMessage(
        kind="STATUS", generation=8, operationSequence=3, taskId="2" * 32, payload={}
    )
    assert _exchange(parent, status_request).kind is RuntimeMessageKind.STATUS
    parent.close()
    thread.join(timeout=1)


@pytest.mark.parametrize(
    ("reason", "outcome"),
    [
        ("OPERATOR_CANCELLED", "CANCELLED"),
        ("LEASE_EXPIRED", "TIMED_OUT"),
        ("RUNTIME_FAILED", "FAILED"),
        ("PROTOCOL_ERROR", "FAILED"),
        ("PARENT_DEATH", "FAILED"),
    ],
)
def test_terminal_outcome_is_truthfully_mapped(reason: str, outcome: str) -> None:
    host, _runtime, parent, _thread = _active_host()
    request = RuntimeMessage(
        kind="ZERO_AND_STOP",
        generation=7,
        operationSequence=1,
        taskId="1" * 32,
        payload=ZeroAndStopPayload(reason="OPERATOR_CANCELLED"),
    )
    evidence = RuntimeEvidence(stopReason=reason)
    assert host._terminal(request, reason, evidence).payload.outcome == outcome
    parent.close()


def test_parent_death_request_wakes_idle_main_loop_and_exits() -> None:
    host, runtime, parent, thread = _active_host()
    host._request_safety("PARENT_DEATH")
    assert runtime.emergency_stopped.wait(timeout=1)
    thread.join(timeout=1)
    assert not thread.is_alive()
    parent.close()


def test_close_unrelated_fds_preserves_only_explicit_descriptors() -> None:
    extra_a, extra_b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    script = (
        "import os; from mjlab_microduck.rom.parent_death import close_unrelated_fds; "
        f"close_unrelated_fds({{0,1,2}}); "
        f"\ntry: os.fstat({extra_a.fileno()})\nexcept OSError: print('closed')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        pass_fds=(extra_a.fileno(),),
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "closed"
    extra_a.close()
    extra_b.close()


def test_sigterm_wakes_child_and_extra_inherited_fd_is_closed() -> None:
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    extra_a, extra_b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mjlab_microduck.rom.runtime_child",
                "--socket-fd",
                str(child.fileno()),
                "--expected-parent-pid",
                str(os.getpid()),
        ],
        pass_fds=(child.fileno(), extra_a.fileno()),
    )
    child.close()
    extra_fd_path = Path(f"/proc/{process.pid}/fd/{extra_a.fileno()}")
    deadline = time.monotonic() + 2
    while extra_fd_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not extra_fd_path.exists()
    hello = RuntimeMessage(
        kind="HELLO",
        generation=1,
        operationSequence=1,
        taskId=None,
        payload=HelloPayload(runtimeRevision=runtime_revision()),
    )
    assert _exchange(parent, hello).kind is RuntimeMessageKind.ACK
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=3) == 0
    parent.close()
    extra_a.close()
    extra_b.close()


def test_environment_filtering_removes_unrelated_platform_configuration() -> None:
    script = (
        "import os; from mjlab_microduck.rom.runtime_child import clear_runtime_environment; "
        "clear_runtime_environment(); "
        "print('MICRODUCK_ROM_BEARER_TOKEN' in os.environ, "
        "'MICRODUCK_ROM_BEARER_TOKEN_FILE' in os.environ)"
    )
    environment = os.environ.copy()
    environment["MICRODUCK_ROM_BEARER_TOKEN"] = "must-not-survive"
    environment["MICRODUCK_ROM_BEARER_TOKEN_FILE"] = "/must/not/survive"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "False False"


def test_blocked_start_cannot_defeat_local_emergency_zero() -> None:
    host, runtime, parent, thread = _active_host()
    host._handle = None
    runtime.active_handle = None
    host._bundle.bundleVersion = "1.0.0"
    host._bundle.actions[0].availability = "AVAILABLE"
    runtime.start_release.clear()
    request = RuntimeMessage(
        kind="START",
        generation=8,
        operationSequence=1,
        taskId="2" * 32,
        payload=StartPayload(
            actionCode="WALK_VELOCITY",
            bundleDigest=host._bundle.bundleDigest,
            parameters={"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
            scenario={"terrain": "flat", "seed": 1},
            leaseMs=100,
        ),
    )
    parent.sendall(encode_packet(request))
    assert runtime.started.wait(timeout=1)
    assert runtime.emergency_stopped.wait(timeout=1)
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert not host._safety_complete.is_set()
    assert host._cleanup_timed_out.is_set()
    parent.settimeout(0.1)
    with pytest.raises((TimeoutError, ConnectionError, OSError)):
        packet = parent.recv(65_537)
        if not packet:
            raise ConnectionError
        assert decode_packet(packet).kind is not RuntimeMessageKind.TERMINAL
    runtime.start_release.set()
    parent.close()


@pytest.mark.parametrize("operation", ["status", "stop"])
def test_blocked_status_or_stop_cannot_defeat_local_emergency_zero(
    operation: str,
) -> None:
    host, runtime, parent, thread = _active_host()
    with host._state_lock:
        host._lease_deadline = time.monotonic() + 0.1
    if operation == "status":
        runtime.status_release.clear()
        request = RuntimeMessage(
            kind="STATUS",
            generation=7,
            operationSequence=1,
            taskId="1" * 32,
            payload={},
        )
        started = runtime.status_started
        release = runtime.status_release
    else:
        runtime.command_release.clear()
        request = RuntimeMessage(
            kind="ZERO_AND_STOP",
            generation=7,
            operationSequence=1,
            taskId="1" * 32,
            payload=ZeroAndStopPayload(reason="OPERATOR_CANCELLED"),
        )
        started = runtime.command_started
        release = runtime.command_release
    parent.sendall(encode_packet(request))
    assert started.wait(timeout=1)
    assert runtime.emergency_stopped.wait(timeout=1)
    thread.join(timeout=1)
    assert not thread.is_alive()
    if operation == "status":
        terminal = decode_packet(parent.recv(65_537))
        assert terminal.kind is RuntimeMessageKind.TERMINAL_EVENT
        assert host._safety_complete.is_set()
    else:
        assert not host._safety_complete.is_set()
        assert host._cleanup_timed_out.is_set()
        parent.settimeout(0.1)
        with pytest.raises((TimeoutError, ConnectionError, OSError)):
            packet = parent.recv(65_537)
            if not packet:
                raise ConnectionError
            assert decode_packet(packet).kind is not RuntimeMessageKind.TERMINAL
    release.set()
    parent.close()
