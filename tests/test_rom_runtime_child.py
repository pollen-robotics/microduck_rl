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
from mjlab_microduck.rom.runtime import RuntimeEvidence, RuntimeHandle
from mjlab_microduck.rom.runtime_child import RuntimeChildHost
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
    )


def _active_host() -> tuple[RuntimeChildHost, FakeMicroduckRuntime, socket.socket, threading.Thread]:
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    runtime = FakeMicroduckRuntime()
    host = RuntimeChildHost(child)
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
    parent.close()
    thread.join(timeout=1)


def test_parent_eof_initiates_local_zero_stop() -> None:
    _host, runtime, parent, thread = _active_host()
    parent.close()
    assert runtime.emergency_stopped.wait(timeout=1)
    assert runtime.safe_stopped.wait(timeout=1)
    assert runtime.safe_stop_calls[-1][1] == "PARENT_EOF"
    thread.join(timeout=1)


def test_deadman_initiates_emergency_zero_while_command_call_is_blocked() -> None:
    _host, runtime, parent, thread = _active_host()
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
    runtime.command_release.set()
    parent.close()
    thread.join(timeout=1)


def test_normal_stop_returns_child_to_idle_for_next_generation() -> None:
    host, runtime, parent, thread = _active_host()
    stop = RuntimeMessage(
        kind="ZERO_AND_STOP", generation=7, operationSequence=1, taskId="1" * 32,
        payload=ZeroAndStopPayload(reason="OPERATOR_CANCELLED"),
    )
    terminal = _exchange(parent, stop)
    assert terminal.kind is RuntimeMessageKind.TERMINAL
    assert terminal.payload.outcome == "CANCELLED"
    assert thread.is_alive()
    start_request = RuntimeMessage(
        kind="START", generation=8, operationSequence=2, taskId="2" * 32,
        payload=StartPayload(
            actionCode="WALK_VELOCITY", bundleDigest=host._bundle.bundleDigest,
            parameters={"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
            scenario={"terrain": "flat", "seed": 2}, leaseMs=500,
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
        kind="ZERO_AND_STOP", generation=7, operationSequence=1, taskId="1" * 32,
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
        [sys.executable, "-m", "mjlab_microduck.rom.runtime_child", "--socket-fd", str(child.fileno())],
        pass_fds=(child.fileno(), extra_a.fileno()),
    )
    child.close()
    extra_fd_path = Path(f"/proc/{process.pid}/fd/{extra_a.fileno()}")
    deadline = time.monotonic() + 2
    while extra_fd_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not extra_fd_path.exists()
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=3) == 0
    parent.close()
    extra_a.close()
    extra_b.close()


def test_environment_filtering_removes_unrelated_platform_configuration() -> None:
    script = (
        "import os; from mjlab_microduck.rom.runtime_child import clear_runtime_environment; "
        "clear_runtime_environment(); print('MICRODUCK_ROM_BEARER_TOKEN' in os.environ)"
    )
    environment = os.environ.copy()
    environment["MICRODUCK_ROM_BEARER_TOKEN"] = "must-not-survive"
    completed = subprocess.run(
        [sys.executable, "-c", script], env=environment, capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "False"


def test_blocked_start_cannot_defeat_local_emergency_zero() -> None:
    host, runtime, parent, thread = _active_host()
    host._handle = None
    runtime.active_handle = None
    host._bundle.bundleVersion = "1.0.0"
    host._bundle.actions[0].availability = "AVAILABLE"
    runtime.start_release.clear()
    request = RuntimeMessage(
        kind="START", generation=8, operationSequence=1, taskId="2" * 32,
        payload=StartPayload(
            actionCode="WALK_VELOCITY", bundleDigest=host._bundle.bundleDigest,
            parameters={"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
            scenario={"terrain": "flat", "seed": 1}, leaseMs=100,
        ),
    )
    parent.sendall(encode_packet(request))
    assert runtime.started.wait(timeout=1)
    assert runtime.emergency_stopped.wait(timeout=1)
    thread.join(timeout=1)
    assert not thread.is_alive()
    runtime.start_release.set()
    parent.close()


@pytest.mark.parametrize("operation", ["status", "stop"])
def test_blocked_status_or_stop_cannot_defeat_local_emergency_zero(operation: str) -> None:
    host, runtime, parent, thread = _active_host()
    with host._state_lock:
        host._lease_deadline = time.monotonic() + 0.1
    if operation == "status":
        runtime.status_release.clear()
        request = RuntimeMessage(
            kind="STATUS", generation=7, operationSequence=1, taskId="1" * 32, payload={}
        )
        started = runtime.status_started
        release = runtime.status_release
    else:
        runtime.command_release.clear()
        request = RuntimeMessage(
            kind="ZERO_AND_STOP", generation=7, operationSequence=1, taskId="1" * 32,
            payload=ZeroAndStopPayload(reason="OPERATOR_CANCELLED"),
        )
        started = runtime.command_started
        release = runtime.command_release
    parent.sendall(encode_packet(request))
    assert started.wait(timeout=1)
    assert runtime.emergency_stopped.wait(timeout=1)
    thread.join(timeout=1)
    assert not thread.is_alive()
    release.set()
    parent.close()
