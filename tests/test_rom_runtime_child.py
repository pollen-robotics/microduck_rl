from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from mjlab_microduck.rom.parent_death import verify_seqpacket_socket
from mjlab_microduck.rom.process_protocol import (
    CommandPayload,
    HelloPayload,
    LoadPayload,
    RuntimeMessage,
    RuntimeMessageKind,
    decode_packet,
    encode_packet,
)
from mjlab_microduck.rom.runtime import RuntimeHandle
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
        bundleDigest="sha256:" + "a" * 64,
        model=SimpleNamespace(digest="sha256:" + "b" * 64),
        actions=[SimpleNamespace(actionCode="WALK_VELOCITY", policyRef="walk")],
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
