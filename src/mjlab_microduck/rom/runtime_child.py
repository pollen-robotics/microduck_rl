"""Process-isolated owner of the governed MuJoCo/ONNX runtime."""

from __future__ import annotations

import argparse
import os
import queue
import signal
import socket
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from .action_catalog import (
    action_template,
    validate_code_owned_lease,
    validate_code_owned_parameters,
)
from .contracts import PolicyBundle, TaskCreateRequest, TaskEvidence
from .main import load_qualified_bundle
from .mujoco_runtime import MicroduckMujocoRuntime
from .parent_death import install_parent_death_signal, verify_seqpacket_socket
from .process_protocol import (
    PACKET_MAX_BYTES,
    AckPayload,
    CommandPayload,
    ErrorDetail,
    ErrorPayload,
    HelloPayload,
    LoadPayload,
    ProtocolViolation,
    ReadyPayload,
    RuntimeMessage,
    RuntimeMessageKind,
    RuntimeOperationKind,
    ShutdownPayload,
    StartPayload,
    StatusPayload,
    TerminalPayload,
    ZeroAndStopPayload,
    decode_packet,
    encode_packet,
)
from .runtime import RuntimeEvidence, RuntimeHandle, SimulationRuntime
from .runtime_identity import runtime_revision

_ALLOWED_ENVIRONMENT = frozenset(
    {"HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "MUJOCO_GL", "OMP_NUM_THREADS", "PATH"}
)
_ERROR_CODES = {
    RuntimeMessageKind.HELLO: "PROTOCOL_INCOMPATIBLE",
    RuntimeMessageKind.LOAD: "BUNDLE_UNAVAILABLE",
    RuntimeMessageKind.START: "START_FAILED",
    RuntimeMessageKind.COMMAND: "COMMAND_REJECTED",
    RuntimeMessageKind.STATUS: "STATUS_FAILED",
    RuntimeMessageKind.ZERO_AND_STOP: "STOP_FAILED",
    RuntimeMessageKind.SHUTDOWN: "SHUTDOWN_FAILED",
}


def clear_runtime_environment() -> None:
    kept = {name: value for name, value in os.environ.items() if name in _ALLOWED_ENVIRONMENT}
    os.environ.clear()
    os.environ.update(kept)


class RuntimeChildHost:
    """Own one runtime and enforce its lease independently of IPC execution."""

    def __init__(
        self,
        control: socket.socket,
        *,
        bundle_root: Path | None = None,
        runtime_factory: Callable[[Path, PolicyBundle], SimulationRuntime] = MicroduckMujocoRuntime,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if control.family != socket.AF_UNIX or (control.type & 0xF) != socket.SOCK_SEQPACKET:
            raise ValueError("runtime socket must be Unix SOCK_SEQPACKET")
        self._socket = control
        self._bundle_root = bundle_root
        self._runtime_factory = runtime_factory
        self._clock = clock
        self._messages: queue.Queue[RuntimeMessage | None] = queue.Queue(maxsize=8)
        self._send_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._safety_requested = threading.Event()
        self._safety_started = threading.Event()
        self._safety_complete = threading.Event()
        self._safety_start_lock = threading.Lock()
        self._safety_reason: str | None = None
        self._runtime: SimulationRuntime | None = None
        self._bundle: PolicyBundle | None = None
        self._handle: RuntimeHandle | None = None
        self._generation: int | None = None
        self._task_id: str | None = None
        self._lease_deadline: float | None = None
        self._last_sequence = -1
        self._last_request: RuntimeMessage | None = None

    def _send(self, message: RuntimeMessage) -> bool:
        try:
            with self._send_lock:
                self._socket.sendall(encode_packet(message))
            return True
        except OSError:
            return False

    def _response(self, request: RuntimeMessage, kind: RuntimeMessageKind, payload: object) -> RuntimeMessage:
        return RuntimeMessage(
            kind=kind,
            generation=request.generation,
            operationSequence=request.operationSequence,
            taskId=request.taskId,
            payload=payload,
        )

    def _error(self, request: RuntimeMessage, *, retryable: bool = False) -> None:
        operation = RuntimeOperationKind(request.kind.value)
        self._send(
            self._response(
                request,
                RuntimeMessageKind.ERROR,
                ErrorPayload(
                    operationKind=operation,
                    code=_ERROR_CODES[request.kind],
                    detail=ErrorDetail(retryable=retryable),
                ),
            )
        )

    def _receive(self) -> None:
        while not self._stop.is_set():
            try:
                packet, _ancillary, flags, _address = self._socket.recvmsg(PACKET_MAX_BYTES + 1)
            except OSError:
                packet = b""
                flags = 0
            if not packet:
                self._request_safety("PARENT_EOF")
                self._put_message(None)
                return
            if flags & socket.MSG_TRUNC or len(packet) > PACKET_MAX_BYTES:
                self._request_safety("PROTOCOL_ERROR")
                self._put_message(None)
                return
            try:
                message = decode_packet(packet)
            except ProtocolViolation:
                self._request_safety("PROTOCOL_ERROR")
                self._put_message(None)
                return
            if message.kind not in {
                RuntimeMessageKind.HELLO,
                RuntimeMessageKind.LOAD,
                RuntimeMessageKind.START,
                RuntimeMessageKind.COMMAND,
                RuntimeMessageKind.STATUS,
                RuntimeMessageKind.ZERO_AND_STOP,
                RuntimeMessageKind.SHUTDOWN,
            }:
                self._request_safety("PROTOCOL_ERROR")
                self._put_message(None)
                return
            if message.kind is RuntimeMessageKind.ZERO_AND_STOP:
                assert isinstance(message.payload, ZeroAndStopPayload)
                with self._state_lock:
                    matches = (
                        self._handle is not None
                        and message.generation == self._generation
                        and message.taskId == self._task_id
                    )
                    if matches:
                        self._last_request = message
                if not matches:
                    self._error(message)
                else:
                    self._request_safety(message.payload.reason)
                self._put_message(None)
                return
            self._put_message(message)

    def _put_message(self, message: RuntimeMessage | None) -> None:
        try:
            self._messages.put_nowait(message)
        except queue.Full:
            self._request_safety("PROTOCOL_ERROR")

    def _request_safety(self, reason: str) -> None:
        with self._state_lock:
            if self._safety_reason is None:
                self._safety_reason = reason
        self._safety_requested.set()

    def _deadman(self) -> None:
        while not self._stop.wait(0.01):
            with self._state_lock:
                deadline = self._lease_deadline
            if deadline is not None and self._clock() >= deadline:
                self._request_safety("LEASE_EXPIRED")
            if not self._safety_requested.is_set():
                continue
            self._perform_safety_stop()
            return

    def _perform_safety_stop(self) -> None:
        with self._safety_start_lock:
            if self._safety_started.is_set():
                return
            self._safety_started.set()
        with self._state_lock:
            runtime = self._runtime
            handle = self._handle
            reason = self._safety_reason or "RUNTIME_FAILED"
            request = self._last_request
            self._lease_deadline = None
        if runtime is None:
            self._safety_complete.set()
            return
        # This lock-independent call makes zero/disable intent visible even when a
        # native START, COMMAND, STATUS, or STOP call is wedged in another thread.
        try:
            runtime.emergency_stop(reason)
        except Exception:  # noqa: BLE001 - native safety failures are contained
            evidence = RuntimeEvidence(
                metrics={"safetyFailure": "EMERGENCY_STOP_FAILED"},
                stopReason=reason,
            )
        else:
            evidence = RuntimeEvidence(stopReason=reason)
        if handle is not None:
            try:
                template = action_template(self._bundle_action_code())
                zero: Mapping[str, object] = template.lease.zeroCommand if template.lease else {}
                runtime.command(handle, zero)
            except Exception:  # noqa: BLE001 - safe-stop still must be attempted
                evidence = RuntimeEvidence(
                    metrics={"safetyFailure": "ZERO_COMMAND_FAILED"},
                    stopReason=reason,
                )
            try:
                evidence = runtime.safe_stop(handle, reason)
            except Exception:  # noqa: BLE001 - child exits after bounded evidence
                evidence = RuntimeEvidence(
                    metrics={"safetyFailure": "SAFE_STOP_FAILED"},
                    stopReason=reason,
                )
        with self._state_lock:
            self._handle = None
        if request is not None and reason != "PARENT_EOF":
            self._send(self._terminal(request, reason, evidence))
        self._safety_complete.set()

    def _bundle_action_code(self) -> str:
        return self._active_action_code

    def _terminal(self, request: RuntimeMessage, reason: str, evidence: RuntimeEvidence) -> RuntimeMessage:
        assert self._bundle is not None
        action = next(item for item in self._bundle.actions if item.actionCode == self._active_action_code)
        policy = next(item for item in self._bundle.policies if item.policyRef == action.policyRef)
        outcome = "TIMED_OUT" if reason == "LEASE_EXPIRED" else "CANCELLED"
        payload = TerminalPayload(
            outcome=outcome,
            evidence=TaskEvidence(
                bundleDigest=self._bundle.bundleDigest,
                policyDigest=policy.digest,
                modelDigest=self._bundle.model.digest,
                metrics=dict(evidence.metrics),
                stopReason=reason,
            ),
        )
        return self._response(request, RuntimeMessageKind.TERMINAL, payload)

    def _handle_message(self, message: RuntimeMessage) -> bool:
        self._last_request = message
        if message.operationSequence <= self._last_sequence:
            self._error(message)
            return False
        self._last_sequence = message.operationSequence
        try:
            if message.kind is RuntimeMessageKind.HELLO:
                assert isinstance(message.payload, HelloPayload)
                if message.payload.runtimeRevision != runtime_revision():
                    self._error(message)
                    return False
                self._send(self._response(message, RuntimeMessageKind.ACK, AckPayload(acknowledgedKind="HELLO")))
            elif message.kind is RuntimeMessageKind.LOAD:
                assert isinstance(message.payload, LoadPayload)
                root = Path(message.payload.bundleRoot) if message.payload.bundleRoot else self._bundle_root
                if root is None:
                    raise ValueError
                bundle = load_qualified_bundle(root)
                if bundle.bundleDigest != message.payload.bundleDigest:
                    raise ValueError
                runtime = self._runtime_factory(root, bundle)
                self._bundle, self._runtime = bundle, runtime
                self._send(self._response(message, RuntimeMessageKind.READY, ReadyPayload(runtimeRevision=runtime_revision(), bundleDigest=bundle.bundleDigest)))
            elif message.kind is RuntimeMessageKind.START:
                self._start(message)
            elif message.kind is RuntimeMessageKind.COMMAND:
                self._command(message)
            elif message.kind is RuntimeMessageKind.STATUS:
                if not self._matches_active(message):
                    raise ValueError
                assert self._runtime is not None
                self._send(self._response(message, RuntimeMessageKind.STATUS, StatusPayload(status=self._runtime.status())))
            elif message.kind is RuntimeMessageKind.ZERO_AND_STOP:
                assert isinstance(message.payload, ZeroAndStopPayload)
                if not self._matches_active(message):
                    raise ValueError
                self._request_safety(message.payload.reason)
                self._safety_complete.wait()
                return False
            elif message.kind is RuntimeMessageKind.SHUTDOWN:
                assert isinstance(message.payload, ShutdownPayload)
                if self._handle is not None:
                    raise ValueError
                self._send(self._response(message, RuntimeMessageKind.ACK, AckPayload(acknowledgedKind="SHUTDOWN")))
                return False
        except Exception:  # noqa: BLE001 - peer receives only code-owned errors
            self._error(message)
            self._request_safety("RUNTIME_FAILED")
            return False
        return True

    def _matches_active(self, message: RuntimeMessage) -> bool:
        with self._state_lock:
            return (
                self._handle is not None
                and message.generation == self._generation
                and message.taskId == self._task_id
                and not self._safety_requested.is_set()
            )

    def _start(self, message: RuntimeMessage) -> None:
        assert isinstance(message.payload, StartPayload)
        if self._runtime is None or self._bundle is None or self._handle is not None:
            raise ValueError
        if message.payload.bundleDigest != self._bundle.bundleDigest:
            raise ValueError
        validate_code_owned_parameters(message.payload.actionCode, message.payload.parameters)
        validate_code_owned_lease(message.payload.actionCode, message.payload.leaseMs)
        action = next(item for item in self._bundle.actions if item.actionCode == message.payload.actionCode)
        if action.availability != "AVAILABLE":
            raise ValueError
        request = TaskCreateRequest(
            schema="MICRODUCK_SIM_TASK_V1",
            taskId=message.taskId,
            actionCode=message.payload.actionCode,
            bundleVersion=self._bundle.bundleVersion,
            bundleDigest=self._bundle.bundleDigest,
            parameters=message.payload.parameters,
            scenario=message.payload.scenario,
            leaseMs=message.payload.leaseMs,
            requestedBy="runtime-supervisor",
        )
        with self._state_lock:
            self._generation = message.generation
            self._task_id = message.taskId
            self._active_action_code = message.payload.actionCode
            self._lease_deadline = self._clock() + message.payload.leaseMs / 1000
        self._runtime.validate(action, request)
        handle = self._runtime.start(action, request)
        with self._state_lock:
            if self._safety_requested.is_set():
                self._runtime.safe_stop(handle, self._safety_reason or "RUNTIME_FAILED")
                return
            self._handle = handle
        self._send(self._response(message, RuntimeMessageKind.ACK, AckPayload(acknowledgedKind="START")))

    def _command(self, message: RuntimeMessage) -> None:
        assert isinstance(message.payload, CommandPayload)
        if not self._matches_active(message):
            raise ValueError
        validate_code_owned_parameters(self._active_action_code, message.payload.parameters)
        validate_code_owned_lease(self._active_action_code, message.payload.leaseMs)
        with self._state_lock:
            self._lease_deadline = self._clock() + message.payload.leaseMs / 1000
        assert self._runtime is not None and self._handle is not None
        self._runtime.command(self._handle, message.payload.parameters)
        if self._safety_requested.is_set():
            return
        self._send(self._response(message, RuntimeMessageKind.ACK, AckPayload(acknowledgedKind="COMMAND")))

    def run(self) -> int:
        receiver = threading.Thread(target=self._receive, name="runtime-child-ipc", daemon=True)
        deadman = threading.Thread(target=self._deadman, name="runtime-child-deadman", daemon=True)
        receiver.start()
        deadman.start()
        while not self._stop.is_set():
            message = self._messages.get()
            if message is None or not self._handle_message(message):
                break
        if self._safety_requested.is_set() and not self._safety_complete.is_set():
            self._perform_safety_stop()
        self._stop.set()
        if self._safety_requested.is_set():
            self._safety_complete.wait(timeout=2.0)
        try:
            self._socket.close()
        except OSError:
            pass
        return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-fd", type=int, required=True)
    parser.add_argument("--bundle-root", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    control = verify_seqpacket_socket(args.socket_fd)
    termination = threading.Event()
    signal.signal(signal.SIGTERM, lambda _signum, _frame: termination.set())
    install_parent_death_signal()
    clear_runtime_environment()
    host = RuntimeChildHost(control, bundle_root=args.bundle_root)
    watcher = threading.Thread(
        target=lambda: (termination.wait(), host._request_safety("PARENT_DEATH")),
        daemon=True,
    )
    watcher.start()
    return host.run()


if __name__ == "__main__":
    raise SystemExit(main())
