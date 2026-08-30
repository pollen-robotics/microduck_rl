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
from dataclasses import dataclass
from pathlib import Path

from .action_catalog import (
    action_template,
    validate_code_owned_lease,
    validate_code_owned_parameters,
)
from .contracts import PolicyBundle, TaskCreateRequest, TaskEvidence
from .main import load_qualified_bundle
from .mujoco_runtime import MicroduckMujocoRuntime
from .parent_death import (
    close_unrelated_fds,
    install_parent_death_signal,
    verify_seqpacket_socket,
)
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
    TerminalEventPayload,
    TerminalPayload,
    ZeroAndStopPayload,
    decode_packet,
    encode_packet,
)
from .runtime import RuntimeEvidence, RuntimeHandle, RuntimeSample, SimulationRuntime
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
_FATAL_CLEANUP_TIMEOUT_S = 0.25
_TRUTHFUL_SAFETY_FAILURES = frozenset()
_UNTRUTHFUL_SAFETY_FAILURES = frozenset(
    {
        "EMERGENCY_STOP_FAILED",
        "RUNTIME_UNRESPONSIVE",
        "SAFE_STOP_FAILED",
        "WATCHDOG_FAILURE",
        "ZERO_COMMAND_FAILED",
    }
)


def _cleanup_evidence_is_truthful(evidence: RuntimeEvidence) -> bool:
    """Accept cleanup only when code-owned evidence proves containment."""
    if "safetyFailure" not in evidence.metrics:
        return True
    safety_failure = evidence.metrics["safetyFailure"]
    if not isinstance(safety_failure, str):
        return False
    if safety_failure in _TRUTHFUL_SAFETY_FAILURES:
        return True
    if safety_failure in _UNTRUTHFUL_SAFETY_FAILURES:
        return False
    return False


@dataclass(frozen=True, slots=True)
class _RuntimeCompletion:
    generation: int
    task_id: str
    handle: RuntimeHandle
    sample: RuntimeSample
    outcome: str
    reason: str


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
        fatal_cleanup_timeout_s: float = _FATAL_CLEANUP_TIMEOUT_S,
    ) -> None:
        if control.family != socket.AF_UNIX or (control.type & 0xF) != socket.SOCK_SEQPACKET:
            raise ValueError("runtime socket must be Unix SOCK_SEQPACKET")
        self._socket = control
        self._bundle_root = bundle_root
        self._runtime_factory = runtime_factory
        self._clock = clock
        if fatal_cleanup_timeout_s <= 0:
            raise ValueError("fatal cleanup timeout must be positive")
        self._fatal_cleanup_timeout_s = fatal_cleanup_timeout_s
        self._messages: queue.Queue[RuntimeMessage | _RuntimeCompletion | None] = queue.Queue(maxsize=8)
        self._send_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._safety_requested = threading.Event()
        self._safety_started = threading.Event()
        self._safety_complete = threading.Event()
        self._cleanup_timed_out = threading.Event()
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
        self._operation_active = threading.Event()
        self._event_sequence = 0
        self._completion_claim = threading.Lock()
        self._sample_thread: threading.Thread | None = None
        self._sample_stop = threading.Event()
        self._latest_sample_metrics: dict[str, object] = {}
        self._completed_identity: tuple[int, str] | None = None
        self._truthfully_stopped_completion: tuple[int, str, RuntimeHandle] | None = None

    @property
    def sample_monitor_alive(self) -> bool:
        return self._sample_thread is not None and self._sample_thread.is_alive()

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
            if (
                message.kind is RuntimeMessageKind.ZERO_AND_STOP
                and self._operation_active.is_set()
            ):
                with self._state_lock:
                    self._last_request = message
                self._request_safety("RUNTIME_UNRESPONSIVE")
                return
            self._put_message(message)

    def _put_message(
        self, message: RuntimeMessage | _RuntimeCompletion | None
    ) -> None:
        try:
            self._messages.put_nowait(message)
        except queue.Full:
            self._request_safety("PROTOCOL_ERROR")

    def _request_safety(self, reason: str) -> None:
        with self._state_lock:
            if self._safety_reason is None:
                self._safety_reason = reason
        self._safety_requested.set()
        try:
            self._messages.put_nowait(None)
        except queue.Full:
            pass

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
        if handle is None and self._operation_active.is_set() and self._task_id is not None:
            # START may already own native resources but has not returned its handle.
            # Emergency zero is bounded; truthful cleanup acknowledgement is impossible.
            self._cleanup_timed_out.set()
            return
        cleanup_result: list[tuple[RuntimeEvidence, bool]] = []

        def cleanup() -> None:
            cleanup_evidence = evidence
            if handle is None:
                cleanup_result.append(
                    (cleanup_evidence, _cleanup_evidence_is_truthful(cleanup_evidence))
                )
                return
            try:
                template = action_template(self._bundle_action_code())
                zero: Mapping[str, object] = template.lease.zeroCommand if template.lease else {}
                runtime.command(handle, zero)
            except Exception:  # noqa: BLE001 - safe-stop still must be attempted
                cleanup_evidence = RuntimeEvidence(
                    metrics={"safetyFailure": "ZERO_COMMAND_FAILED"},
                    stopReason=reason,
                )
            try:
                raw_stopped = runtime.safe_stop(handle, reason)
                stopped = RuntimeEvidence(
                    metrics=raw_stopped.metrics, stopReason=raw_stopped.stopReason
                )
                if "safetyFailure" not in cleanup_evidence.metrics:
                    cleanup_evidence = stopped
            except Exception:  # noqa: BLE001 - child exits after bounded evidence
                cleanup_evidence = RuntimeEvidence(
                    metrics={"safetyFailure": "SAFE_STOP_FAILED"},
                    stopReason=reason,
                )
                cleanup_result.append((cleanup_evidence, False))
                return
            cleanup_result.append(
                (cleanup_evidence, _cleanup_evidence_is_truthful(cleanup_evidence))
            )

        if handle is not None:
            cleanup_thread = threading.Thread(
                target=cleanup, name="runtime-child-best-effort-cleanup", daemon=True
            )
            cleanup_thread.start()
            cleanup_thread.join(timeout=self._fatal_cleanup_timeout_s)
            if cleanup_thread.is_alive():
                self._cleanup_timed_out.set()
                return
            evidence, cleanup_truthful = cleanup_result[0]
            if not cleanup_truthful:
                self._retire_uncertain_cleanup()
                return
            self._sample_stop.set()
            monitor = self._sample_thread
            if monitor is not None:
                monitor.join(timeout=self._fatal_cleanup_timeout_s)
            if monitor is not None and monitor.is_alive():
                self._cleanup_timed_out.set()
                return
            with self._state_lock:
                self._handle = None
                self._sample_thread = None
        if not _cleanup_evidence_is_truthful(evidence):
            self._retire_uncertain_cleanup()
            return
        if (
            request is not None
            and request.taskId is not None
            and self._task_id is not None
            and reason != "PARENT_EOF"
        ):
            terminal = self._terminal(request, reason, evidence)
            if reason == "LEASE_EXPIRED":
                self._event_sequence += 1
                assert isinstance(terminal.payload, TerminalPayload)
                terminal = RuntimeMessage(
                    kind=RuntimeMessageKind.TERMINAL_EVENT,
                    generation=request.generation,
                    operationSequence=0,
                    taskId=request.taskId,
                    payload=TerminalEventPayload(
                        eventSequence=self._event_sequence,
                        terminal=terminal.payload,
                    ),
                )
            self._send(terminal)
        self._safety_complete.set()

    def _bundle_action_code(self) -> str:
        return self._active_action_code

    def _terminal(self, request: RuntimeMessage, reason: str, evidence: RuntimeEvidence) -> RuntimeMessage:
        if not _cleanup_evidence_is_truthful(evidence):
            raise ValueError("cleanup evidence does not prove containment")
        assert self._bundle is not None
        action = next(item for item in self._bundle.actions if item.actionCode == self._active_action_code)
        policy = next(item for item in self._bundle.policies if item.policyRef == action.policyRef)
        if reason == "LEASE_EXPIRED":
            outcome = "TIMED_OUT"
        elif reason in {"OPERATOR_CANCELLED", "CANCELLED", "USER_CANCELLED"}:
            outcome = "CANCELLED"
        else:
            outcome = "FAILED"
        metrics = dict(self._latest_sample_metrics)
        for key in sorted(evidence.metrics):
            candidate = metrics | {key: evidence.metrics[key]}
            try:
                RuntimeEvidence(metrics=candidate)
            except (TypeError, ValueError):
                break
            metrics = candidate
        payload = TerminalPayload(
            outcome=outcome,
            evidence=TaskEvidence(
                bundleDigest=self._bundle.bundleDigest,
                policyDigest=policy.digest,
                modelDigest=self._bundle.model.digest,
                metrics=metrics,
                stopReason=reason,
            ),
        )
        return self._response(request, RuntimeMessageKind.TERMINAL, payload)

    def _start_runtime_monitor(self) -> None:
        template = action_template(self._active_action_code)
        deadline = (
            self._clock() + template.completion.maxDurationMs / 1000
            if template.completion is not None
            else None
        )
        self._sample_stop.clear()

        def monitor() -> None:
            assert self._runtime is not None and self._handle is not None
            runtime, handle = self._runtime, self._handle
            assert self._generation is not None and self._task_id is not None
            generation, task_id = self._generation, self._task_id
            sample = RuntimeSample(running=True)
            reason = "MAX_DURATION_EXCEEDED"
            outcome = "TIMED_OUT"
            try:
                while (
                    (deadline is None or self._clock() < deadline)
                    and not self._safety_requested.is_set()
                    and not self._sample_stop.is_set()
                ):
                    sample = runtime.sample(handle)
                    if sample.metrics:
                        with self._state_lock:
                            self._latest_sample_metrics = dict(sample.metrics)
                    if sample.terminalState is not None:
                        reason = sample.stopReason or (
                            "TASK_COMPLETE" if sample.terminalState == "SUCCEEDED" else "RUNTIME_FAILED"
                        )
                        outcome = sample.terminalState
                        break
                    self._sample_stop.wait(0.02)
                else:
                    if self._safety_requested.is_set() or self._sample_stop.is_set():
                        return
            except Exception:  # noqa: BLE001 - runtime detail is never serialized
                sample = RuntimeSample(running=False, terminalState="FAILED")
                reason, outcome = "RUNTIME_EXCEPTION", "FAILED"
            self._put_message(
                _RuntimeCompletion(generation, task_id, handle, sample, outcome, reason)
            )

        self._sample_thread = threading.Thread(
            target=monitor, name="runtime-child-sample-monitor", daemon=True
        )
        self._sample_thread.start()

    def _start_discrete_monitor(self) -> None:
        """Compatibility shim for focused tests; every active action is monitored."""
        self._start_runtime_monitor()

    def _retire_uncertain_cleanup(self) -> None:
        self._cleanup_timed_out.set()
        self._stop.set()
        self._put_message(None)

    def _handle_runtime_completion(self, completion: _RuntimeCompletion) -> None:
        monitor = self._sample_thread
        if monitor is None:
            stopped = self._truthfully_stopped_completion
            if (
                stopped is not None
                and completion.generation == stopped[0]
                and completion.task_id == stopped[1]
                and completion.handle is stopped[2]
            ):
                self._truthfully_stopped_completion = None
                return
            self._retire_uncertain_cleanup()
            return
        monitor.join(timeout=self._fatal_cleanup_timeout_s)
        if monitor.is_alive():
            self._retire_uncertain_cleanup()
            return
        with self._completion_claim:
            with self._state_lock:
                if completion.handle is not self._handle or self._safety_requested.is_set():
                    return
            assert self._runtime is not None
            try:
                raw_stopped = self._runtime.safe_stop(completion.handle, completion.reason)
                stopped = RuntimeEvidence(
                    metrics=raw_stopped.metrics, stopReason=raw_stopped.stopReason
                )
            except Exception:  # noqa: BLE001 - uncertain cleanup requires exact reap
                self._retire_uncertain_cleanup()
                return
            if not _cleanup_evidence_is_truthful(stopped):
                self._retire_uncertain_cleanup()
                return
            metrics = dict(completion.sample.metrics)
            for key in sorted(stopped.metrics):
                candidate = metrics | {key: stopped.metrics[key]}
                try:
                    RuntimeEvidence(metrics=candidate)
                except (TypeError, ValueError):
                    break
                metrics = candidate
            assert self._bundle is not None
            action = next(item for item in self._bundle.actions if item.actionCode == self._active_action_code)
            policy = next(item for item in self._bundle.policies if item.policyRef == action.policyRef)
            terminal = TerminalPayload(
                outcome=completion.outcome,
                evidence=TaskEvidence(
                    bundleDigest=self._bundle.bundleDigest,
                    policyDigest=policy.digest,
                    modelDigest=self._bundle.model.digest,
                    metrics=metrics, stopReason=completion.reason,
                ),
            )
            with self._state_lock:
                generation, task_id = self._generation, self._task_id
                if (
                    generation != completion.generation
                    or task_id != completion.task_id
                ):
                    self._retire_uncertain_cleanup()
                    return
                self._handle = None
                self._lease_deadline = None
                self._event_sequence += 1
                event_sequence = self._event_sequence
            assert generation is not None and task_id is not None
            self._send(RuntimeMessage(
                kind=RuntimeMessageKind.TERMINAL_EVENT,
                generation=generation, operationSequence=0, taskId=task_id,
                payload=TerminalEventPayload(eventSequence=event_sequence, terminal=terminal),
            ))
            with self._state_lock:
                self._completed_identity = (generation, task_id)
                self._generation = None
                self._task_id = None
                self._last_request = None
                self._sample_thread = None

    def _retire_sample_monitor(self) -> bool:
        """Stop and join the task monitor before allowing this child to be reused."""
        monitor = self._sample_thread
        if monitor is None:
            return True
        self._sample_stop.set()
        monitor.join(timeout=self._fatal_cleanup_timeout_s)
        if monitor.is_alive():
            assert self._runtime is not None
            try:
                self._runtime.emergency_stop("RUNTIME_UNRESPONSIVE")
            except Exception:  # noqa: BLE001, S110 - exact reap remains the cleanup barrier
                pass
            self._retire_uncertain_cleanup()
            return False
        self._sample_thread = None
        return True

    def _handle_message(self, message: RuntimeMessage) -> bool:
        self._last_request = message
        if message.operationSequence <= self._last_sequence:
            self._error(message)
            return False
        self._last_sequence = message.operationSequence
        with self._state_lock:
            completed_identity = self._completed_identity
            active_handle = self._handle
        if (
            active_handle is None
            and completed_identity == (message.generation, message.taskId)
            and message.kind in {
                RuntimeMessageKind.COMMAND,
                RuntimeMessageKind.STATUS,
                RuntimeMessageKind.ZERO_AND_STOP,
            }
        ):
            # The unsolicited terminal won the race. The supervisor aborts the
            # pending operation when it consumes that event, so a late correlated
            # response would only poison the next exchange on this reusable socket.
            return True
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
                self._normal_stop(message, message.payload.reason)
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

    def _normal_stop(self, message: RuntimeMessage, reason: str) -> None:
        with self._completion_claim:
            assert self._runtime is not None and self._handle is not None
            handle = self._handle
            if not self._retire_sample_monitor():
                return
            template = action_template(self._active_action_code)
            zero: Mapping[str, object] = template.lease.zeroCommand if template.lease else {}
            self._runtime.command(handle, zero)
            raw_evidence = self._runtime.safe_stop(handle, reason)
            evidence = RuntimeEvidence(
                metrics=raw_evidence.metrics, stopReason=raw_evidence.stopReason
            )
            if not _cleanup_evidence_is_truthful(evidence):
                self._retire_uncertain_cleanup()
                return
            self._send(self._terminal(message, reason, evidence))
            with self._state_lock:
                self._truthfully_stopped_completion = (
                    message.generation, message.taskId, handle
                )
                self._handle = None
                self._generation = None
                self._task_id = None
                self._lease_deadline = None
                self._last_request = None

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
            self._latest_sample_metrics = {}
            self._completed_identity = None
            self._truthfully_stopped_completion = None
        self._runtime.validate(action, request)
        handle = self._runtime.start(action, request)
        with self._state_lock:
            if self._safety_requested.is_set():
                self._runtime.safe_stop(handle, self._safety_reason or "RUNTIME_FAILED")
                return
            self._handle = handle
        self._send(self._response(message, RuntimeMessageKind.ACK, AckPayload(acknowledgedKind="START")))
        self._event_sequence = 0
        self._start_runtime_monitor()

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
            if message is None:
                break
            if isinstance(message, _RuntimeCompletion):
                self._handle_runtime_completion(message)
                continue
            result: list[bool] = []

            def execute(
                current: RuntimeMessage = message, outcome: list[bool] = result
            ) -> None:
                outcome.append(self._handle_message(current))

            self._operation_active.set()
            operation = threading.Thread(
                target=execute, name="runtime-child-operation", daemon=True
            )
            operation.start()
            while operation.is_alive() and not self._safety_requested.wait(0.01):
                pass
            self._operation_active.clear()
            if self._safety_requested.is_set() or not result or not result[0]:
                break
        if self._safety_requested.is_set() and not self._safety_complete.is_set():
            self._perform_safety_stop()
        self._stop.set()
        if self._safety_requested.is_set():
            self._safety_complete.wait(timeout=self._fatal_cleanup_timeout_s + 0.05)
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
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
    close_unrelated_fds({args.socket_fd})
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
