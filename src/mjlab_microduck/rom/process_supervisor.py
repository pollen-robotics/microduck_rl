"""Single-owner supervisor for the isolated simulator runtime process."""

from __future__ import annotations

import os
import queue
import select
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .contracts import RobotStatus, TaskCommandRequest, TaskCreateRequest
from .process_protocol import (
    PACKET_MAX_BYTES,
    AckPayload,
    CommandPayload,
    HelloPayload,
    LoadPayload,
    RuntimeMessage,
    RuntimeMessageKind,
    ShutdownPayload,
    StartPayload,
    StatusPayload,
    StatusRequestPayload,
    TerminalPayload,
    ZeroAndStopPayload,
    decode_packet,
    encode_packet,
)
from .runtime_identity import runtime_revision
from .supervisor_state import SupervisorState


class SupervisorUnavailable(RuntimeError):
    """The child cannot safely accept the requested operation."""


class SupervisorOperationError(RuntimeError):
    """A bounded child operation failed or became ambiguous."""


@dataclass(frozen=True, slots=True)
class SupervisorSnapshot:
    state: SupervisorState
    generation: int
    child_healthy: bool
    cached_status: RobotStatus | None
    quarantine_reason: str | None
    slot_releasable: bool
    pid: int | None = None


@dataclass(frozen=True, slots=True)
class ChildLaunch:
    argv: tuple[str, ...]
    pass_fds: tuple[int, ...] = ()
    close_after_spawn: tuple[int, ...] = ()
    env: Mapping[str, str] | None = None


type LaunchFactory = Callable[[int], ChildLaunch]
type IntentKind = Literal["ready", "start", "command", "status", "stop", "close"]


@dataclass(slots=True)
class _Intent:
    kind: IntentKind
    args: tuple[Any, ...] = ()
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None


class RuntimeProcessSupervisor:
    """Own exactly one child and all operations on its process/socket."""

    def __init__(
        self,
        *,
        bundle_root: Path | str,
        bundle_digest: str,
        launch_factory: LaunchFactory | None = None,
        operation_timeout_s: float = 1.0,
        terminate_timeout_s: float = 0.25,
        queue_size: int = 8,
        terminal_callback: Callable[[TerminalPayload], None] | None = None,
    ) -> None:
        if min(operation_timeout_s, terminate_timeout_s) <= 0 or queue_size <= 0:
            raise ValueError("supervisor bounds must be positive")
        self._bundle_root = str(bundle_root)
        self._bundle_digest = bundle_digest
        self._launch_factory = launch_factory or self._default_launch
        self._operation_timeout = operation_timeout_s
        self._terminate_timeout = terminate_timeout_s
        self._terminal_callback = terminal_callback
        self._queue: queue.Queue[_Intent] = queue.Queue(maxsize=queue_size)
        self._snapshot_lock = threading.Lock()
        self._snapshot_value = SupervisorSnapshot(
            SupervisorState.NO_CHILD, 0, False, None, None, True
        )
        self._process: subprocess.Popen[bytes] | None = None
        self._socket: socket.socket | None = None
        self._generation = 0
        self._sequence = 0
        self._active_task: str | None = None
        self._trace: list[str] = []
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="microduck-runtime-supervisor", daemon=True
        )
        self._thread.start()

    def _default_launch(self, socket_fd: int) -> ChildLaunch:
        return ChildLaunch(
            (
                sys.executable,
                "-m",
                "mjlab_microduck.rom.runtime_child",
                "--socket-fd",
                str(socket_fd),
                "--bundle-root",
                self._bundle_root,
            )
        )

    @property
    def trace(self) -> tuple[str, ...]:
        return tuple(self._trace)

    def snapshot(self) -> SupervisorSnapshot:
        with self._snapshot_lock:
            return self._snapshot_value

    def readiness(self) -> bool:
        snap = self.snapshot()
        return snap.child_healthy and snap.state in {
            SupervisorState.IDLE,
            SupervisorState.RUNNING,
        }

    def ensure_ready(self) -> SupervisorSnapshot:
        return self._submit("ready")

    def start(self, request: TaskCreateRequest) -> AckPayload:
        return self._submit("start", request)

    def command(
        self,
        task_id: str,
        command: TaskCommandRequest | Mapping[str, object],
        lease_ms: int | None = None,
    ) -> AckPayload:
        return self._submit("command", task_id, command, lease_ms)

    def status(self, task_id: str) -> RobotStatus:
        return self._submit("status", task_id)

    def stop(self, task_id: str, reason: str) -> TerminalPayload:
        return self._submit("stop", task_id, reason)

    def close(self) -> None:
        if self._closed and not self._thread.is_alive():
            return
        try:
            self._submit("close")
        except SupervisorUnavailable:
            pass
        self._thread.join(self._operation_timeout + self._terminate_timeout + 1)

    def _submit(self, kind: IntentKind, *args: object) -> Any:
        if self._closed and kind != "close":
            raise SupervisorUnavailable("supervisor is closed")
        intent = _Intent(kind=kind, args=args)  # type: ignore[arg-type]
        try:
            self._queue.put(intent, timeout=self._operation_timeout)
        except queue.Full as exc:
            raise SupervisorUnavailable("supervisor intent queue is full") from exc
        wait_bound = self._operation_timeout * 3 + self._terminate_timeout * 2 + 1
        if not intent.done.wait(wait_bound):
            raise SupervisorUnavailable(
                "supervisor owner did not complete bounded intent"
            )
        if intent.error is not None:
            raise intent.error
        return intent.result

    def _publish(
        self,
        state: SupervisorState,
        *,
        healthy: bool | None = None,
        status: RobotStatus | None | object = ...,
        reason: str | None | object = ...,
        slot: bool | None = None,
    ) -> None:
        old = self._snapshot_value
        new = SupervisorSnapshot(
            state=state,
            generation=self._generation,
            child_healthy=old.child_healthy if healthy is None else healthy,
            cached_status=old.cached_status if status is ... else status,  # type: ignore[arg-type]
            quarantine_reason=old.quarantine_reason if reason is ... else reason,  # type: ignore[arg-type]
            slot_releasable=old.slot_releasable if slot is None else slot,
            pid=self._process.pid if self._process is not None else None,
        )
        with self._snapshot_lock:
            self._snapshot_value = new

    def _run(self) -> None:
        while True:
            intent = self._queue.get()
            try:
                intent.result = self._dispatch(intent)
            except Exception as exc:  # noqa: BLE001 - owner must wake callers
                intent.error = exc
            finally:
                intent.done.set()
            if intent.kind == "close":
                return

    def _dispatch(self, intent: _Intent) -> Any:
        if intent.kind == "ready":
            if self._process is None:
                self._spawn()
            if self.snapshot().state is not SupervisorState.IDLE:
                raise SupervisorUnavailable("runtime child is not idle")
            return self.snapshot()
        if intent.kind == "close":
            self._closed = True
            self._close_owned_child()
            return None
        if self._process is None:
            self._spawn()
        if intent.kind == "start":
            return self._start(intent.args[0])
        if intent.kind == "command":
            return self._command(*intent.args)
        if intent.kind == "status":
            return self._status(intent.args[0])
        if intent.kind == "stop":
            return self._stop(intent.args[0], intent.args[1])
        raise AssertionError(intent.kind)

    def _spawn(self) -> None:
        self._generation += 1
        self._sequence = 0
        self._publish(SupervisorState.SPAWNING, healthy=False, reason=None, slot=False)
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        launch = self._launch_factory(child.fileno())
        pass_fds = tuple(dict.fromkeys((child.fileno(), *launch.pass_fds)))
        try:
            self._process = subprocess.Popen(
                list(launch.argv), pass_fds=pass_fds, env=launch.env, close_fds=True
            )
        except BaseException:
            parent.close()
            child.close()
            self._publish(SupervisorState.NO_CHILD, healthy=False, slot=True)
            raise
        finally:
            for inherited_fd in launch.close_after_spawn:
                try:
                    os.close(inherited_fd)
                except OSError:
                    continue
        child.close()
        self._socket = parent
        try:
            hello = self._exchange(
                RuntimeMessageKind.HELLO,
                None,
                HelloPayload(runtimeRevision=runtime_revision()),
                {RuntimeMessageKind.ACK},
            )
            assert isinstance(hello.payload, AckPayload)
            if hello.payload.acknowledgedKind.value != "HELLO":
                raise SupervisorOperationError("wrong HELLO acknowledgment")
            ready = self._exchange(
                RuntimeMessageKind.LOAD,
                None,
                LoadPayload(
                    bundleDigest=self._bundle_digest, bundleRoot=self._bundle_root
                ),
                {RuntimeMessageKind.READY},
            )
            if (
                ready.payload.bundleDigest != self._bundle_digest
                or ready.payload.runtimeRevision != runtime_revision()
            ):  # type: ignore[union-attr]
                raise SupervisorOperationError("wrong child readiness identity")
        except BaseException as exc:
            self._quarantine(f"SPAWN_FAILED:{type(exc).__name__}")
            raise SupervisorOperationError("child readiness failed") from exc
        self._publish(SupervisorState.IDLE, healthy=True, reason=None, slot=True)

    def _start(self, request: TaskCreateRequest) -> AckPayload:
        if self.snapshot().state is not SupervisorState.IDLE:
            raise SupervisorUnavailable("motion slot is unavailable")
        self._active_task = request.taskId
        self._publish(SupervisorState.STARTING, slot=False)
        payload = StartPayload(
            actionCode=request.actionCode,
            bundleDigest=request.bundleDigest,
            parameters=request.parameters,
            scenario=request.scenario,
            leaseMs=request.leaseMs or 1000,
        )
        response = self._guarded_exchange(
            RuntimeMessageKind.START, request.taskId, payload, {RuntimeMessageKind.ACK}
        )
        assert isinstance(response.payload, AckPayload)
        if response.payload.acknowledgedKind.value != "START":
            self._quarantine("START_ACK_MISMATCH")
            raise SupervisorOperationError("wrong START acknowledgment")
        self._publish(SupervisorState.RUNNING, slot=False)
        return response.payload

    def _command(
        self,
        task_id: str,
        command: TaskCommandRequest | Mapping[str, object],
        lease_ms: int | None,
    ) -> AckPayload:
        self._require_active(task_id)
        if isinstance(command, TaskCommandRequest):
            parameters, lease = command.parameters, command.leaseMs
        else:
            parameters, lease = command, lease_ms
        if lease is None:
            raise ValueError("lease_ms is required")
        response = self._guarded_exchange(
            RuntimeMessageKind.COMMAND,
            task_id,
            CommandPayload(parameters=parameters, leaseMs=lease),
            {RuntimeMessageKind.ACK},
        )
        assert isinstance(response.payload, AckPayload)
        return response.payload

    def _status(self, task_id: str) -> RobotStatus:
        self._require_active(task_id)
        response = self._guarded_exchange(
            RuntimeMessageKind.STATUS,
            task_id,
            StatusRequestPayload(),
            {RuntimeMessageKind.STATUS},
        )
        assert isinstance(response.payload, StatusPayload)
        self._publish(self.snapshot().state, status=response.payload.status)
        return response.payload.status

    def _stop(self, task_id: str, reason: str) -> TerminalPayload:
        self._require_active(task_id)
        self._publish(SupervisorState.STOPPING, slot=False)
        response = self._guarded_exchange(
            RuntimeMessageKind.ZERO_AND_STOP,
            task_id,
            ZeroAndStopPayload(reason=reason),
            {RuntimeMessageKind.TERMINAL},
        )
        assert isinstance(response.payload, TerminalPayload)
        self._active_task = None
        self._publish(SupervisorState.IDLE, healthy=True, slot=True)
        if self._terminal_callback is not None:
            self._terminal_callback(response.payload)
        return response.payload

    def _require_active(self, task_id: str) -> None:
        if (
            self.snapshot().state is not SupervisorState.RUNNING
            or task_id != self._active_task
        ):
            raise SupervisorUnavailable("task does not own the runtime")

    def _guarded_exchange(self, *args: Any) -> RuntimeMessage:
        try:
            return self._exchange(*args)
        except BaseException as exc:
            self._trace.append(
                "OPERATION_TIMEOUT"
                if isinstance(exc, TimeoutError)
                else "OPERATION_FAILED"
            )
            self._quarantine(type(exc).__name__)
            raise SupervisorOperationError("runtime operation failed closed") from exc

    def _exchange(
        self,
        kind: RuntimeMessageKind,
        task_id: str | None,
        payload: object,
        expected: set[RuntimeMessageKind],
    ) -> RuntimeMessage:
        if self._socket is None or self._process is None:
            raise SupervisorUnavailable("no owned child")
        self._sequence += 1
        request = RuntimeMessage(
            kind=kind,
            generation=self._generation,
            operationSequence=self._sequence,
            taskId=task_id,
            payload=payload,
        )
        self._socket.sendall(encode_packet(request))
        deadline = time.monotonic() + self._operation_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("child operation deadline")
            if self._process.poll() is not None:
                raise ChildProcessError("child exited before response")
            readable, _, _ = select.select([self._socket], [], [], min(remaining, 0.05))
            if not readable:
                continue
            packet, _ancillary, flags, _address = self._socket.recvmsg(
                PACKET_MAX_BYTES + 1
            )
            if not packet or flags & socket.MSG_TRUNC:
                raise ConnectionError("child transport closed or truncated")
            response = decode_packet(packet)
            if (
                response.generation != self._generation
                or response.operationSequence != self._sequence
                or response.taskId != task_id
                or response.kind not in expected
            ):
                raise SupervisorOperationError("ambiguous child response")
            return response

    def _quarantine(self, reason: str) -> None:
        self._publish(
            SupervisorState.QUARANTINED, healthy=False, reason=reason, slot=False
        )
        self._trace.append("QUARANTINED")
        self._terminate_and_reap()

    def _terminate_and_reap(self) -> None:
        process = self._process
        owned_socket = self._socket
        if process is None:
            self._publish(SupervisorState.NO_CHILD, healthy=False, slot=True)
            return
        self._publish(SupervisorState.TERMINATING, slot=False)
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            self._trace.append("SIGTERM_SENT")
            try:
                process.wait(timeout=self._terminate_timeout)
            except subprocess.TimeoutExpired:
                self._trace.append("TERM_TIMEOUT")
                self._publish(SupervisorState.KILLING, slot=False)
                process.kill()
                self._trace.append("SIGKILL_SENT")
        self._publish(SupervisorState.REAPING, slot=False)
        process.wait(timeout=max(self._operation_timeout, 0.1))
        if process.poll() is None:
            raise RuntimeError("exact child reap was not confirmed")
        self._trace.append("CHILD_REAPED")
        if owned_socket is not None:
            owned_socket.close()
        self._process = None
        self._socket = None
        self._active_task = None
        self._publish(SupervisorState.NO_CHILD, healthy=False, slot=True)
        self._trace.append("NO_CHILD")

    def _close_owned_child(self) -> None:
        if self._process is None:
            self._publish(SupervisorState.NO_CHILD, healthy=False, slot=True)
            return
        if self.snapshot().state is SupervisorState.IDLE:
            try:
                self._exchange(
                    RuntimeMessageKind.SHUTDOWN,
                    None,
                    ShutdownPayload(reason="SUPERVISOR_SHUTDOWN"),
                    {RuntimeMessageKind.ACK},
                )
            except Exception as exc:  # noqa: BLE001 - shutdown escalates below
                self._trace.append(f"SHUTDOWN_FAILED:{type(exc).__name__}")
        self._terminate_and_reap()
