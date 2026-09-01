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
from itertools import count
from pathlib import Path
from typing import Any, Literal

from .contracts import RobotStatus, TaskCommandRequest, TaskCreateRequest
from .process_protocol import (
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
    TerminalEventPayload,
    TerminalPayload,
    ZeroAndStopPayload,
    decode_packet,
    encode_packet,
    receive_packet,
)
from .runtime_identity import runtime_revision
from .supervisor_state import SupervisorEvent, SupervisorState, transition


class SupervisorUnavailable(RuntimeError):
    """The child cannot safely accept the requested operation."""


class SupervisorOperationError(RuntimeError):
    """A bounded child operation failed or became ambiguous."""


class SupervisorTaskTerminalized(SupervisorOperationError):
    """The active task safely terminalized while another operation was pending."""


@dataclass(frozen=True, slots=True)
class CorrelatedTerminalDelivery:
    """Supervisor-validated identity attached to every terminal callback."""

    generation: int
    task_id: str
    event_sequence: int
    terminal: TerminalPayload

    @property
    def outcome(self) -> str:
        return self.terminal.outcome

    @property
    def evidence(self):
        return self.terminal.evidence


@dataclass(frozen=True, slots=True)
class StartAcknowledgement:
    """Exact child identity accepted by the sole process owner at START ACK."""

    generation: int
    task_id: str
    acknowledgement: AckPayload


@dataclass(frozen=True, slots=True)
class ReapReceipt:
    """Immutable proof that one exact owned child crossed ``Popen.wait()``."""

    generation: int
    pid: int
    ownership_identity: int


@dataclass(frozen=True, slots=True)
class SupervisorSnapshot:
    state: SupervisorState
    generation: int
    child_healthy: bool
    cached_status: RobotStatus | None
    quarantine_reason: str | None
    slot_releasable: bool
    terminal_delivery_outstanding: bool = False
    cached_terminal: CorrelatedTerminalDelivery | None = None
    pid: int | None = None
    ownership_identity: int | None = None


@dataclass(frozen=True, slots=True)
class ChildLaunch:
    argv: tuple[str, ...]
    pass_fds: tuple[int, ...] = ()
    close_after_spawn: tuple[int, ...] = ()
    env: Mapping[str, str] | None = None
    stdin: int = subprocess.DEVNULL
    stdout: int = subprocess.DEVNULL
    stderr: int = subprocess.DEVNULL


type LaunchFactory = Callable[[int], ChildLaunch]
type IntentKind = Literal[
    "ready", "start", "command", "status", "stop", "close", "delivery"
]


@dataclass(slots=True)
class _Intent:
    kind: IntentKind
    args: tuple[Any, ...] = ()
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class _TerminalDelivery:
    sequence: int
    payload: CorrelatedTerminalDelivery


# A PID can be reused and a supervisor generation restarts at one for a new
# service.  This process-wide sequence gives every successfully spawned Popen a
# distinct ownership identity for the lifetime of PID 1.
_PROCESS_OWNERSHIP_IDENTITIES = count(1)
_CHILD_BOOTSTRAP = (
    "import os,runpy,sys;"
    "expected=int(sys.argv.pop(1));"
    "os.getppid()==expected or os._exit(70);"
    "runpy.run_module('mjlab_microduck.rom.runtime_child',run_name='__main__',alter_sys=True)"
)


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
        terminal_callback: Callable[[CorrelatedTerminalDelivery], None] | None = None,
        terminal_retry_delay_s: float = 0.05,
        terminal_retry_limit: int = 3,
        owner_thread_name: str = "microduck-runtime-supervisor",
        qualification_max_steps: int | None = None,
    ) -> None:
        if (
            min(operation_timeout_s, terminate_timeout_s, terminal_retry_delay_s) <= 0
            or queue_size <= 0
            or terminal_retry_limit <= 0
        ):
            raise ValueError("supervisor bounds must be positive")
        if qualification_max_steps is not None and (
            isinstance(qualification_max_steps, bool)
            or not 100 <= qualification_max_steps <= 2_000
        ):
            raise ValueError("qualification step bound is invalid")
        self._bundle_root = str(bundle_root)
        self._bundle_digest = bundle_digest
        self._launch_factory = launch_factory or self._default_launch
        self._operation_timeout = operation_timeout_s
        self._terminate_timeout = terminate_timeout_s
        self._terminal_callback = terminal_callback
        self._terminal_retry_delay = terminal_retry_delay_s
        self._terminal_retry_limit = terminal_retry_limit
        self._qualification_max_steps = qualification_max_steps
        # One callback, one queued delivery, and one owner-held pending delivery
        # bound the number of acknowledgements that can race owner scheduling.
        self._terminal_ack_queue: queue.Queue[tuple[int, bool]] = queue.Queue(maxsize=3)
        self._owner_shutdown = threading.Event()
        self._terminal_queue: queue.Queue[_TerminalDelivery | None] | None = None
        self._terminal_thread: threading.Thread | None = None
        self._terminal_shutdown_sent = False
        if terminal_callback is not None:
            self._terminal_queue = queue.Queue(maxsize=1)
            self._terminal_thread = threading.Thread(
                target=self._deliver_terminals,
                name=f"microduck-terminal-delivery-{id(self):x}",
                daemon=True,
            )
            self._terminal_thread.start()
        self._queue: queue.Queue[_Intent] = queue.Queue(maxsize=queue_size)
        self._snapshot_lock = threading.Lock()
        self._trace_lock = threading.Lock()
        self._snapshot_value = SupervisorSnapshot(
            SupervisorState.NO_CHILD, 0, False, None, None, True
        )
        self._process: subprocess.Popen[bytes] | None = None
        self._socket: socket.socket | None = None
        self._generation = 0
        self._sequence = 0
        self._active_task: str | None = None
        self._last_event_sequence = 0
        self._ownership_identity: int | None = None
        self._reap_receipt: ReapReceipt | None = None
        self._terminal_delivery_sequence = 0
        self._terminal_delivery_outstanding: int | None = None
        self._pending_terminal_delivery: _TerminalDelivery | None = None
        self._outstanding_terminal_delivery: _TerminalDelivery | None = None
        self._terminal_delivery_attempts = 0
        self._terminal_retry_not_before = 0.0
        self._trace: list[str] = []
        self._closed = False
        self._closing = threading.Event()
        self._close_lock = threading.Lock()
        self._submission_lock = threading.Lock()
        self._close_intent: _Intent | None = None
        self._thread = threading.Thread(
            target=self._run, name=owner_thread_name, daemon=True
        )
        self._thread.start()

    def _deliver_terminals(self) -> None:
        assert self._terminal_queue is not None and self._terminal_callback is not None
        while True:
            delivery = self._terminal_queue.get()
            if delivery is None:
                return
            try:
                self._terminal_callback(delivery.payload)
            except Exception as exc:  # noqa: BLE001 - callback boundary is isolated
                self._record(f"TERMINAL_DELIVERY_FAILED:{type(exc).__name__}")
                success = False
            else:
                success = True
            if self._owner_shutdown.is_set():
                return
            try:
                self._terminal_ack_queue.put_nowait((delivery.sequence, success))
            except queue.Full:
                self._record("TERMINAL_ACK_CAPACITY_VIOLATION")
                return

    def _default_launch(self, socket_fd: int) -> ChildLaunch:
        allowed = {
            "HOME",
            "LANG",
            "LC_ALL",
            "LD_LIBRARY_PATH",
            "MUJOCO_GL",
            "OMP_NUM_THREADS",
            "PATH",
        }
        argv = (
                "/usr/bin/setpriv",
                "--pdeathsig",
                "SIGTERM",
                sys.executable,
                "-P",
                "-c",
                _CHILD_BOOTSTRAP,
                str(os.getpid()),
                "--socket-fd",
                str(socket_fd),
                "--expected-parent-pid",
                str(os.getpid()),
                "--bundle-root",
                self._bundle_root,
            )
        if self._qualification_max_steps is not None:
            argv += (
                "--qualification-max-steps",
                str(self._qualification_max_steps),
            )
        return ChildLaunch(
            argv,
            env={key: value for key, value in os.environ.items() if key in allowed},
        )

    @property
    def trace(self) -> tuple[str, ...]:
        with self._trace_lock:
            return tuple(self._trace)

    def _record(self, value: str) -> None:
        with self._trace_lock:
            self._trace.append(value)

    def snapshot(self) -> SupervisorSnapshot:
        with self._snapshot_lock:
            return self._snapshot_value

    def readiness(self) -> bool:
        snap = self.snapshot()
        return (
            snap.child_healthy
            and snap.slot_releasable
            and snap.state
            in {
                SupervisorState.IDLE,
                SupervisorState.RUNNING,
            }
        )

    @property
    def terminal_delivery_alive(self) -> bool:
        """Whether the optional callback worker still owns execution resources."""
        return self._terminal_thread is not None and self._terminal_thread.is_alive()

    def ensure_ready(self) -> SupervisorSnapshot:
        return self._submit("ready")

    def start(
        self,
        request: TaskCreateRequest,
        register_acknowledgement: Callable[[StartAcknowledgement], None] | None = None,
        register_dispatch: Callable[[], None] | None = None,
    ) -> StartAcknowledgement:
        return self._submit(
            "start", request, register_acknowledgement, register_dispatch
        )

    def command(
        self,
        task_id: str,
        command: TaskCommandRequest | Mapping[str, object],
        lease_ms: int | None = None,
    ) -> AckPayload:
        return self._submit("command", task_id, command, lease_ms)

    def status(self, task_id: str) -> RobotStatus:
        return self._submit("status", task_id)

    def stop(
        self,
        task_id: str,
        reason: str,
        register_dispatch: Callable[[], None] | None = None,
    ) -> TerminalPayload:
        return self._submit("stop", task_id, reason, register_dispatch)

    def close(self) -> ReapReceipt | None:
        if (
            self._closed
            and not self._thread.is_alive()
            and not self.terminal_delivery_alive
        ):
            return None
        receipt: ReapReceipt | None = None
        if self._thread.is_alive():
            receipt = self._claim_close()
            self._thread.join(self._operation_timeout + self._terminate_timeout + 1)
            if self._thread.is_alive() or self.snapshot().pid is not None:
                raise SupervisorUnavailable(
                    "close did not prove owner termination and exact reap"
                )
        self._shutdown_terminal_delivery()
        return receipt

    def _claim_close(self) -> ReapReceipt | None:
        with self._submission_lock, self._close_lock:
            intent = self._close_intent
            if intent is None:
                intent = _Intent(kind="close")
                self._close_intent = intent
                self._closing.set()
        wait_bound = self._operation_timeout * 3 + self._terminate_timeout * 2 + 1
        if not intent.done.wait(wait_bound):
            raise SupervisorUnavailable("close could not reach the process owner")
        if intent.error is not None:
            raise SupervisorUnavailable(
                "close process containment failed"
            ) from intent.error
        if intent.result is not None and not isinstance(intent.result, ReapReceipt):
            raise SupervisorUnavailable("close returned invalid exact-reap evidence")
        return intent.result

    def _shutdown_terminal_delivery(self) -> None:
        thread = self._terminal_thread
        terminal_queue = self._terminal_queue
        if thread is None or terminal_queue is None or not thread.is_alive():
            return
        if not self._terminal_shutdown_sent:
            try:
                terminal_queue.put(None, timeout=self._terminate_timeout)
            except queue.Full:
                self._record("TERMINAL_WORKER_SENTINEL_BACKPRESSURE")
            else:
                self._terminal_shutdown_sent = True
        thread.join(self._terminate_timeout)
        if thread.is_alive():
            self._record("TERMINAL_WORKER_ABANDONED")
            raise SupervisorUnavailable(
                "child contained but terminal delivery worker did not terminate"
            )
        self._record("TERMINAL_WORKER_TERMINATED")

    def _submit(self, kind: IntentKind, *args: object) -> Any:
        with self._submission_lock:
            if (self._closed or self._closing.is_set()) and kind != "close":
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
        terminal: CorrelatedTerminalDelivery | None | object = ...,
        delivery_outstanding: bool | None = None,
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
            terminal_delivery_outstanding=(
                old.terminal_delivery_outstanding
                if delivery_outstanding is None
                else delivery_outstanding
            ),
            cached_terminal=old.cached_terminal if terminal is ... else terminal,  # type: ignore[arg-type]
            pid=self._process.pid if self._process is not None else None,
            ownership_identity=(
                self._ownership_identity if self._process is not None else None
            ),
        )
        with self._snapshot_lock:
            self._snapshot_value = new

    def _advance(self, event: SupervisorEvent, **publish: Any) -> None:
        """Apply Task 1's total transition function to every normal lifecycle edge."""
        decision = transition(self.snapshot().state, event)
        self._publish(decision.next_state, **publish)

    def _run(self) -> None:
        while True:
            self._drain_terminal_acks()
            if self._closing.is_set():
                intent = self._close_intent
                assert intent is not None
                try:
                    intent.result = self._dispatch(intent)
                except Exception as exc:  # noqa: BLE001 - close wakes its owner
                    intent.error = exc
                finally:
                    intent.done.set()
                    self._fail_queued_intents()
                    self._owner_shutdown.set()
                return
            # Terminal durability has priority over public traffic once close is
            # ruled out. The monotonic retry deadline prevents hot retry loops.
            self._flush_pending_terminal()
            try:
                intent = self._queue.get(timeout=0.01)
            except queue.Empty:
                self._poll_unsolicited()
                continue
            try:
                intent.result = self._dispatch(intent)
            except Exception as exc:  # noqa: BLE001 - owner must wake callers
                intent.error = exc
            finally:
                intent.done.set()

    def _fail_queued_intents(self) -> None:
        while True:
            try:
                intent = self._queue.get_nowait()
            except queue.Empty:
                return
            intent.error = SupervisorUnavailable("supervisor close claimed ownership")
            intent.done.set()

    def _drain_terminal_acks(self) -> None:
        while True:
            try:
                sequence, success = self._terminal_ack_queue.get_nowait()
            except queue.Empty:
                return
            self._complete_terminal_delivery(sequence, success)

    def _dispatch(self, intent: _Intent) -> Any:
        if intent.kind == "ready":
            if self._process is None:
                self._spawn()
            if (
                self.snapshot().state is not SupervisorState.IDLE
                or not self.snapshot().slot_releasable
            ):
                raise SupervisorUnavailable("runtime child is not idle")
            return self.snapshot()
        if intent.kind == "close":
            self._closed = True
            self._close_owned_child()
            return self._reap_receipt
        if intent.kind == "delivery":
            self._complete_terminal_delivery(*intent.args)
            return None
        if self._process is None:
            self._spawn()
        if intent.kind == "start":
            if self._process is not None and self._process.poll() is not None:
                self._record("IDLE_CHILD_EXITED")
                self._terminate_and_reap()
                self._spawn()
            return self._start(intent.args[0], intent.args[1], intent.args[2])
        if intent.kind == "command":
            return self._command(*intent.args)
        if intent.kind == "status":
            return self._status(intent.args[0])
        if intent.kind == "stop":
            return self._stop(intent.args[0], intent.args[1], intent.args[2])
        raise AssertionError(intent.kind)

    def _spawn(self) -> None:
        self._generation += 1
        self._sequence = 0
        # No reap evidence may cross a spawn boundary, even if the kernel later
        # assigns the new process the same numeric PID.
        self._reap_receipt = None
        self._ownership_identity = None
        self._advance(
            SupervisorEvent.SPAWN_REQUESTED, healthy=False, reason=None, slot=False
        )
        parent: socket.socket | None = None
        child: socket.socket | None = None
        launch: ChildLaunch | None = None
        try:
            parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            launch = self._launch_factory(child.fileno())
            pass_fds = tuple(dict.fromkeys((child.fileno(), *launch.pass_fds)))
            self._process = subprocess.Popen(
                list(launch.argv),
                pass_fds=pass_fds,
                env=launch.env,
                close_fds=True,
                stdin=launch.stdin,
                stdout=launch.stdout,
                stderr=launch.stderr,
            )
            self._ownership_identity = next(_PROCESS_OWNERSHIP_IDENTITIES)
        except BaseException:
            if parent is not None:
                parent.close()
            if child is not None:
                child.close()
            self._publish(SupervisorState.NO_CHILD, healthy=False, slot=True)
            raise
        finally:
            for inherited_fd in launch.close_after_spawn if launch else ():
                try:
                    os.close(inherited_fd)
                except OSError:
                    continue
        assert child is not None and parent is not None
        child.close()
        self._socket = parent
        # Publish the exact owned PID before readiness exchange so diagnostics and
        # containment proofs can bind to the process even while it is SPAWNING.
        self._publish(SupervisorState.SPAWNING, healthy=False, slot=False)
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
        except Exception as exc:
            self._record(
                "OPERATION_TIMEOUT"
                if isinstance(exc, TimeoutError)
                else "OPERATION_FAILED"
            )
            self._quarantine("SPAWN_FAILED")
            raise SupervisorOperationError("child readiness failed") from exc
        self._advance(
            SupervisorEvent.READY_RECEIVED, healthy=True, reason=None, slot=True
        )

    def _start(
        self,
        request: TaskCreateRequest,
        register_acknowledgement: Callable[[StartAcknowledgement], None] | None,
        register_dispatch: Callable[[], None] | None,
    ) -> StartAcknowledgement:
        if (
            self.snapshot().state is not SupervisorState.IDLE
            or not self.snapshot().slot_releasable
        ):
            raise SupervisorUnavailable("motion slot is unavailable")
        self._active_task = request.taskId
        self._last_event_sequence = 0
        payload = StartPayload(
            actionCode=request.actionCode,
            bundleDigest=request.bundleDigest,
            parameters=request.parameters,
            scenario=request.scenario,
            leaseMs=request.leaseMs,
        )
        def dispatched() -> None:
            self._advance(SupervisorEvent.START_SENT, slot=False, terminal=None)
            if register_dispatch is not None:
                register_dispatch()

        response = self._guarded_exchange(
            RuntimeMessageKind.START,
            request.taskId,
            payload,
            {RuntimeMessageKind.ACK},
            on_sent=dispatched,
        )
        assert isinstance(response.payload, AckPayload)
        if response.payload.acknowledgedKind.value != "START":
            self._quarantine("START_ACK_MISMATCH")
            raise SupervisorOperationError("wrong START acknowledgment")
        result = StartAcknowledgement(
            generation=self._generation,
            task_id=request.taskId,
            acknowledgement=response.payload,
        )
        try:
            if register_acknowledgement is not None:
                register_acknowledgement(result)
        except Exception as exc:
            self._record("START_REGISTRATION_FAILED")
            self._quarantine("START_REGISTRATION_FAILED", protocol_usable=True)
            raise SupervisorOperationError(
                "START acknowledgement registration failed closed"
            ) from exc
        self._advance(SupervisorEvent.START_ACK, slot=False)
        return result

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
        if response.payload.acknowledgedKind.value != "COMMAND":
            self._quarantine("COMMAND_ACK_MISMATCH")
            raise SupervisorOperationError("wrong COMMAND acknowledgment")
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

    def _stop(
        self,
        task_id: str,
        reason: str,
        register_dispatch: Callable[[], None] | None,
    ) -> TerminalPayload:
        self._require_active(task_id)

        def dispatched() -> None:
            self._advance(SupervisorEvent.STOP_CLAIMED, slot=False)
            if register_dispatch is not None:
                register_dispatch()

        response = self._guarded_exchange(
            RuntimeMessageKind.ZERO_AND_STOP,
            task_id,
            ZeroAndStopPayload(reason=reason),
            {RuntimeMessageKind.TERMINAL},
            on_sent=dispatched,
        )
        assert isinstance(response.payload, TerminalPayload)
        delivery = CorrelatedTerminalDelivery(
            self._generation, task_id, 0, response.payload
        )
        self._active_task = None
        delivery_required = self._terminal_queue is not None
        self._advance(
            SupervisorEvent.TERMINAL_ACK,
            healthy=True,
            slot=not delivery_required,
            terminal=delivery,
            delivery_outstanding=delivery_required,
        )
        self._queue_terminal_delivery(delivery)
        return response.payload

    def _require_active(self, task_id: str) -> None:
        if (
            self.snapshot().state is not SupervisorState.RUNNING
            or task_id != self._active_task
        ):
            raise SupervisorUnavailable("task does not own the runtime")

    def _guarded_exchange(self, *args: Any, **kwargs: Any) -> RuntimeMessage:
        try:
            return self._exchange(*args, **kwargs)
        except SupervisorTaskTerminalized:
            raise
        except BaseException as exc:
            self._record(
                "OPERATION_TIMEOUT"
                if isinstance(exc, TimeoutError)
                else "OPERATION_FAILED"
            )
            trustworthy = isinstance(exc, TimeoutError) and self._process is not None
            reason = (
                "OPERATION_TIMEOUT" if isinstance(exc, TimeoutError) else "OPERATION_FAILED"
            )
            self._quarantine(reason, protocol_usable=trustworthy)
            raise SupervisorOperationError("runtime operation failed closed") from exc

    def _exchange(
        self,
        kind: RuntimeMessageKind,
        task_id: str | None,
        payload: object,
        expected: set[RuntimeMessageKind],
        *,
        on_sent: Callable[[], None] | None = None,
    ) -> RuntimeMessage:
        if self._socket is None or self._process is None:
            raise SupervisorUnavailable("no owned child")
        self._consume_prequeued_terminal()
        self._sequence += 1
        request = RuntimeMessage(
            kind=kind,
            generation=self._generation,
            operationSequence=self._sequence,
            taskId=task_id,
            payload=payload,
        )
        try:
            self._socket.sendall(encode_packet(request))
        except OSError:
            # The peer can exit immediately after publishing a terminal.  A
            # failed write must not discard that already-queued authoritative
            # packet and downgrade truthful completion to a runtime failure.
            self._consume_prequeued_terminal()
            raise
        if on_sent is not None:
            on_sent()
        deadline = time.monotonic() + self._operation_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("child operation deadline")
            readable, _, _ = select.select([self._socket], [], [], min(remaining, 0.05))
            if not readable:
                if self._process.poll() is not None:
                    raise ChildProcessError("child exited before response")
                continue
            packet = receive_packet(self._socket)
            if not packet:
                raise ConnectionError("child transport closed")
            response = decode_packet(packet)
            if response.kind is RuntimeMessageKind.TERMINAL_EVENT:
                self._accept_terminal_event(response)
                raise SupervisorTaskTerminalized(
                    "task terminalized while operation was pending"
                )
            if (
                response.generation != self._generation
                or response.operationSequence != self._sequence
                or response.taskId != task_id
                or response.kind not in expected
            ):
                raise SupervisorOperationError("ambiguous child response")
            return response

    def _consume_prequeued_terminal(self) -> None:
        """Accept one terminal queued before a new synchronous request."""
        owned_socket = self._socket
        if owned_socket is None:
            return
        readable, _, _ = select.select([owned_socket], [], [], 0)
        if not readable:
            return
        packet = receive_packet(owned_socket)
        if not packet:
            raise ConnectionError("child transport closed")
        response = decode_packet(packet)
        if response.kind is not RuntimeMessageKind.TERMINAL_EVENT:
            raise SupervisorOperationError("unexpected prequeued response")
        self._accept_terminal_event(response)
        raise SupervisorTaskTerminalized(
            "task terminalized before operation was submitted"
        )

    def _poll_unsolicited(self) -> None:
        owned_socket = self._socket
        process = self._process
        if owned_socket is None or process is None:
            return
        # A process may exit immediately after writing its terminal packet.  The
        # socket is the authoritative ordering boundary, so drain one queued
        # packet before consulting ``poll()``.  Reversing these checks discards
        # a truthful terminal and can also leave a post-START exit cached as
        # RUNNING forever.
        readable, _, _ = select.select([owned_socket], [], [], 0)
        if not readable:
            if process.poll() is not None:
                self._record("OPERATION_FAILED")
                self._quarantine("CHILD_EXITED_WITHOUT_TERMINAL")
            return
        try:
            packet = receive_packet(owned_socket)
            if not packet:
                # A lease-expiry terminal is the child’s final packet: the
                # safety path closes its control socket immediately afterward.
                # EOF is therefore an ordered containment boundary, not an
                # unsolicited protocol response, once a correlated terminal
                # has already been accepted.
                cached_terminal = self.snapshot().cached_terminal
                if (
                    cached_terminal is not None
                    and cached_terminal.generation == self._generation
                ):
                    self._record("CHILD_EOF_AFTER_TERMINAL")
                    self._quarantine("CHILD_EXITED_AFTER_TERMINAL")
                    return
                raise ConnectionError("child transport closed")
            message = decode_packet(packet)
            if message.kind is not RuntimeMessageKind.TERMINAL_EVENT:
                raise SupervisorOperationError("unexpected unsolicited response")
            self._accept_terminal_event(message)
        except Exception:  # noqa: BLE001 - any invalid peer behavior quarantines
            self._record("OPERATION_FAILED")
            self._quarantine("UNSOLICITED_PROTOCOL_FAILURE")
            return
        if process.poll() is not None:
            # The terminal is already queued for durable delivery.  Contain and
            # reap the exact exited PID without downgrading that terminal.
            self._quarantine("CHILD_EXITED_AFTER_TERMINAL")

    def _accept_terminal_event(self, message: RuntimeMessage) -> None:
        payload = message.payload
        if not isinstance(payload, TerminalEventPayload):
            raise SupervisorOperationError("malformed terminal event")
        if (
            self.snapshot().state is not SupervisorState.RUNNING
            or message.generation != self._generation
            or message.taskId != self._active_task
            or message.operationSequence != 0
            or payload.eventSequence != self._last_event_sequence + 1
        ):
            raise SupervisorOperationError("stale or replayed terminal event")
        terminal = payload.terminal
        assert message.taskId is not None
        delivery = CorrelatedTerminalDelivery(
            message.generation, message.taskId, payload.eventSequence, terminal
        )
        self._last_event_sequence = payload.eventSequence
        self._active_task = None
        delivery_required = self._terminal_queue is not None
        self._advance(
            SupervisorEvent.TERMINAL_ACK,
            healthy=True,
            slot=not delivery_required,
            terminal=delivery,
            delivery_outstanding=delivery_required,
        )
        self._queue_terminal_delivery(delivery)

    def _queue_terminal_delivery(self, terminal: CorrelatedTerminalDelivery) -> None:
        if self._terminal_queue is None:
            return
        self._terminal_delivery_sequence += 1
        delivery = _TerminalDelivery(self._terminal_delivery_sequence, terminal)
        self._terminal_delivery_outstanding = delivery.sequence
        self._outstanding_terminal_delivery = delivery
        self._terminal_delivery_attempts = 0
        self._terminal_retry_not_before = 0.0
        self._pending_terminal_delivery = delivery
        self._flush_pending_terminal()

    def _flush_pending_terminal(self) -> None:
        delivery = self._pending_terminal_delivery
        terminal_queue = self._terminal_queue
        if delivery is None or terminal_queue is None:
            return
        if time.monotonic() < self._terminal_retry_not_before:
            return
        try:
            terminal_queue.put_nowait(delivery)
        except queue.Full:
            return
        self._pending_terminal_delivery = None
        self._terminal_delivery_attempts += 1

    def _complete_terminal_delivery(self, sequence: int, success: bool) -> None:
        if sequence != self._terminal_delivery_outstanding:
            self._flush_pending_terminal()
            return
        if not success:
            self._record("TERMINAL_DELIVERY_RETRY")
            delivery = self._outstanding_terminal_delivery
            if (
                delivery is None
                or self._terminal_delivery_attempts >= self._terminal_retry_limit
            ):
                self._record("TERMINAL_DELIVERY_PERMANENT_FAILURE")
                return
            self._pending_terminal_delivery = delivery
            self._terminal_retry_not_before = (
                time.monotonic() + self._terminal_retry_delay
            )
            return
        self._terminal_delivery_outstanding = None
        self._outstanding_terminal_delivery = None
        self._pending_terminal_delivery = None
        self._publish(self.snapshot().state, slot=True, delivery_outstanding=False)

    def _quarantine(self, reason: str, *, protocol_usable: bool = False) -> None:
        self._advance(
            SupervisorEvent.OPERATION_TIMEOUT, healthy=False, reason=reason, slot=False
        )
        self._record("QUARANTINED")
        if protocol_usable and self._active_task is not None:
            self._best_effort_zero()
        self._terminate_and_reap()

    def _best_effort_zero(self) -> None:
        """Attempt one independently bounded stop without trusting it as release proof."""
        if (
            self._socket is None
            or self._process is None
            or self._process.poll() is not None
        ):
            return
        self._sequence += 1
        request = RuntimeMessage(
            kind=RuntimeMessageKind.ZERO_AND_STOP,
            generation=self._generation,
            operationSequence=self._sequence,
            taskId=self._active_task,
            payload=ZeroAndStopPayload(reason="RUNTIME_UNRESPONSIVE"),
        )
        try:
            self._socket.sendall(encode_packet(request))
            readable, _, _ = select.select(
                [self._socket], [], [], min(self._terminate_timeout, 0.1)
            )
            if readable:
                packet = receive_packet(self._socket)
                response = decode_packet(packet)
                if (
                    response.kind is RuntimeMessageKind.TERMINAL
                    and response.generation == self._generation
                    and response.operationSequence == self._sequence
                    and response.taskId == self._active_task
                ):
                    self._record("BEST_EFFORT_STOP_ACK")
        except (OSError, ValueError):
            return

    def _terminate_and_reap(self) -> None:
        process = self._process
        owned_socket = self._socket
        ownership_identity = self._ownership_identity
        if process is None:
            self._publish(SupervisorState.NO_CHILD, healthy=False, slot=True)
            return
        self._advance(SupervisorEvent.TERMINATION_CLAIMED, slot=False)
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            self._record("SIGTERM_SENT")
            try:
                process.wait(timeout=self._terminate_timeout)
            except subprocess.TimeoutExpired:
                self._record("TERM_TIMEOUT")
                self._advance(SupervisorEvent.TERM_TIMEOUT, slot=False)
                process.kill()
                self._record("SIGKILL_SENT")
                self._advance(SupervisorEvent.SIGKILL_SENT, slot=False)
        if self.snapshot().state is SupervisorState.TERMINATING:
            self._advance(SupervisorEvent.CHILD_EXITED, slot=False)
        process.wait(timeout=max(self._operation_timeout, 0.1))
        if process.poll() is None:
            raise RuntimeError("exact child reap was not confirmed")
        if ownership_identity is None:
            raise RuntimeError("owned process identity was lost before exact reap")
        self._reap_receipt = ReapReceipt(
            generation=self._generation,
            pid=process.pid,
            ownership_identity=ownership_identity,
        )
        self._record("CHILD_REAPED")
        if owned_socket is not None:
            owned_socket.close()
        self._process = None
        self._socket = None
        self._ownership_identity = None
        self._active_task = None
        self._advance(
            SupervisorEvent.CHILD_REAPED,
            healthy=False,
            slot=not self.snapshot().terminal_delivery_outstanding,
        )
        self._record("NO_CHILD")

    def _close_owned_child(self) -> None:
        if self._process is None:
            self._publish(SupervisorState.NO_CHILD, healthy=False, slot=True)
            return
        if self.snapshot().state is SupervisorState.IDLE:
            try:
                response = self._exchange(
                    RuntimeMessageKind.SHUTDOWN,
                    None,
                    ShutdownPayload(reason="SUPERVISOR_SHUTDOWN"),
                    {RuntimeMessageKind.ACK},
                )
                assert isinstance(response.payload, AckPayload)
                if response.payload.acknowledgedKind.value != "SHUTDOWN":
                    raise SupervisorOperationError("wrong SHUTDOWN acknowledgment")
            except Exception as exc:  # noqa: BLE001 - shutdown escalates below
                self._record(f"SHUTDOWN_FAILED:{type(exc).__name__}")
        self._terminate_and_reap()
