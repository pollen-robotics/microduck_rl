"""Durable task service backed exclusively by the isolated runtime supervisor."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import Event, Lock
from typing import Any, Protocol

from .action_catalog import (
    action_template,
    code_owned_action_definition,
    validate_action_definition_envelope,
    validate_bundle_action_envelope,
)
from .contracts import (
    ActionDefinition,
    PolicyBundle,
    RobotStatus,
    TaskCommandRequest,
    TaskCreateRequest,
    TaskEvidence,
    sha256_prefixed,
)
from .process_protocol import AckPayload, TerminalPayload
from .process_supervisor import (
    CorrelatedTerminalDelivery,
    StartAcknowledgement,
    SupervisorOperationError,
    SupervisorSnapshot,
    SupervisorTaskTerminalized,
    SupervisorUnavailable,
)
from .store import CommandSequenceConflict as StoreCommandSequenceConflict
from .store import IllegalTaskTransition, SqliteTaskStore, TaskIdConflict
from .store import StaleCommand as StoreStaleCommand


class SimulatorServiceError(ValueError):
    code = "INTERNAL_ERROR"


class BundleMismatch(SimulatorServiceError):
    code = "BUNDLE_MISMATCH"


class ActionUnavailable(SimulatorServiceError):
    code = "ACTION_UNAVAILABLE"


class InvalidParameters(SimulatorServiceError):
    code = "PARAMETER_INVALID"


class PreconditionFailed(SimulatorServiceError):
    code = "PRECONDITION_FAILED"


class NotReady(SimulatorServiceError):
    code = "NOT_READY"


class RuntimeException(SimulatorServiceError):
    code = "RUNTIME_EXCEPTION"


class RobotBusy(SimulatorServiceError):
    code = "ROBOT_BUSY"


class TaskNotFound(SimulatorServiceError):
    code = "TASK_NOT_FOUND"


class TaskConflict(SimulatorServiceError):
    code = "TASK_ID_CONFLICT"


class CommandSequenceConflict(SimulatorServiceError):
    code = "COMMAND_SEQUENCE_CONFLICT"


class StaleCommand(SimulatorServiceError):
    code = "STALE_COMMAND"


class ProcessSupervisor(Protocol):
    def ensure_ready(self) -> SupervisorSnapshot: ...
    def snapshot(self) -> SupervisorSnapshot: ...
    def readiness(self) -> bool: ...
    def start(
        self,
        request: TaskCreateRequest,
        register_acknowledgement: Callable[[StartAcknowledgement], None] | None = None,
        register_dispatch: Callable[[], None] | None = None,
    ) -> StartAcknowledgement: ...
    def command(self, task_id: str, command: TaskCommandRequest) -> AckPayload: ...
    def status(self, task_id: str) -> RobotStatus: ...
    def stop(
        self,
        task_id: str,
        reason: str,
        register_dispatch: Callable[[], None] | None = None,
    ) -> TerminalPayload: ...
    def close(self) -> None: ...


type SupervisorFactory = Callable[
    [Callable[[CorrelatedTerminalDelivery], None]], ProcessSupervisor
]
TERMINAL = frozenset({"SUCCEEDED", "FAILED", "CANCELLED", "TIMED_OUT", "UNKNOWN"})


@dataclass(slots=True)
class _PendingCommand:
    sequence: int
    digest: str
    done: Event
    result: Any = None
    error: BaseException | None = None


@dataclass(slots=True)
class _ActiveTask:
    generation: int
    request: TaskCreateRequest
    action: ActionDefinition
    continuous: bool
    stop_claimed: bool = False
    deadline: float | None = None
    supervisor_generation: int | None = None
    pending_command: _PendingCommand | None = None
    latest_command_sequence: int | None = None
    latest_command_digest: str | None = None
    latest_command_result: Any = None


class SimulatorTaskService:
    """Own durable identity only; all simulator resources remain in the child."""

    def __init__(
        self,
        bundle: PolicyBundle,
        store: SqliteTaskStore,
        supervisor_factory: SupervisorFactory,
        *,
        pollIntervalS: float = 0.05,
        runtimeCallTimeoutS: float = 0.25,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        validate_bundle_action_envelope(bundle)
        if pollIntervalS <= 0 or not 0 < runtimeCallTimeoutS <= 5:
            raise ValueError("invalid service bound")
        # `pollIntervalS` remains accepted for V1 constructor compatibility;
        # runtime sampling now belongs exclusively to the child process.
        self._bundle, self._store, self._monotonic_clock = (
            bundle,
            store,
            monotonic_clock,
        )
        self._store.mark_interrupted_unknown()
        self._lock, self._active, self._next_generation = Lock(), None, 1
        self._watchdog_healthy, self._readiness_failure_reason = True, None
        # Compatibility argument now bounds duplicate callers waiting for the
        # one supervisor-owned COMMAND acknowledgement.
        self._runtime_call_timeout_s = runtimeCallTimeoutS
        self._supervisor = supervisor_factory(self._terminal_callback)
        self._shutdown_reaped_pid: int | None = None
        try:
            self._supervisor.ensure_ready()
        except Exception:  # noqa: BLE001 - startup diagnostics fail closed.
            self._readiness_failure_reason = "RUNTIME_UNAVAILABLE"

    def create_task(self, request: TaskCreateRequest):
        self._reconcile_reaped_terminal()
        request_hash = sha256_prefixed(request)
        existing = self._store.get(request.taskId)
        if existing is not None:
            return self._create_idempotent(request, request_hash)
        with self._lock:
            if self._active is not None:
                raise RobotBusy("robot already has an active task")
        self._require_motion_ready()
        action = self._validate_request(request)
        self._require_preconditions(action, request)
        with self._lock:
            if self._store.get(request.taskId) is not None:
                return self._create_idempotent(request, request_hash)
            if self._active is not None:
                raise RobotBusy("robot already has an active task")
            try:
                snapshot, created = self._store.create(request, request_hash)
            except TaskIdConflict as exc:
                raise TaskConflict(str(exc)) from exc
            if not created:
                return snapshot
            active = _ActiveTask(
                self._next_generation,
                request,
                action,
                action.executionMode == "CONTINUOUS_LEASE",
            )
            self._next_generation += 1
            self._active = active
            self._store.transition(
                request.taskId, "VALIDATING", event_type="TASK_VALIDATING"
            )
        try:
            registered: StartAcknowledgement | None = None

            def register_acknowledgement(result: StartAcknowledgement) -> None:
                nonlocal registered
                with self._lock:
                    if (
                        self._active is not active
                        or result.task_id != request.taskId
                        or result.generation <= 0
                        or registered is not None
                    ):
                        raise SupervisorOperationError(
                            "ambiguous START acknowledgement identity"
                        )
                    active.supervisor_generation = result.generation
                    registered = result

            start_result = self._supervisor.start(request, register_acknowledgement)
            if start_result != registered:
                with self._lock:
                    active.stop_claimed = True
                self._request_stop(active, "RUNTIME_UNRESPONSIVE")
                raise SupervisorOperationError(
                    "runtime START acknowledgement was not registered atomically"
                )
            deadline = (
                self._monotonic_clock() + request.leaseMs / 1000
                if active.continuous and request.leaseMs
                else None
            )
            with self._lock:
                if self._active is not active:
                    return (
                        snapshot
                        if not active.continuous
                        else self._store.get(request.taskId)
                    )
                if active.stop_claimed:
                    return self._store.get(request.taskId)
                if active.continuous:
                    active.deadline = deadline
                    return self._store.start_continuous(request.taskId, deadline)
                self._store.transition(
                    request.taskId, "RUNNING", event_type="TASK_STARTED"
                )
                return snapshot
        except SupervisorTaskTerminalized:
            return self._store.get(request.taskId)
        except (SupervisorUnavailable, SupervisorOperationError) as exc:
            self._persist_failure(active, "RUNTIME_UNRESPONSIVE")
            raise RuntimeException("could not start simulator runtime") from exc
        except Exception as exc:
            self._persist_failure(active, "RUNTIME_EXCEPTION")
            if not active.continuous:
                return self._store.get(request.taskId)
            raise RuntimeException("could not start simulator runtime") from exc

    def _create_idempotent(self, request, request_hash):
        try:
            return self._store.create(request, request_hash)[0]
        except TaskIdConflict as exc:
            raise TaskConflict(str(exc)) from exc

    def get_task(self, task_id: str):
        result = self._store.get(task_id)
        if result is None:
            raise TaskNotFound(f"task not found: {task_id}")
        return result

    def cancel_task(self, task_id: str):
        with self._lock:
            snapshot = self._store.get(task_id)
            if snapshot is None:
                raise TaskNotFound(f"task not found: {task_id}")
            active = self._active
            if (
                active is None
                or active.request.taskId != task_id
                or snapshot.state in TERMINAL
                or active.stop_claimed
            ):
                return snapshot
            active.stop_claimed = True
        return self._request_stop(active, "CANCELLED")

    def command(self, task_id: str, command: TaskCommandRequest):
        self._require_motion_ready(allow_running=True)
        owner = False
        with self._lock:
            snapshot, active = self._store.get(task_id), self._active
            if snapshot is None:
                raise TaskNotFound(f"task not found: {task_id}")
            if (
                active is None
                or active.request.taskId != task_id
                or not active.continuous
            ):
                raise InvalidParameters("task does not accept continuous commands")
            if snapshot.state != "RUNNING" or active.stop_claimed:
                raise InvalidParameters("task is not running")
            if (
                active.deadline is not None
                and self._monotonic_clock() >= active.deadline
            ):
                active.stop_claimed = True
                expired = True
            else:
                expired = False
            if expired:
                pending = None
            else:
                self._validate_command(command, active.action)
                digest = sha256_prefixed(command)
                if active.pending_command is not None:
                    pending = active.pending_command
                    if command.commandSequence < pending.sequence:
                        raise StaleCommand("command sequence is stale")
                    if command.commandSequence != pending.sequence:
                        raise RuntimeException("another command delivery is pending")
                    if digest != pending.digest:
                        raise CommandSequenceConflict(
                            "command sequence content conflicts"
                        )
                elif active.latest_command_sequence is not None:
                    if command.commandSequence < active.latest_command_sequence:
                        raise StaleCommand("command sequence is stale")
                    if command.commandSequence == active.latest_command_sequence:
                        if digest != active.latest_command_digest:
                            raise CommandSequenceConflict(
                                "command sequence content conflicts"
                            )
                        return active.latest_command_result
                    pending = _PendingCommand(command.commandSequence, digest, Event())
                    active.pending_command = pending
                    owner = True
                else:
                    pending = _PendingCommand(command.commandSequence, digest, Event())
                    active.pending_command = pending
                    owner = True
        if expired:
            self._request_stop(active, "LEASE_EXPIRED")
            raise InvalidParameters("task is not running")
        assert pending is not None
        if not owner:
            if not pending.done.wait(self._runtime_call_timeout_s):
                raise RuntimeException("command acknowledgement is still pending")
            if pending.error is not None:
                raise pending.error
            if pending.result is None:
                raise RuntimeException("command completed without a durable result")
            return pending.result
        containment_failure = False
        try:
            self._supervisor.command(task_id, command)
        except SupervisorTaskTerminalized:
            operation_error: BaseException | None = RuntimeException(
                "task terminalized during command"
            )
        except (SupervisorUnavailable, SupervisorOperationError) as exc:
            operation_error = RuntimeException("simulator command was unresponsive")
            operation_error.__cause__ = exc
            containment_failure = True
        except Exception as exc:  # noqa: BLE001 - normalize the supervisor boundary.
            operation_error = RuntimeException("simulator command was unresponsive")
            operation_error.__cause__ = exc
            containment_failure = True
        else:
            operation_error = None
        with self._lock:
            try:
                if pending.done.is_set():
                    pass
                elif operation_error is not None:
                    pending.error = operation_error
                elif (
                    self._active is not active
                    or active.pending_command is not pending
                    or active.stop_claimed
                    or (current := self._store.get(task_id)) is None
                    or current.state != "RUNNING"
                ):
                    pending.error = RuntimeException("task terminalized during command")
                else:
                    deadline = self._monotonic_clock() + command.leaseMs / 1000
                    accepted, _ = self._store.record_command(
                        task_id, command, digest, deadline
                    )
                    if accepted is None:
                        pending.error = RuntimeException(
                            "command completed without a durable result"
                        )
                    else:
                        pending.result = accepted
                        active.deadline = deadline
                        active.latest_command_sequence = command.commandSequence
                        active.latest_command_digest = digest
                        active.latest_command_result = accepted
            except StoreStaleCommand as exc:
                pending.error = StaleCommand(str(exc))
            except StoreCommandSequenceConflict as exc:
                pending.error = CommandSequenceConflict(str(exc))
            except IllegalTaskTransition:
                pending.error = RuntimeException("task terminalized during command")
            except Exception as exc:  # noqa: BLE001 - all duplicates share failure.
                if not pending.done.is_set():
                    failure = RuntimeException("command durability failed")
                    failure.__cause__ = exc
                    pending.error = failure
                    containment_failure = True
            finally:
                if self._active is active and active.pending_command is pending:
                    active.pending_command = None
                if not pending.done.is_set():
                    pending.done.set()
        if containment_failure:
            try:
                self._request_stop(active, "RUNTIME_UNRESPONSIVE")
                self._persist_failure(active, "RUNTIME_UNRESPONSIVE")
            except Exception as exc:  # noqa: BLE001 - preserve shared public error.
                # The reservation is already completed with the normalized
                # shared error. Keep durable ownership fail-closed if even the
                # containment boundary itself violates its exception contract.
                self._watchdog_healthy = False
                self._readiness_failure_reason = "RUNTIME_UNAVAILABLE"
                if pending.error is not None and pending.error.__cause__ is None:
                    pending.error.__cause__ = exc
        if pending.error is not None:
            raise pending.error
        if pending.result is None:
            raise RuntimeException("command completed without a durable result")
        return pending.result

    def events_after(self, task_id: str, sequence: int, *, page_size: int = 100):
        if (
            not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or not -1 <= sequence <= 2**63 - 1
        ):
            raise InvalidParameters("afterSequence must be a signed 64-bit cursor")
        self.get_task(task_id)
        return self._store.events_after(task_id, sequence, page_size=page_size)

    def robot_status(self) -> RobotStatus:
        snap = self._supervisor.snapshot()
        status = snap.cached_status or _initial_status(ready=snap.child_healthy)
        if snap.child_healthy:
            return status
        reason = (
            snap.quarantine_reason
            or self._readiness_failure_reason
            or "RUNTIME_UNAVAILABLE"
        )
        return status.model_copy(
            update={
                "limp": True,
                "health": dict(status.health)
                | {"ready": False, "healthy": False, "reasonCodes": [reason]},
            }
        )

    def motion_readiness(self) -> tuple[bool, tuple[str, ...]]:
        self._reconcile_reaped_terminal()
        if not self._watchdog_healthy:
            return False, (self._readiness_failure_reason or "WATCHDOG_UNHEALTHY",)
        if self._supervisor.readiness():
            return True, ()
        snap = self._supervisor.snapshot()
        return False, (
            snap.quarantine_reason
            or self._readiness_failure_reason
            or "RUNTIME_UNAVAILABLE",
        )

    def _reconcile_reaped_terminal(self) -> None:
        """Release durable ownership only after the supervisor proves cleanup."""
        snap = self._supervisor.snapshot()
        if not snap.slot_releasable:
            return
        with self._lock:
            active = self._active
            if active is None:
                return
            durable = self._store.get(active.request.taskId)
            if durable is not None and durable.state in TERMINAL:
                self._active = None

    def tick(self) -> None:
        with self._lock:
            active = self._active
            expired = (
                active is not None
                and active.continuous
                and active.deadline is not None
                # The child owns the exact lease deadline and publishes its
                # truthful local-deadman terminal.  The parent containment
                # timer starts only after two bounded acknowledgement windows,
                # avoiding a duplicate ZERO_AND_STOP race at the same instant.
                and self._monotonic_clock()
                >= active.deadline + 2 * self._runtime_call_timeout_s
                and not active.stop_claimed
            )
            if expired:
                active.stop_claimed = True
        if expired:
            self._request_stop(active, "LEASE_EXPIRED")
        # Unsolicited child terminals arrive only through the supervisor's
        # correlated callback. Cached snapshots are diagnostic, never replayed.
        snap = self._supervisor.snapshot()
        if (
            active is not None
            and active.supervisor_generation == snap.generation
            and snap.state.value == "NO_CHILD"
            and not snap.child_healthy
            and snap.slot_releasable
        ):
            current = self._store.get(active.request.taskId)
            if current is not None and current.state not in TERMINAL:
                self._persist_failure(active, "RUNTIME_UNRESPONSIVE")

    def watchdog_failed(self) -> None:
        self._watchdog_healthy, self._readiness_failure_reason = (
            False,
            "WATCHDOG_UNHEALTHY",
        )
        with self._lock:
            active = self._active
            if active is not None:
                active.stop_claimed = True
        if active is not None:
            self._request_stop(active, "WATCHDOG_FAILURE")

    def close(self) -> None:
        owned_pid = self._supervisor.snapshot().pid
        try:
            self._supervisor.close()
        finally:
            if (
                owned_pid is not None
                and getattr(self._supervisor, "last_reaped_pid", None) == owned_pid
            ):
                self._shutdown_reaped_pid = owned_pid

    @property
    def shutdown_reaped_pid(self) -> int | None:
        """Expose only a PID whose exact owned process crossed the reap barrier."""
        return self._shutdown_reaped_pid

    def _request_stop(self, active: _ActiveTask, reason: str):
        dispatch_callback: Callable[[], None] | None = None
        if reason == "CANCELLED":

            def record_cancel_dispatch() -> None:
                with self._lock:
                    self._store.append_event(
                        active.request.taskId,
                        "TASK_CANCEL_REQUESTED",
                        {"code": "CANCELLED"},
                    )

            dispatch_callback = record_cancel_dispatch
        try:
            terminal = self._supervisor.stop(
                active.request.taskId, reason, dispatch_callback
            )
            self._terminal_callback(
                CorrelatedTerminalDelivery(
                    generation=self._supervisor.snapshot().generation,
                    task_id=active.request.taskId,
                    event_sequence=0,
                    terminal=terminal,
                )
            )
        except SupervisorTaskTerminalized:
            pass
        except SupervisorUnavailable:
            snap = self._supervisor.snapshot()
            queued = snap.cached_terminal
            if not (
                snap.terminal_delivery_outstanding
                and queued is not None
                and queued.generation == active.supervisor_generation
                and queued.task_id == active.request.taskId
            ):
                self._persist_failure(active, "RUNTIME_UNRESPONSIVE")
        except SupervisorOperationError:
            self._persist_failure(active, "RUNTIME_UNRESPONSIVE")
        return self._store.get(active.request.taskId)

    def _terminal_callback(self, delivery: CorrelatedTerminalDelivery) -> None:
        pending: _PendingCommand | None = None
        with self._lock:
            active = self._active
            if active is None:
                return
            if (
                delivery.task_id != active.request.taskId
                or delivery.generation != active.supervisor_generation
            ):
                raise RuntimeError("terminal delivery does not match active generation")
            terminal = delivery.terminal
            current = self._store.get(active.request.taskId)
            if current is None:
                raise RuntimeError("active durable task is missing")
            if current.state in TERMINAL:
                pending = active.pending_command
                if pending is not None and not pending.done.is_set():
                    pending.error = RuntimeException(
                        "task terminalized during command"
                    )
                    active.pending_command = None
                    pending.done.set()
                self._active = None
                return
            reason = terminal.evidence.stopReason or (
                "TASK_COMPLETE" if terminal.outcome == "SUCCEEDED" else "RUNTIME_FAILED"
            )
            self._advance_running(active.request.taskId)
            self._store.transition(
                active.request.taskId,
                terminal.outcome,
                event_type=f"TASK_{terminal.outcome}",
                payload={"code": reason},
                evidence=terminal.evidence,
                stop_reason=reason,
            )
            pending = active.pending_command
            if pending is not None and not pending.done.is_set():
                pending.error = RuntimeException("task terminalized during command")
                active.pending_command = None
            self._active = None
            if pending is not None:
                pending.done.set()

    def _persist_failure(self, active: _ActiveTask, reason: str) -> None:
        with self._lock:
            if self._active is not active:
                return
            policy = next(
                item
                for item in self._bundle.policies
                if item.policyRef == active.action.policyRef
            )
            evidence = TaskEvidence(
                bundleDigest=self._bundle.bundleDigest,
                policyDigest=policy.digest,
                modelDigest=self._bundle.model.digest,
                metrics={"safetyFailure": reason},
                stopReason=reason,
            )
            self._store.transition(
                active.request.taskId,
                "FAILED",
                event_type="TASK_FAILED",
                payload={"code": reason},
                evidence=evidence,
                stop_reason=reason,
            )
            if self._supervisor.snapshot().slot_releasable:
                self._active = None

    def _advance_running(self, task_id: str) -> None:
        current = self._store.get(task_id)
        if current is not None and current.state == "VALIDATING":
            try:
                self._store.transition(task_id, "RUNNING", event_type="TASK_STARTED")
            except IllegalTaskTransition:
                pass

    def _require_motion_ready(self, *, allow_running: bool = False) -> None:
        snap = self._supervisor.snapshot()
        if allow_running and snap.child_healthy and snap.state.value == "RUNNING":
            return
        if snap.state.value == "NO_CHILD" and snap.slot_releasable:
            try:
                self._supervisor.ensure_ready()
            except (SupervisorUnavailable, SupervisorOperationError) as exc:
                raise NotReady("simulator is not ready for motion") from exc
        if not self.motion_readiness()[0]:
            raise NotReady("simulator is not ready for motion")

    def _validate_request(self, request: TaskCreateRequest) -> ActionDefinition:
        if (
            request.bundleVersion != self._bundle.bundleVersion
            or request.bundleDigest != self._bundle.bundleDigest
        ):
            raise BundleMismatch("requested bundle does not match installed bundle")
        action = next(
            (x for x in self._bundle.actions if x.actionCode == request.actionCode),
            None,
        )
        if action is None or action.availability != "AVAILABLE":
            raise ActionUnavailable(f"action is unavailable: {request.actionCode}")
        template = action_template(action.actionCode)
        _validate_json(request.parameters, template.parameter_schema)
        if template.execution_mode == "DISCRETE" and request.leaseMs is not None:
            raise InvalidParameters("discrete actions do not accept leaseMs")
        if template.execution_mode == "CONTINUOUS_LEASE":
            if request.leaseMs is None:
                raise InvalidParameters("continuous actions require leaseMs")
            assert template.lease is not None
            if (
                not template.lease.minLeaseMs
                <= request.leaseMs
                <= template.lease.maxLeaseMs
                or request.leaseMs < template.lease.commandCadenceMs
            ):
                raise InvalidParameters("leaseMs is outside the action lease bounds")
        validate_action_definition_envelope(action)
        return action

    def _validate_command(
        self, command: TaskCommandRequest, action: ActionDefinition
    ) -> None:
        template = action_template(action.actionCode)
        assert template.lease is not None
        _validate_json(command.parameters, template.parameter_schema)
        if (
            not template.lease.minLeaseMs
            <= command.leaseMs
            <= template.lease.maxLeaseMs
            or command.leaseMs < template.lease.commandCadenceMs
        ):
            raise InvalidParameters("leaseMs is outside the action lease bounds")

    def _require_preconditions(
        self, action: ActionDefinition, request: TaskCreateRequest
    ) -> None:
        expected = code_owned_action_definition(
            action.actionCode,
            availability=action.availability,
            policy_ref=action.policyRef,
            unavailable_reason=action.unavailableReason,
            qualification_refs=action.qualificationRefs,
        )
        if (
            request.scenario.terrain
            not in (expected.preconditions or {})["allowedTerrains"]
        ):
            raise PreconditionFailed("scenario terrain is not allowed")
        status = self.robot_status()
        if (
            status.fallen
            or status.limp
            or status.health.get("ready") is False
            or status.health.get("healthy") is False
        ):
            raise PreconditionFailed("runtime is not ready")


def _initial_status(*, ready: bool = False) -> RobotStatus:
    from datetime import UTC, datetime

    return RobotStatus(
        schema="BIPED_POSE_V1",
        timestamp=datetime.now(UTC),
        basePositionM=(0.0, 0.0, 0.0),
        baseOrientationXyzw=(0.0, 0.0, 0.0, 1.0),
        baseLinearVelocityMps=(0.0, 0.0, 0.0),
        baseAngularVelocityRadps=(0.0, 0.0, 0.0),
        jointPositionsRad=(0.0,) * 14,
        jointVelocitiesRadps=(0.0,) * 14,
        policyTarget={},
        requestedMotion={},
        appliedMotion={},
        simulationTimeS=0.0,
        loopFrequencyHz=0.0,
        fallen=False,
        limp=not ready,
        health={"ready": ready, "healthy": ready},
    )


def _validate_json(
    value: Any, schema: Mapping[str, Any], path: str = "parameters"
) -> None:
    expected = schema.get("type")
    if expected and not _matches(value, expected):
        raise InvalidParameters(f"{path} must be of type {expected}")
    if isinstance(value, Mapping):
        props = schema.get("properties", {})
        if any(x not in value for x in schema.get("required", [])):
            raise InvalidParameters(f"{path} is missing required properties")
        if schema.get("additionalProperties") is False and set(value) - set(props):
            raise InvalidParameters(f"{path} contains undeclared properties")
        for key, nested in value.items():
            if isinstance(props.get(key), Mapping):
                _validate_json(nested, props[key], f"{path}.{key}")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if (
            not math.isfinite(value)
            or value < schema.get("minimum", value)
            or value > schema.get("maximum", value)
        ):
            raise InvalidParameters(f"{path} is outside bounds")


def _matches(value: Any, expected: str | list[str]) -> bool:
    types = (expected,) if isinstance(expected, str) else tuple(expected)
    return any(
        (x == "object" and isinstance(value, Mapping))
        or (x == "array" and isinstance(value, list))
        or (x == "string" and isinstance(value, str))
        or (x == "boolean" and isinstance(value, bool))
        or (x == "integer" and isinstance(value, int) and not isinstance(value, bool))
        or (
            x == "number"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        )
        or (x == "null" and value is None)
        for x in types
    )


__all__ = [
    "ActionUnavailable",
    "BundleMismatch",
    "CommandSequenceConflict",
    "InvalidParameters",
    "NotReady",
    "PreconditionFailed",
    "RobotBusy",
    "RuntimeException",
    "SimulatorServiceError",
    "SimulatorTaskService",
    "StaleCommand",
    "TaskConflict",
    "TaskNotFound",
]
