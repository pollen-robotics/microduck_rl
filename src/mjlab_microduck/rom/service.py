"""Durable execution service for one-at-a-time MicroDuck discrete actions."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import Any

from .action_catalog import (
    action_template,
    code_owned_action_definition,
    validate_action_definition_envelope,
    validate_bundle_action_envelope,
)
from .contracts import (
    ActionDefinition,
    PolicyArtifact,
    PolicyBundle,
    RobotStatus,
    TaskCommandRequest,
    TaskCreateRequest,
    TaskEvidence,
)
from .runtime import (
    RuntimeEvidence,
    RuntimeHandle,
    RuntimeMetric,
    RuntimeSample,
    SimulationRuntime,
)
from .store import (
    CommandSequenceConflict as StoreCommandSequenceConflict,
)
from .store import (
    IllegalTaskTransition,
    SqliteTaskStore,
    TaskIdConflict,
)
from .store import (
    StaleCommand as StoreStaleCommand,
)


class SimulatorServiceError(ValueError):
    """A stable public service error suitable for an API error response."""

    code = "INTERNAL_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)


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


class _RuntimeCallTimedOut(RuntimeError):
    """A supervised runtime operation exceeded its monotonic deadline."""


class _RuntimeCallSuperseded(RuntimeError):
    """A runtime result arrived after its task generation lost ownership."""


@dataclass
class _ActiveTask:
    generation: int
    request: TaskCreateRequest
    action: ActionDefinition | None = None
    stop_event: Event = field(default_factory=Event)
    cancel_recorded: bool = False
    handle: RuntimeHandle | None = None
    deadline: float | None = None
    continuous: bool = False
    stop_claimed: bool = False
    terminalized: bool = False
    emergency_claimed: bool = False


@dataclass
class _Outcome:
    state: str
    reason: str
    metrics: dict[str, RuntimeMetric] = field(default_factory=dict)


class SimulatorTaskService:
    """Owns the single motion slot and maps runtime outcomes to durable tasks."""

    def __init__(
        self,
        bundle: PolicyBundle,
        store: SqliteTaskStore,
        runtime: SimulationRuntime,
        *,
        pollIntervalS: float = 0.05,
        runtimeCallTimeoutS: float = 0.25,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        validate_bundle_action_envelope(bundle)
        if pollIntervalS <= 0:
            raise ValueError("pollIntervalS must be positive")
        if runtimeCallTimeoutS <= 0 or runtimeCallTimeoutS > 5.0:
            raise ValueError(
                "runtimeCallTimeoutS must be between zero and five seconds"
            )
        self._bundle = bundle
        self._store = store
        self._runtime = runtime
        self._store.mark_interrupted_unknown()
        self._poll_interval_s = pollIntervalS
        self._runtime_call_timeout_s = runtimeCallTimeoutS
        self._monotonic_clock = monotonic_clock
        self._lock = Lock()
        self._runtime_operation_lock = Lock()
        self._active: _ActiveTask | None = None
        self._watchdog_healthy = True
        self._readiness_failure_reason: str | None = None
        self._next_generation = 1
        self._global_emergency_claimed = False
        self._emergency_stop_failed = False
        self._last_status = runtime.status()

    def create_task(self, request: TaskCreateRequest):
        """Accept a valid discrete task and begin its durable worker lifecycle."""
        request_hash = _request_hash(request)
        with self._lock:
            existing = self._store.get(request.taskId)
            if existing is not None:
                try:
                    snapshot, _ = self._store.create(request, request_hash)
                except TaskIdConflict as exc:
                    raise TaskConflict(str(exc)) from exc
                return snapshot

        self._require_motion_ready()
        action = self._validate_request(request)
        self._require_preconditions(action, request)
        with self._lock:
            existing = self._store.get(request.taskId)
            if existing is not None:
                try:
                    snapshot, _ = self._store.create(request, request_hash)
                except TaskIdConflict as exc:
                    raise TaskConflict(str(exc)) from exc
                return snapshot
            if not self._watchdog_healthy:
                raise NotReady("simulator is not ready for motion")
            if self._active is not None:
                raise RobotBusy("robot already has an active task")
            try:
                snapshot, created = self._store.create(request, request_hash)
            except TaskIdConflict as exc:
                raise TaskConflict(str(exc)) from exc
            if not created:
                return snapshot
            active = _ActiveTask(
                generation=self._next_generation,
                request=request,
                action=action,
                continuous=action.executionMode == "CONTINUOUS_LEASE",
            )
            self._next_generation += 1
            self._active = active
        if active.continuous:
            return self._start_continuous(active, action)
        with self._lock:
            if self._active is not active or active.terminalized:
                return self._store.get(request.taskId) or snapshot
            Thread(
                target=self._run_task,
                args=(active, action),
                name=f"microduck-task-{request.taskId}",
                daemon=True,
            ).start()
        return snapshot

    def get_task(self, task_id: str):
        """Return a durable task or a stable not-found error."""
        snapshot = self._store.get(task_id)
        if snapshot is None:
            raise TaskNotFound(f"task not found: {task_id}")
        return snapshot

    def cancel_task(self, task_id: str):
        """Request an idempotent worker-owned safe stop without skipping states."""
        with self._lock:
            snapshot = self._store.get(task_id)
            if snapshot is None:
                raise TaskNotFound(f"task not found: {task_id}")
            active = self._active
            if active is None or active.request.taskId != task_id:
                return snapshot
            if snapshot.state in {
                "SUCCEEDED",
                "FAILED",
                "CANCELLED",
                "TIMED_OUT",
                "UNKNOWN",
            }:
                return snapshot
            if active.continuous:
                continuous = True
            else:
                continuous = False
                active.stop_event.set()
                if not active.cancel_recorded:
                    active.cancel_recorded = True
                    self._store.append_event(
                        task_id, "TASK_CANCEL_REQUESTED", {"code": "CANCELLED"}
                    )
                return self._store.get(task_id) or snapshot
        assert continuous
        return self._stop_continuous(active, "CANCELLED", "CANCELLED")

    def events_after(self, task_id: str, sequence: int, *, page_size: int = 100):
        """Return ordered durable events, preserving the not-found API contract."""
        if (
            not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or not -1 <= sequence <= 2**63 - 1
        ):
            raise InvalidParameters("afterSequence must be a signed 64-bit cursor")
        self.get_task(task_id)
        return self._store.events_after(task_id, sequence, page_size=page_size)

    def robot_status(self):
        """Expose the runtime's bounded robot status snapshot."""
        with self._lock:
            active = self._active
            failure_reason = self._readiness_failure_reason
        if failure_reason is not None:
            return self._unhealthy_cached_status(failure_reason)
        try:
            return self._runtime_status(active)
        except _RuntimeCallTimedOut:
            return self._unhealthy_cached_status("RUNTIME_UNRESPONSIVE")
        except Exception:  # noqa: BLE001 - diagnostics use the last bounded snapshot.
            return self._unhealthy_cached_status("RUNTIME_UNAVAILABLE")

    def motion_readiness(self) -> tuple[bool, tuple[str, ...]]:
        """Return the one fail-closed predicate used for new or renewed motion."""
        reasons: list[str] = []
        with self._lock:
            healthy = self._watchdog_healthy
            failure_reason = self._readiness_failure_reason
            active = self._active
        if not healthy:
            reasons.append(failure_reason or "WATCHDOG_UNHEALTHY")
            return False, tuple(sorted(set(reasons)))
        try:
            health = self._runtime_status(active).health
            if health.get("ready") is not True or health.get("healthy") is not True:
                reasons.append("RUNTIME_UNAVAILABLE")
        except _RuntimeCallTimedOut:
            reasons.append("RUNTIME_UNRESPONSIVE")
        except Exception:  # noqa: BLE001 - readiness must not expose runtime failures.
            reasons.append("RUNTIME_UNAVAILABLE")
        unique = tuple(sorted(set(reasons)))
        return not unique, unique

    def watchdog_failed(self) -> None:
        """Permanently fail closed and terminalize continuous ownership safely."""
        with self._lock:
            self._watchdog_healthy = False
            if self._readiness_failure_reason is None:
                self._readiness_failure_reason = "WATCHDOG_UNHEALTHY"
            active = self._active
            if active is None or not active.continuous:
                return
        self._stop_continuous(
            active,
            "FAILED",
            "WATCHDOG_FAILURE",
            sample_metrics={"safetyFailure": "WATCHDOG_FAILURE"},
        )

    def command(self, task_id: str, command: TaskCommandRequest):
        """Accept a monotonic continuous command and renew its target-side lease."""
        self._require_motion_ready()
        expired: _ActiveTask | None = None
        with self._lock:
            snapshot = self._store.get(task_id)
            if snapshot is None:
                raise TaskNotFound(f"task not found: {task_id}")
            active = self._active
            if (
                active is None
                or active.request.taskId != task_id
                or not active.continuous
            ):
                raise InvalidParameters("task does not accept continuous commands")
            if snapshot.state != "RUNNING" or active.handle is None:
                raise InvalidParameters("task is not running")
            if (
                active.deadline is not None
                and self._monotonic_clock() >= active.deadline
            ):
                expired = active
            if expired is None:
                assert active.action is not None
                self._validate_command(command, active.action)
                deadline = self._monotonic_clock() + command.leaseMs / 1_000
                try:
                    accepted, created = self._store.record_command(
                        task_id, command, _command_hash(command), deadline
                    )
                except StoreStaleCommand as exc:
                    raise StaleCommand(str(exc)) from exc
                except StoreCommandSequenceConflict as exc:
                    raise CommandSequenceConflict(str(exc)) from exc
                if not created:
                    return accepted
                handle = active.handle
        if expired is not None:
            self._stop_continuous(expired, "TIMED_OUT", "LEASE_EXPIRED")
            raise InvalidParameters("task is not running")
        assert handle is not None
        try:
            self._invoke_runtime(
                "command",
                lambda: self._runtime.command(handle, command.parameters),
                active,
            )
        except _RuntimeCallTimedOut as exc:
            raise RuntimeException("simulator command was unresponsive") from exc
        except _RuntimeCallSuperseded as exc:
            raise RuntimeException("simulator command lost task ownership") from exc
        except Exception as exc:
            self._stop_continuous(active, "FAILED", "RUNTIME_EXCEPTION")
            raise RuntimeException("could not apply simulator command") from exc
        with self._lock:
            if self._active is active and not active.terminalized:
                active.deadline = deadline
        return accepted

    def tick(self) -> None:
        """Observe runtime safety before enforcing the continuous lease deadline."""
        with self._lock:
            active = self._active
            if active is None or not active.continuous or active.handle is None:
                return
            handle = active.handle
            deadline = active.deadline
        if deadline is not None and self._monotonic_clock() >= deadline:
            self._stop_continuous(active, "TIMED_OUT", "LEASE_EXPIRED")
            return
        try:
            sample = self._invoke_runtime(
                "sample", lambda: self._runtime.sample(handle), active
            )
        except (_RuntimeCallTimedOut, _RuntimeCallSuperseded):
            return
        except Exception:  # noqa: BLE001 - runtime faults must terminalize ownership.
            self._stop_continuous(active, "FAILED", "RUNTIME_EXCEPTION")
            return
        if sample.terminalState is not None:
            self._stop_continuous(
                active,
                "FAILED",
                _continuous_terminal_reason(sample),
                sample_metrics=sample.metrics,
            )
            return
        with self._lock:
            deadline = active.deadline if self._active is active else None
        if deadline is not None and self._monotonic_clock() >= deadline:
            self._stop_continuous(active, "TIMED_OUT", "LEASE_EXPIRED")

    def _validate_request(self, request: TaskCreateRequest) -> ActionDefinition:
        if (
            request.bundleVersion != self._bundle.bundleVersion
            or request.bundleDigest != self._bundle.bundleDigest
        ):
            raise BundleMismatch(
                "requested bundle version or digest does not match the installed bundle"
            )
        action = next(
            (
                item
                for item in self._bundle.actions
                if item.actionCode == request.actionCode
            ),
            None,
        )
        if action is None or action.availability != "AVAILABLE":
            raise ActionUnavailable(f"action is unavailable: {request.actionCode}")
        template = action_template(action.actionCode)
        _validate_json_schema(request.parameters, template.parameter_schema)
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
            ):
                raise InvalidParameters("leaseMs is outside the action lease bounds")
            if request.leaseMs < template.lease.commandCadenceMs:
                raise InvalidParameters("leaseMs is shorter than the command cadence")
        validate_action_definition_envelope(action)
        policy = _policy_for(self._bundle, action)
        if policy.taskId not in template.task_ids:
            raise ActionUnavailable("action policy identity is not executable")
        return action

    def _require_motion_ready(self) -> None:
        ready, _ = self.motion_readiness()
        if not ready:
            raise NotReady("simulator is not ready for motion")

    def _validate_command(
        self, command: TaskCommandRequest, action: ActionDefinition
    ) -> None:
        template = action_template(action.actionCode)
        assert template.lease is not None
        _validate_json_schema(command.parameters, template.parameter_schema)
        if (
            not template.lease.minLeaseMs
            <= command.leaseMs
            <= template.lease.maxLeaseMs
        ):
            raise InvalidParameters("leaseMs is outside the action lease bounds")
        if command.leaseMs < template.lease.commandCadenceMs:
            raise InvalidParameters("leaseMs is shorter than the command cadence")
        validate_action_definition_envelope(action)

    def _start_continuous(self, active: _ActiveTask, action: ActionDefinition):
        """Establish continuous ownership without holding the service mutex."""
        request = active.request
        try:
            self._store.transition(
                request.taskId, "VALIDATING", event_type="TASK_VALIDATING"
            )
            self._invoke_runtime(
                "validate", lambda: self._runtime.validate(action, request), active
            )
            handle = self._invoke_runtime(
                "start", lambda: self._runtime.start(action, request), active
            )
            assert request.leaseMs is not None
            deadline = self._monotonic_clock() + request.leaseMs / 1_000
            with self._lock:
                if self._active is not active or active.terminalized:
                    raise _RuntimeCallSuperseded
                active.handle = handle
                active.deadline = deadline
                return self._store.start_continuous(request.taskId, deadline)
        except _RuntimeCallTimedOut as exc:
            raise RuntimeException("simulator runtime start was unresponsive") from exc
        except _RuntimeCallSuperseded:
            return self._store.get(request.taskId)
        except Exception as exc:
            self._stop_continuous(active, "FAILED", "RUNTIME_EXCEPTION")
            raise RuntimeException("could not start simulator runtime") from exc

    def _stop_continuous(
        self,
        active: _ActiveTask,
        state: str,
        reason: str,
        *,
        sample_metrics: Mapping[str, RuntimeMetric] | None = None,
    ):
        """Claim one stop, run bounded safety calls, and persist one terminal state."""
        with self._lock:
            if self._active is not active or active.terminalized:
                return self._store.get(active.request.taskId)
            if active.stop_claimed:
                return self._store.get(active.request.taskId)
            active.stop_claimed = True
            active.stop_event.set()
            handle = active.handle
        assert active.action is not None
        safety_failures: list[str] = []
        zero_parameters = _zero_parameters(active.action)
        if handle is not None:
            try:
                self._invoke_runtime(
                    "zero_command",
                    lambda: self._runtime.command(handle, zero_parameters),
                    active,
                )
            except (_RuntimeCallTimedOut, _RuntimeCallSuperseded):
                return self._store.get(active.request.taskId)
            except Exception:  # noqa: BLE001 - zero failure must not skip safe stopping.
                safety_failures.append("ZERO_COMMAND_FAILED")
        stop_evidence = RuntimeEvidence()
        try:
            stop_evidence = self._invoke_runtime(
                "safe_stop",
                lambda: self._runtime.safe_stop(handle, reason),
                active,
            )
        except (_RuntimeCallTimedOut, _RuntimeCallSuperseded):
            return self._store.get(active.request.taskId)
        except Exception:  # noqa: BLE001 - terminal persistence survives stop failure.
            safety_failures.append("SAFE_STOP_FAILED")
        safety_code = "_AND_".join(safety_failures) if safety_failures else None
        if safety_code is not None:
            stop_evidence = RuntimeEvidence(metrics={"safetyFailure": safety_code})
        evidence = self._evidence_for(
            active.action,
            _Outcome(state, reason),
            sample_metrics or {},
            stop_evidence,
        )
        payload: dict[str, str] = {"code": reason}
        if safety_code is not None:
            payload["safetyCode"] = safety_code
        with self._lock:
            if self._active is not active or active.terminalized:
                return self._store.get(active.request.taskId)
            active.terminalized = True
            result = self._persist_terminal(
                active,
                state,
                reason,
                evidence=evidence,
                payload=payload,
            )
            self._active = None
            return result

    def _require_preconditions(
        self, action: ActionDefinition, request: TaskCreateRequest
    ) -> None:
        scenario_terrain = request.scenario.terrain
        expected = code_owned_action_definition(
            action.actionCode,
            availability=action.availability,
            policy_ref=action.policyRef,
            unavailable_reason=action.unavailableReason,
            qualification_refs=action.qualificationRefs,
        )
        conditions = expected.preconditions or {}
        allowed_terrains = conditions["allowedTerrains"]
        if scenario_terrain not in allowed_terrains:
            raise PreconditionFailed("scenario terrain is not allowed for this action")
        with self._lock:
            active = self._active
        try:
            status = self._runtime_status(active)
        except _RuntimeCallTimedOut as exc:
            raise NotReady("simulator runtime status was unresponsive") from exc
        except Exception as exc:
            raise RuntimeException("could not read simulator runtime status") from exc
        if (
            status.fallen
            or status.limp
            or status.health.get("ready") is False
            or status.health.get("healthy") is False
        ):
            raise PreconditionFailed("simulator runtime or robot health is not ready")

    def _invoke_runtime(
        self,
        operation: str,
        function: Callable[[], Any],
        active: _ActiveTask | None,
    ) -> Any:
        """Run one serialized runtime call under an independent monotonic deadline."""
        completed = Event()
        outcome: dict[str, Any] = {}

        def invoke() -> None:
            try:
                with self._runtime_operation_lock:
                    if active is not None and not self._owns_generation(active):
                        raise _RuntimeCallSuperseded
                    outcome["result"] = function()
            except BaseException as exc:  # noqa: BLE001 - propagate runtime-defined errors.
                outcome["error"] = exc
            finally:
                completed.set()

        Thread(
            target=invoke,
            name=f"microduck-runtime-{operation}",
            daemon=True,
        ).start()
        deadline = time.monotonic() + self._runtime_call_timeout_s
        while not completed.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                self._runtime_unresponsive(active, operation)
                raise _RuntimeCallTimedOut(operation)
            completed.wait(remaining)
        error = outcome.get("error")
        if error is not None:
            raise error
        if active is not None and not self._owns_generation(active):
            raise _RuntimeCallSuperseded(operation)
        return outcome.get("result")

    def _owns_generation(self, active: _ActiveTask) -> bool:
        with self._lock:
            return (
                self._active is active
                and self._active.generation == active.generation
                and not active.terminalized
            )

    def _runtime_unresponsive(self, active: _ActiveTask | None, operation: str) -> None:
        """Fail closed exactly once; late runtime results have no service authority."""
        terminalize = False
        emergency = False
        with self._lock:
            self._watchdog_healthy = False
            self._readiness_failure_reason = "RUNTIME_UNRESPONSIVE"
            if (
                active is not None
                and self._active is active
                and not active.terminalized
            ):
                active.stop_claimed = True
                active.terminalized = True
                active.stop_event.set()
                terminalize = True
                if not active.emergency_claimed:
                    active.emergency_claimed = True
                    emergency = True
                self._active = None
            elif active is None and not self._global_emergency_claimed:
                self._global_emergency_claimed = True
                emergency = True
        if emergency:
            try:
                self._runtime.emergency_stop("RUNTIME_UNRESPONSIVE")
            except Exception:  # noqa: BLE001 - durable fail-closed state remains authoritative.
                with self._lock:
                    self._emergency_stop_failed = True
        if terminalize:
            assert active is not None and active.action is not None
            evidence = self._evidence_for(
                active.action,
                _Outcome("FAILED", "RUNTIME_UNRESPONSIVE"),
                {
                    "safetyFailure": "RUNTIME_UNRESPONSIVE",
                    "runtimeOperation": operation,
                },
                RuntimeEvidence(),
            )
            self._persist_terminal(
                active,
                "FAILED",
                "RUNTIME_UNRESPONSIVE",
                evidence=evidence,
                payload={
                    "code": "RUNTIME_UNRESPONSIVE",
                    "runtimeOperation": operation,
                },
            )

    def _persist_terminal(
        self,
        active: _ActiveTask,
        state: str,
        reason: str,
        *,
        evidence: TaskEvidence,
        payload: Mapping[str, str],
    ):
        """Advance any reserved lifecycle to one immutable terminal snapshot."""
        for _ in range(3):
            current = self._store.get(active.request.taskId)
            if current is None:
                return None
            if current.state in {
                "SUCCEEDED",
                "FAILED",
                "CANCELLED",
                "TIMED_OUT",
                "UNKNOWN",
            }:
                return current
            try:
                if current.state == "ACCEPTED":
                    self._store.transition(
                        active.request.taskId,
                        "VALIDATING",
                        event_type="TASK_VALIDATING",
                    )
                    continue
                if current.state == "VALIDATING":
                    self._store.transition(
                        active.request.taskId,
                        "RUNNING",
                        event_type="TASK_STARTED",
                    )
                    continue
                return self._store.transition(
                    active.request.taskId,
                    state,
                    event_type=f"TASK_{state}",
                    payload=dict(payload),
                    evidence=evidence,
                    stop_reason=reason,
                )
            except IllegalTaskTransition:
                continue
        return self._store.get(active.request.taskId)

    def _runtime_status(self, active: _ActiveTask | None) -> RobotStatus:
        status = self._invoke_runtime("status", self._runtime.status, active)
        if not isinstance(status, RobotStatus):
            raise TypeError("runtime status must use the RobotStatus contract")
        with self._lock:
            self._last_status = status
        return status

    def _unhealthy_cached_status(self, reason: str) -> RobotStatus:
        with self._lock:
            cached = self._last_status
        health = dict(cached.health)
        health.update({"ready": False, "healthy": False, "reasonCodes": [reason]})
        return cached.model_copy(update={"limp": True, "health": health})

    def _run_task(self, active: _ActiveTask, action: ActionDefinition) -> None:
        request = active.request
        outcome = _Outcome(state="FAILED", reason="RUNTIME_EXCEPTION")
        sample_metrics: dict[str, RuntimeMetric] = {}
        handle: RuntimeHandle | None = None
        started = False
        try:
            self._store.transition(
                request.taskId, "VALIDATING", event_type="TASK_VALIDATING"
            )
            self._require_preconditions(action, request)
            self._invoke_runtime(
                "validate", lambda: self._runtime.validate(action, request), active
            )
            self._store.transition(request.taskId, "RUNNING", event_type="TASK_STARTED")
            started = True
            if active.stop_event.is_set():
                outcome = _Outcome(state="CANCELLED", reason="CANCELLED")
            else:
                handle = self._invoke_runtime(
                    "start", lambda: self._runtime.start(action, request), active
                )
                outcome = self._sample_until_terminal(active, action, handle)
                sample_metrics = outcome.metrics
        except (_RuntimeCallTimedOut, _RuntimeCallSuperseded):
            return
        except PreconditionFailed:
            outcome = _Outcome(state="FAILED", reason="PRECONDITION_FAILED")
        except Exception:  # noqa: BLE001 - runtime implementations define arbitrary failure types.
            outcome = _Outcome(state="FAILED", reason="RUNTIME_EXCEPTION")
        finally:
            if not active.terminalized:
                self._finish_discrete(
                    active,
                    action,
                    outcome,
                    sample_metrics,
                    handle,
                    started,
                )

    def _finish_discrete(
        self,
        active: _ActiveTask,
        action: ActionDefinition,
        outcome: _Outcome,
        sample_metrics: Mapping[str, RuntimeMetric],
        handle: RuntimeHandle | None,
        started: bool,
    ) -> None:
        """Bound and persist discrete safe stop without owning the service mutex."""
        if not started:
            try:
                self._store.transition(
                    active.request.taskId, "RUNNING", event_type="TASK_STARTED"
                )
                started = True
            except IllegalTaskTransition:
                # The store remains authoritative if external recovery won a race.
                pass
        stop_evidence = RuntimeEvidence()
        try:
            stop_evidence = self._invoke_runtime(
                "safe_stop",
                lambda: self._runtime.safe_stop(handle, outcome.reason),
                active,
            )
        except (_RuntimeCallTimedOut, _RuntimeCallSuperseded):
            return
        except Exception:  # noqa: BLE001 - a failed safe stop is a runtime failure.
            outcome = _Outcome(
                state="FAILED", reason="RUNTIME_EXCEPTION", metrics=dict(sample_metrics)
            )
        evidence = self._evidence_for(action, outcome, sample_metrics, stop_evidence)
        with self._lock:
            if self._active is active and not active.terminalized:
                active.terminalized = True
                if started:
                    self._persist_terminal(
                        active,
                        outcome.state,
                        outcome.reason,
                        evidence=evidence,
                        payload={"code": outcome.reason},
                    )
                self._active = None

    def _sample_until_terminal(
        self, active: _ActiveTask, action: ActionDefinition, handle: RuntimeHandle
    ) -> _Outcome:
        completion = action_template(action.actionCode).completion
        assert completion is not None
        deadline = time.monotonic() + completion.maxDurationMs / 1_000
        while True:
            if active.stop_event.is_set():
                return _Outcome(state="CANCELLED", reason="CANCELLED")
            sample = self._invoke_runtime(
                "sample", lambda: self._runtime.sample(handle), active
            )
            if sample.terminalState is not None:
                return _Outcome(
                    state=sample.terminalState,
                    reason=_terminal_result_code(
                        sample.terminalState, sample.stopReason
                    ),
                    metrics=dict(sample.metrics),
                )
            if time.monotonic() >= deadline:
                return _Outcome(state="TIMED_OUT", reason="MAX_DURATION_EXCEEDED")
            active.stop_event.wait(
                min(self._poll_interval_s, max(0.0, deadline - time.monotonic()))
            )

    def _evidence_for(
        self,
        action: ActionDefinition,
        outcome: _Outcome,
        sample_metrics: Mapping[str, RuntimeMetric],
        stop_evidence: RuntimeEvidence,
    ) -> TaskEvidence:
        policy = _policy_for(self._bundle, action)
        metrics = _merge_metrics(stop_evidence.metrics, sample_metrics)
        return TaskEvidence(
            bundleDigest=self._bundle.bundleDigest,
            policyDigest=policy.digest,
            modelDigest=self._bundle.model.digest,
            metrics=metrics,
            stopReason=outcome.reason,
        )


def _request_hash(request: TaskCreateRequest) -> str:
    from .contracts import sha256_prefixed

    return sha256_prefixed(request)


def _command_hash(command: TaskCommandRequest) -> str:
    from .contracts import sha256_prefixed

    return sha256_prefixed(command)


def _zero_parameters(action: ActionDefinition) -> dict[str, float]:
    """Return the exact neutral command from the code-owned lease contract."""
    template = action_template(action.actionCode)
    assert template.lease is not None
    zero = dict(template.lease.zeroCommand)
    _validate_json_schema(zero, template.parameter_schema)
    return zero


def _policy_for(bundle: PolicyBundle, action: ActionDefinition) -> PolicyArtifact:
    assert action.policyRef is not None
    policy = next(
        (item for item in bundle.policies if item.policyRef == action.policyRef), None
    )
    if policy is None:
        raise RuntimeError(f"missing policy artifact: {action.policyRef}")
    return policy


def _merge_metrics(
    stop_metrics: Mapping[str, RuntimeMetric],
    sample_metrics: Mapping[str, RuntimeMetric],
) -> dict[str, RuntimeMetric]:
    """Keep primary runtime outcome metrics before lower-priority safe-stop diagnostics."""
    merged: dict[str, RuntimeMetric] = {}
    for metrics in (sample_metrics, stop_metrics):
        for key in sorted(metrics):
            if key in merged:
                continue
            candidate = merged | {key: metrics[key]}
            try:
                RuntimeEvidence(metrics=candidate)
            except (TypeError, ValueError):
                continue
            merged = candidate
    return merged


def _terminal_result_code(state: str, runtime_reason: str | None) -> str:
    """Map runtime-local detail onto the fixed public discrete-task result vocabulary."""
    if state == "SUCCEEDED":
        return "TASK_COMPLETE"
    if runtime_reason == "FALLEN":
        return "FALLEN"
    return "RUNTIME_FAILED"


_CONTINUOUS_FATAL_REASONS = frozenset(
    {
        "CONTROL_LOOP_OVERRUN",
        "FALLEN",
        "JOINT_LIMIT",
        "NON_FINITE_POLICY_OUTPUT",
        "NON_FINITE_STATE",
        "RUNTIME_EXCEPTION",
    }
)


def _continuous_terminal_reason(sample: RuntimeSample) -> str:
    if sample.stopReason in _CONTINUOUS_FATAL_REASONS:
        return sample.stopReason
    return "RUNTIME_FAILED"


def _validate_json_schema(
    value: Any, schema: Mapping[str, Any], path: str = "parameters"
) -> None:
    """Validate the small JSON Schema subset emitted by the action catalog."""
    expected = schema.get("type")
    if expected is not None and not _matches_type(value, expected):
        raise InvalidParameters(f"{path} must be of type {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise InvalidParameters(f"{path} must be one of the declared values")
    if isinstance(value, Mapping):
        _validate_object(value, schema, path)
    elif isinstance(value, list):
        _validate_array(value, schema, path)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        _validate_number(value, schema, path)
    elif isinstance(value, str):
        _validate_string(value, schema, path)


def _matches_type(value: Any, expected: str | list[str]) -> bool:
    expected_types = (expected,) if isinstance(expected, str) else tuple(expected)
    return any(
        (item == "object" and isinstance(value, Mapping))
        or (item == "array" and isinstance(value, list))
        or (item == "string" and isinstance(value, str))
        or (item == "boolean" and isinstance(value, bool))
        or (
            item == "integer" and isinstance(value, int) and not isinstance(value, bool)
        )
        or (
            item == "number"
            and isinstance(value, int | float)
            and not isinstance(value, bool)
        )
        or (item == "null" and value is None)
        for item in expected_types
    )


def _validate_object(
    value: Mapping[str, Any], schema: Mapping[str, Any], path: str
) -> None:
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    for name in required:
        if name not in value:
            raise InvalidParameters(f"{path}.{name} is required")
    if schema.get("additionalProperties") is False:
        unknown = set(value) - set(properties)
        if unknown:
            raise InvalidParameters(f"{path} contains undeclared properties")
    for name, nested_value in value.items():
        nested_schema = properties.get(name)
        if isinstance(nested_schema, Mapping):
            _validate_json_schema(nested_value, nested_schema, f"{path}.{name}")


def _validate_array(value: list[Any], schema: Mapping[str, Any], path: str) -> None:
    if "minItems" in schema and len(value) < schema["minItems"]:
        raise InvalidParameters(f"{path} has too few items")
    if "maxItems" in schema and len(value) > schema["maxItems"]:
        raise InvalidParameters(f"{path} has too many items")
    item_schema = schema.get("items")
    if isinstance(item_schema, Mapping):
        for index, item in enumerate(value):
            _validate_json_schema(item, item_schema, f"{path}[{index}]")


def _validate_number(value: float, schema: Mapping[str, Any], path: str) -> None:
    if not math.isfinite(value):
        raise InvalidParameters(f"{path} must be finite")
    if "minimum" in schema and value < schema["minimum"]:
        raise InvalidParameters(f"{path} is below minimum")
    if "maximum" in schema and value > schema["maximum"]:
        raise InvalidParameters(f"{path} exceeds maximum")


def _validate_string(value: str, schema: Mapping[str, Any], path: str) -> None:
    if "minLength" in schema and len(value) < schema["minLength"]:
        raise InvalidParameters(f"{path} is too short")
    if "maxLength" in schema and len(value) > schema["maxLength"]:
        raise InvalidParameters(f"{path} is too long")
