"""Durable execution service for one-at-a-time MicroDuck discrete actions."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import Any

from .contracts import (
    ActionDefinition,
    PolicyArtifact,
    PolicyBundle,
    TaskCreateRequest,
    TaskEvidence,
)
from .runtime import RuntimeEvidence, RuntimeHandle, RuntimeMetric, SimulationRuntime
from .store import IllegalTaskTransition, SqliteTaskStore, TaskIdConflict


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


class RuntimeException(SimulatorServiceError):
    code = "RUNTIME_EXCEPTION"


class RobotBusy(SimulatorServiceError):
    code = "ROBOT_BUSY"


class TaskNotFound(SimulatorServiceError):
    code = "TASK_NOT_FOUND"


class TaskConflict(SimulatorServiceError):
    code = "TASK_ID_CONFLICT"


@dataclass
class _ActiveTask:
    request: TaskCreateRequest
    stop_event: Event = field(default_factory=Event)
    cancel_recorded: bool = False


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
    ) -> None:
        if bundle.bundleDigest is None:
            raise ValueError("an executable bundle requires a bundle digest")
        if pollIntervalS <= 0:
            raise ValueError("pollIntervalS must be positive")
        self._bundle = bundle
        self._store = store
        self._runtime = runtime
        self._store.mark_interrupted_unknown()
        self._poll_interval_s = pollIntervalS
        self._lock = Lock()
        self._active: _ActiveTask | None = None

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

            action = self._validate_request(request)
            if self._active is not None:
                raise RobotBusy("robot already has an active task")
            try:
                snapshot, created = self._store.create(request, request_hash)
            except TaskIdConflict as exc:
                raise TaskConflict(str(exc)) from exc
            if not created:
                return snapshot
            active = _ActiveTask(request=request)
            self._active = active
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
            if snapshot.state in {"SUCCEEDED", "FAILED", "CANCELLED", "TIMED_OUT", "UNKNOWN"}:
                return snapshot
            active.stop_event.set()
            if not active.cancel_recorded:
                active.cancel_recorded = True
                self._store.append_event(task_id, "TASK_CANCEL_REQUESTED", {"code": "CANCELLED"})
            return self._store.get(task_id) or snapshot

    def events_after(self, task_id: str, sequence: int):
        """Return ordered durable events, preserving the not-found API contract."""
        self.get_task(task_id)
        return self._store.events_after(task_id, sequence)

    def robot_status(self):
        """Expose the runtime's bounded robot status snapshot."""
        return self._runtime.status()

    def _validate_request(self, request: TaskCreateRequest) -> ActionDefinition:
        if (
            request.bundleVersion != self._bundle.bundleVersion
            or request.bundleDigest != self._bundle.bundleDigest
        ):
            raise BundleMismatch("requested bundle version or digest does not match the installed bundle")
        action = next((item for item in self._bundle.actions if item.actionCode == request.actionCode), None)
        if action is None or action.availability != "AVAILABLE" or action.executionMode != "DISCRETE":
            raise ActionUnavailable(f"action is unavailable: {request.actionCode}")
        if request.leaseMs is not None:
            raise InvalidParameters("discrete actions do not accept leaseMs")
        _validate_json_schema(request.parameters, action.parameterSchema)
        self._require_preconditions(action, request)
        return action

    def _require_preconditions(self, action: ActionDefinition, request: TaskCreateRequest) -> None:
        scenario_terrain = request.scenario.get("terrain")
        conditions = action.preconditions or {}
        allowed_terrains = conditions.get("allowedTerrains", ("flat",))
        if not isinstance(scenario_terrain, str) or scenario_terrain not in allowed_terrains:
            raise PreconditionFailed("scenario terrain is not allowed for this action")
        try:
            status = self._runtime.status()
        except Exception as exc:
            raise RuntimeException("could not read simulator runtime status") from exc
        if (
            status.fallen
            or status.limp
            or status.health.get("ready") is False
            or status.health.get("healthy") is False
        ):
            raise PreconditionFailed("simulator runtime or robot health is not ready")

    def _run_task(self, active: _ActiveTask, action: ActionDefinition) -> None:
        request = active.request
        outcome = _Outcome(state="FAILED", reason="RUNTIME_EXCEPTION")
        sample_metrics: dict[str, RuntimeMetric] = {}
        handle: RuntimeHandle | None = None
        started = False
        try:
            self._store.transition(request.taskId, "VALIDATING", event_type="TASK_VALIDATING")
            self._require_preconditions(action, request)
            self._runtime.validate(action, request)
            self._store.transition(request.taskId, "RUNNING", event_type="TASK_STARTED")
            started = True
            if active.stop_event.is_set():
                outcome = _Outcome(state="CANCELLED", reason="CANCELLED")
            else:
                handle = self._runtime.start(action, request)
                outcome = self._sample_until_terminal(active, action, handle)
                sample_metrics = outcome.metrics
        except PreconditionFailed:
            outcome = _Outcome(state="FAILED", reason="PRECONDITION_FAILED")
        except Exception:  # noqa: BLE001 - runtime implementations define arbitrary failure types.
            outcome = _Outcome(state="FAILED", reason="RUNTIME_EXCEPTION")
        finally:
            if not started:
                try:
                    self._store.transition(request.taskId, "RUNNING", event_type="TASK_STARTED")
                    started = True
                except IllegalTaskTransition:
                    # The store remains the source of truth if an external recovery won a race.
                    pass
            stop_evidence = RuntimeEvidence()
            try:
                stop_evidence = self._runtime.safe_stop(handle, outcome.reason)
            except Exception:  # noqa: BLE001 - a failed safe stop is itself a runtime failure.
                outcome = _Outcome(state="FAILED", reason="RUNTIME_EXCEPTION", metrics=sample_metrics)
            evidence = self._evidence_for(action, outcome, sample_metrics, stop_evidence)
            if started:
                try:
                    self._store.transition(
                        request.taskId,
                        outcome.state,
                        event_type=f"TASK_{outcome.state}",
                        payload={"code": outcome.reason},
                        evidence=evidence,
                        stop_reason=outcome.reason,
                    )
                except IllegalTaskTransition:
                    pass
            with self._lock:
                if self._active is active:
                    self._active = None

    def _sample_until_terminal(
        self, active: _ActiveTask, action: ActionDefinition, handle: RuntimeHandle
    ) -> _Outcome:
        assert action.completion is not None
        deadline = time.monotonic() + action.completion.maxDurationMs / 1_000
        while True:
            if active.stop_event.is_set():
                return _Outcome(state="CANCELLED", reason="CANCELLED")
            sample = self._runtime.sample(handle)
            if sample.terminalState is not None:
                return _Outcome(
                    state=sample.terminalState,
                    reason=_terminal_result_code(sample.terminalState, sample.stopReason),
                    metrics=dict(sample.metrics),
                )
            if time.monotonic() >= deadline:
                return _Outcome(state="TIMED_OUT", reason="MAX_DURATION_EXCEEDED")
            active.stop_event.wait(min(self._poll_interval_s, max(0.0, deadline - time.monotonic())))

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


def _policy_for(bundle: PolicyBundle, action: ActionDefinition) -> PolicyArtifact:
    assert action.policyRef is not None
    policy = next((item for item in bundle.policies if item.policyRef == action.policyRef), None)
    if policy is None:
        raise RuntimeError(f"missing policy artifact: {action.policyRef}")
    return policy


def _merge_metrics(
    stop_metrics: Mapping[str, RuntimeMetric], sample_metrics: Mapping[str, RuntimeMetric]
) -> dict[str, RuntimeMetric]:
    merged = dict(stop_metrics)
    merged.update(sample_metrics)
    if len(merged) > 32:
        return dict(list(merged.items())[:32])
    return merged


def _terminal_result_code(state: str, runtime_reason: str | None) -> str:
    """Map runtime-local detail onto the fixed public discrete-task result vocabulary."""
    if state == "SUCCEEDED":
        return "TASK_COMPLETE"
    if runtime_reason == "FALLEN":
        return "FALLEN"
    return "RUNTIME_FAILED"


def _validate_json_schema(value: Any, schema: Mapping[str, Any], path: str = "parameters") -> None:
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
        or (item == "integer" and isinstance(value, int) and not isinstance(value, bool))
        or (item == "number" and isinstance(value, int | float) and not isinstance(value, bool))
        or (item == "null" and value is None)
        for item in expected_types
    )


def _validate_object(value: Mapping[str, Any], schema: Mapping[str, Any], path: str) -> None:
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
    if "minimum" in schema and value < schema["minimum"]:
        raise InvalidParameters(f"{path} is below minimum")
    if "maximum" in schema and value > schema["maximum"]:
        raise InvalidParameters(f"{path} exceeds maximum")


def _validate_string(value: str, schema: Mapping[str, Any], path: str) -> None:
    if "minLength" in schema and len(value) < schema["minLength"]:
        raise InvalidParameters(f"{path} is too short")
    if "maxLength" in schema and len(value) > schema["maxLength"]:
        raise InvalidParameters(f"{path} is too long")
