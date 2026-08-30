"""A coordinated in-memory runtime double for discrete task tests."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Mapping
from datetime import UTC, datetime
from threading import Event, Lock, Thread
from typing import Any

from mjlab_microduck.rom.contracts import RobotStatus, TaskEvidence
from mjlab_microduck.rom.process_protocol import AckPayload, TerminalPayload
from mjlab_microduck.rom.process_supervisor import (
    CorrelatedTerminalDelivery,
    SupervisorSnapshot,
)
from mjlab_microduck.rom.runtime import RuntimeEvidence, RuntimeHandle, RuntimeSample
from mjlab_microduck.rom.supervisor_state import SupervisorState


class FakeMicroduckRuntime:
    """Fake runtime whose gates let tests control worker progress deterministically."""

    def __init__(self) -> None:
        self.validation_started = Event()
        self.validation_release = Event()
        self.validation_release.set()
        self.started = Event()
        self.start_release = Event()
        self.start_release.set()
        self.safe_stopped = Event()
        self.sample_started = Event()
        self.sample_available = Event()
        self.sample_release = Event()
        self.sample_release.set()
        self.command_started = Event()
        self.command_release = Event()
        self.command_release.set()
        self.safe_stop_started = Event()
        self.safe_stop_release = Event()
        self.safe_stop_release.set()
        self.status_started = Event()
        self.status_release = Event()
        self.status_release.set()
        self.emergency_stopped = Event()
        self._lock = Lock()
        self._samples: deque[RuntimeSample | BaseException] = deque()
        self.safe_stop_calls: list[tuple[RuntimeHandle | None, str]] = []
        self.emergency_stop_calls: list[str] = []
        self.command_calls: list[dict[str, Any]] = []
        self.operation_log: list[tuple[str, Any]] = []
        self.validation_error: BaseException | None = None
        self.start_error: BaseException | None = None
        self.status_error: BaseException | None = None
        self.zero_command_error: BaseException | None = None
        self.safe_stop_error: BaseException | None = None
        self.safe_stop_metrics: dict[str, Any] = {"safeStop": True}
        self.status_value = robot_status()
        self.status_call_count = 0
        self.active_handle: RuntimeHandle | None = None

    def __call__(self, terminal_callback):
        """Build the process-supervisor test double used by service tests."""
        return FakeRuntimeProcessSupervisor(self, terminal_callback)

    def complete_next(
        self, *, state: str, metrics: dict[str, Any], stop_reason: str | None = None
    ) -> None:
        self._samples.append(
            RuntimeSample(
                running=False,
                terminalState=state,
                metrics=metrics,
                stopReason=stop_reason,
            )
        )
        self.sample_available.set()

    def fail_next_sample(self, error: BaseException) -> None:
        self._samples.append(error)
        self.sample_available.set()

    def validate(self, action: Any, request: Any) -> None:
        self.validation_started.set()
        assert self.validation_release.wait(timeout=1.0)
        if self.validation_error is not None:
            raise self.validation_error

    def start(self, action: Any, request: Any) -> RuntimeHandle:
        self.started.set()
        self.start_release.wait()
        if self.start_error is not None:
            raise self.start_error
        handle = RuntimeHandle(taskId=request.taskId)
        with self._lock:
            self.active_handle = handle
        return handle

    def sample(self, handle: RuntimeHandle) -> RuntimeSample:
        self.sample_started.set()
        self.sample_release.wait()
        with self._lock:
            if self._samples:
                next_sample = self._samples.popleft()
                if isinstance(next_sample, BaseException):
                    raise next_sample
                return next_sample
        return RuntimeSample(running=True)

    @property
    def last_command(self) -> dict[str, Any] | None:
        with self._lock:
            return self.command_calls[-1] if self.command_calls else None

    def command(self, handle: RuntimeHandle, parameters: Mapping[str, object]) -> None:
        self.command_started.set()
        self.command_release.wait()
        command = dict(parameters)
        with self._lock:
            self.command_calls.append(command)
            self.operation_log.append(("command", command))
        if (
            command == {"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0}
            and self.zero_command_error
        ):
            raise self.zero_command_error

    def safe_stop(self, handle: RuntimeHandle | None, reason: str) -> RuntimeEvidence:
        self.safe_stop_started.set()
        with self._lock:
            self.safe_stop_calls.append((handle, reason))
            self.operation_log.append(("safe_stop", reason))
            if self.active_handle is not None and handle != self.active_handle:
                raise RuntimeError("safe stop did not receive the active handle")
        self.safe_stop_release.wait()
        with self._lock:
            if handle == self.active_handle:
                self.active_handle = None
        self.safe_stopped.set()
        if self.safe_stop_error is not None:
            raise self.safe_stop_error
        return RuntimeEvidence(metrics=self.safe_stop_metrics, stopReason=reason)

    def emergency_stop(self, reason: str) -> None:
        with self._lock:
            self.emergency_stop_calls.append(reason)
            self.operation_log.append(("emergency_stop", reason))
            self.active_handle = None
        self.emergency_stopped.set()

    def status(self) -> RobotStatus:
        with self._lock:
            self.status_call_count += 1
        self.status_started.set()
        self.status_release.wait()
        if self.status_error is not None:
            raise self.status_error
        return self.status_value


def robot_status(*, healthy: bool = True) -> RobotStatus:
    return RobotStatus(
        schema="BIPED_POSE_V1",
        timestamp=datetime.now(UTC),
        basePositionM=(0.0, 0.0, 0.25),
        baseOrientationXyzw=(0.0, 0.0, 0.0, 1.0),
        baseLinearVelocityMps=(0.0, 0.0, 0.0),
        baseAngularVelocityRadps=(0.0, 0.0, 0.0),
        jointPositionsRad=(0.0,) * 14,
        jointVelocitiesRadps=(0.0,) * 14,
        policyTarget={},
        requestedMotion={},
        appliedMotion={},
        simulationTimeS=0.0,
        loopFrequencyHz=50.0,
        fallen=False,
        limp=False,
        health={"ready": healthy, "healthy": healthy},
    )


class FakeRuntimeProcessSupervisor:
    def __init__(self, runtime, callback) -> None:
        self.runtime, self.callback = runtime, callback
        self.active_request = None
        self.handle = None
        self._generation = 1
        self._terminal = None
        self._operation_lock = Lock()
        self._finish_lock = Lock()

    def snapshot(self):
        running = self.handle is not None
        status = self.runtime.status_value
        healthy = (
            bool(status.health.get("ready", True)) and self.runtime.status_error is None
        )
        return SupervisorSnapshot(
            SupervisorState.RUNNING if running else SupervisorState.IDLE,
            self._generation,
            healthy,
            status,
            None,
            not running,
            cached_terminal=(
                CorrelatedTerminalDelivery(
                    self._generation,
                    self.active_request.taskId,
                    1,
                    self._terminal,
                )
                if self._terminal is not None and self.active_request is not None
                else None
            ),
        )

    def readiness(self):
        snap = self.snapshot()
        return snap.child_healthy and snap.slot_releasable

    def ensure_ready(self):
        if self.runtime.status_error is not None:
            raise self.runtime.status_error
        self.runtime.status()
        return self.snapshot()

    def start(self, request):
        with self._operation_lock:
            self.runtime.validate(request, request)
            try:
                self.handle = self.runtime.start(request, request)
            except Exception:
                self.runtime.safe_stop(None, "RUNTIME_EXCEPTION")
                raise
            self.active_request = request
            Thread(target=self._monitor, daemon=True).start()
            return AckPayload(acknowledgedKind="START")

    def _monitor(self):
        from mjlab_microduck.rom import process_service

        handle = self.handle
        completion = process_service.action_template(
            self.active_request.actionCode
        ).completion
        deadline = (
            time.monotonic() + completion.maxDurationMs / 1000 if completion else None
        )
        while handle is not None and self.handle == handle:
            if not self.runtime._samples:
                self.runtime.sample_available.wait(0.005)
                if deadline is not None and time.monotonic() >= deadline:
                    self._finish("TIMED_OUT", "MAX_DURATION_EXCEEDED", {})
                    return
                continue
            try:
                sample = self.runtime.sample(handle)
            except Exception:  # noqa: BLE001 - emulate sanitized child failure.
                self._finish("FAILED", "RUNTIME_EXCEPTION", {})
                return
            if sample.terminalState is not None:
                reason = (
                    "TASK_COMPLETE"
                    if sample.terminalState == "SUCCEEDED"
                    else (sample.stopReason or "RUNTIME_FAILED")
                )
                self._finish(sample.terminalState, reason, sample.metrics)
                return
            self.runtime.sample_available.clear()
            if deadline is not None and time.monotonic() >= deadline:
                self._finish("TIMED_OUT", "MAX_DURATION_EXCEEDED", {})
                return
            Event().wait(0.001)

    def command(self, task_id, command):
        with self._operation_lock:
            self.runtime.command(self.handle, command.parameters)
            return AckPayload(acknowledgedKind="COMMAND")

    def status(self, task_id):
        return self.runtime.status()

    def stop(self, task_id, reason):
        with self._operation_lock:
            return self._finish(
                (
                    "CANCELLED"
                    if reason == "CANCELLED"
                    else "TIMED_OUT"
                    if reason == "LEASE_EXPIRED"
                    else "FAILED"
                ),
                reason,
                {},
            )

    def _finish(self, outcome, reason, metrics):
        with self._finish_lock:
            return self._finish_owned(outcome, reason, metrics)

    def _finish_owned(self, outcome, reason, metrics):
        if self.handle is None and self._terminal is not None:
            return self._terminal
        handle = self.handle
        if self.active_request is not None and self.active_request.leaseMs is not None:
            self.runtime.command(
                handle, {"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0}
            )
        stopped = self.runtime.safe_stop(handle, reason)
        combined = {}
        if reason == "WATCHDOG_FAILURE":
            combined["safetyFailure"] = reason
        for source in (metrics, stopped.metrics):
            for key in sorted(source):
                try:
                    RuntimeEvidence(metrics=combined | {key: source[key]})
                except (TypeError, ValueError):
                    continue
                combined[key] = source[key]
        terminal = TerminalPayload(
            outcome=outcome,
            evidence=TaskEvidence(
                bundleDigest=self.active_request.bundleDigest,
                policyDigest="sha256:"
                + ("b" if self.active_request.actionCode == "STAND" else "c") * 64,
                modelDigest="sha256:" + "c" * 64,
                metrics=combined,
                stopReason=reason,
            ),
        )
        self.handle = None
        self._terminal = terminal
        delivery = CorrelatedTerminalDelivery(
            self._generation,
            self.active_request.taskId,
            1,
            terminal,
        )
        for _ in range(20):
            try:
                self.callback(delivery)
                break
            except RuntimeError:
                Event().wait(0.005)
        else:
            raise RuntimeError("terminal callback never accepted correlated delivery")
        return terminal

    def close(self):
        if self.handle is not None:
            self.stop(self.active_request.taskId, "SUPERVISOR_SHUTDOWN")
