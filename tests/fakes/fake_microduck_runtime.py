"""A coordinated in-memory runtime double for discrete task tests."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from datetime import UTC, datetime
from threading import Event, Lock
from typing import Any

from mjlab_microduck.rom.contracts import RobotStatus
from mjlab_microduck.rom.runtime import RuntimeEvidence, RuntimeHandle, RuntimeSample


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

    def fail_next_sample(self, error: BaseException) -> None:
        self._samples.append(error)

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
