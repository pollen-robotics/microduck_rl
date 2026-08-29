"""A coordinated in-memory runtime double for discrete task tests."""

from __future__ import annotations

from collections import deque
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
        self.safe_stopped = Event()
        self._lock = Lock()
        self._samples: deque[RuntimeSample | BaseException] = deque()
        self.safe_stop_calls: list[tuple[RuntimeHandle | None, str]] = []
        self.validation_error: BaseException | None = None
        self.start_error: BaseException | None = None
        self.status_error: BaseException | None = None
        self.safe_stop_metrics: dict[str, Any] = {"safeStop": True}
        self.status_value = robot_status()

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
        if self.start_error is not None:
            raise self.start_error
        self.started.set()
        return RuntimeHandle(taskId=request.taskId)

    def sample(self, handle: RuntimeHandle) -> RuntimeSample:
        with self._lock:
            if self._samples:
                next_sample = self._samples.popleft()
                if isinstance(next_sample, BaseException):
                    raise next_sample
                return next_sample
        return RuntimeSample(running=True)

    def safe_stop(self, handle: RuntimeHandle | None, reason: str) -> RuntimeEvidence:
        with self._lock:
            self.safe_stop_calls.append((handle, reason))
        self.safe_stopped.set()
        return RuntimeEvidence(metrics=self.safe_stop_metrics, stopReason=reason)

    def status(self) -> RobotStatus:
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
