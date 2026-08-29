"""Bounded interface between durable ROM tasks and a simulator runtime."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from .contracts import ActionDefinition, RobotStatus, TaskCreateRequest, canonical_json

type RuntimeMetric = str | int | float | bool | None
_MAX_METRICS = 32
_MAX_METRIC_KEY_LENGTH = 64
_MAX_METRIC_STRING_LENGTH = 128
_MAX_METRICS_ENCODED_BYTES = 1_024


def _bounded_metrics(metrics: Mapping[str, RuntimeMetric]) -> dict[str, RuntimeMetric]:
    """Copy the bounded scalar-only metrics safe to persist as task evidence."""
    if len(metrics) > _MAX_METRICS:
        raise ValueError(f"runtime metrics must contain at most {_MAX_METRICS} entries")
    bounded: dict[str, RuntimeMetric] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key or len(key) > _MAX_METRIC_KEY_LENGTH:
            raise ValueError("runtime metric names must be non-empty strings of bounded length")
        if not isinstance(value, str | int | float | bool | type(None)):
            raise TypeError("runtime metrics must be scalar values")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("runtime metrics must be finite")
        if isinstance(value, str) and len(value) > _MAX_METRIC_STRING_LENGTH:
            raise ValueError("runtime metric string values must have bounded length")
        bounded[key] = value
    if len(canonical_json(bounded)) > _MAX_METRICS_ENCODED_BYTES:
        raise ValueError("runtime metrics encoded size exceeds the bounded evidence limit")
    return bounded


@dataclass(frozen=True)
class RuntimeHandle:
    """Opaque runtime ownership token for one discrete task."""

    taskId: str


@dataclass(frozen=True)
class RuntimeSample:
    """A bounded state sample; high-rate trajectories are deliberately not represented."""

    running: bool
    terminalState: Literal["SUCCEEDED", "FAILED"] | None = None
    metrics: Mapping[str, RuntimeMetric] = field(default_factory=dict)
    stopReason: str | None = None

    def __post_init__(self) -> None:
        if self.running == (self.terminalState is not None):
            raise ValueError("a runtime sample must be running or have one terminal state")
        object.__setattr__(self, "metrics", _bounded_metrics(self.metrics))


@dataclass(frozen=True)
class RuntimeEvidence:
    """Bounded final runtime evidence returned after a safe stop."""

    metrics: Mapping[str, RuntimeMetric] = field(default_factory=dict)
    stopReason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", _bounded_metrics(self.metrics))


class SimulationRuntime(Protocol):
    """Runtime operations required by the durable discrete-task service."""

    def validate(self, action: ActionDefinition, request: TaskCreateRequest) -> None: ...

    def start(self, action: ActionDefinition, request: TaskCreateRequest) -> RuntimeHandle: ...

    def command(self, handle: RuntimeHandle, parameters: Mapping[str, object]) -> None: ...

    def sample(self, handle: RuntimeHandle) -> RuntimeSample: ...

    def safe_stop(self, handle: RuntimeHandle | None, reason: str) -> RuntimeEvidence: ...

    def status(self) -> RobotStatus: ...
