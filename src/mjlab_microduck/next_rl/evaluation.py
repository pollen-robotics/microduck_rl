"""Deterministic aggregation for skill-supplied held-out evaluation metrics.

This module deliberately does not simulate a skill.  Skill owners supply the
deterministic scenario metrics; the shared workspace validates and aggregates
that evidence, and binds it to the exact exported ONNX bytes it checked.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from mjlab_microduck.publish.manifest import ManifestError, check_onnx, smoke_run_onnx

from .artifacts import atomic_write_json, sha256_file
from .schema import ArtifactRef, MetricThreshold, SkillSpec


class EvaluationError(ValueError):
    """Held-out evaluation evidence is invalid or does not meet its contract."""


class ArtifactIntegrityError(EvaluationError):
    """A file no longer has the digest recorded by an evaluation report."""


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise EvaluationError(f"{name} must be finite")
    return float(value)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class MetricResult:
    """One threshold checked against one supplied scenario metric."""

    scenario_id: str
    metric_name: str
    unit: str
    direction: str
    limit: float
    value: float
    mandatory: bool
    passed: bool
    normalized_violation: float

    def as_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "metric_name": self.metric_name,
            "unit": self.unit,
            "direction": self.direction,
            "limit": self.limit,
            "value": self.value,
            "mandatory": self.mandatory,
            "passed": self.passed,
            "normalized_violation": self.normalized_violation,
        }


@dataclass(frozen=True)
class ScenarioResult:
    """Metrics supplied by a concrete, deterministic skill scenario."""

    scenario_id: str
    family: str
    seed: int
    metrics: Mapping[str, float]
    threshold_results: tuple[MetricResult, ...] = ()

    def __post_init__(self) -> None:
        _text(self.scenario_id, "scenario_id")
        _text(self.family, "scenario family")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise EvaluationError("scenario seed must be an integer")
        if not isinstance(self.metrics, Mapping) or not self.metrics:
            raise EvaluationError("scenario metrics must be a non-empty mapping")
        parsed: dict[str, float] = {}
        for name, value in self.metrics.items():
            parsed[_text(name, "metric name")] = _finite_number(value, f"metric {name!r}")
        object.__setattr__(self, "metrics", MappingProxyType(parsed))

    @property
    def normalized_violation(self) -> float:
        return max((result.normalized_violation for result in self.threshold_results), default=0.0)

    def as_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "family": self.family,
            "seed": self.seed,
            "metrics": dict(self.metrics),
            "threshold_results": [result.as_dict() for result in self.threshold_results],
        }


class ScenarioEvaluator(Protocol):
    """Skill-owned evaluator interface; this workspace never invents physics metrics."""

    def evaluate(self, policy: Path, scenario_id: str, family: str, seed: int) -> ScenarioResult:
        """Return deterministic raw metrics for one concrete scenario."""


@dataclass(frozen=True)
class EvaluationReport:
    """Immutable evidence for one policy against one exact skill specification."""

    skill_id: str
    spec_version: str
    evaluator_revision: str
    scenarios: tuple[ScenarioResult, ...]
    passed: bool
    policy: ArtifactRef | None = None

    @property
    def metric_results(self) -> tuple[MetricResult, ...]:
        return tuple(result for scenario in self.scenarios for result in scenario.threshold_results)

    def verify_policy(self, policy: str | Path) -> None:
        """Raise unless *policy* is exactly the byte artifact this report evaluated."""
        if self.policy is None:
            raise ArtifactIntegrityError("evaluation report has no bound policy artifact")
        actual = sha256_file(policy)
        if actual != self.policy.sha256:
            raise ArtifactIntegrityError(
                f"policy digest mismatch: report has {self.policy.sha256}, file has {actual}"
            )

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "skill_id": self.skill_id,
            "spec_version": self.spec_version,
            "evaluator_revision": self.evaluator_revision,
            "passed": self.passed,
            "scenarios": [scenario.as_dict() for scenario in self.scenarios],
        }
        if self.policy is not None:
            result["policy"] = self.policy.as_dict()
        return result

    def write(self, path: str | Path) -> Path:
        """Create a canonical report once; an existing report is immutable evidence."""
        target = Path(path)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite immutable evaluation report {target}")
        atomic_write_json(target, self.as_dict())
        return target


def _validate_spec(spec: SkillSpec) -> None:
    if set(spec.training_seeds) & set(spec.evaluation_seeds):
        raise EvaluationError("training and evaluation seeds overlap")
    for threshold in spec.metrics:
        _text(threshold.name, "metric name")
        _text(threshold.unit, "metric unit")
        if threshold.direction not in MetricThreshold.DIRECTIONS:
            raise EvaluationError(f"unknown metric direction {threshold.direction!r}")
        _finite_number(threshold.limit, f"metric {threshold.name!r} limit")


def _threshold_result(scenario: ScenarioResult, threshold: MetricThreshold) -> MetricResult:
    try:
        value = scenario.metrics[threshold.name]
    except KeyError as error:
        raise EvaluationError(
            f"scenario {scenario.scenario_id!r} is missing metric {threshold.name!r}"
        ) from error
    if threshold.direction == "minimum":
        passed = value >= threshold.limit
        raw_violation = max(0.0, threshold.limit - value)
    else:
        passed = value <= threshold.limit
        raw_violation = max(0.0, value - threshold.limit)
    scale = max(1.0, abs(threshold.limit))
    normalized_violation = raw_violation / scale
    if not math.isfinite(normalized_violation):
        raise EvaluationError(f"metric {threshold.name!r} has a non-finite normalized violation")
    return MetricResult(
        scenario.scenario_id,
        threshold.name,
        threshold.unit,
        threshold.direction,
        float(threshold.limit),
        value,
        threshold.mandatory,
        passed,
        normalized_violation,
    )


def evaluate_thresholds(
    spec: SkillSpec,
    scenarios: Sequence[ScenarioResult],
    *,
    evaluator_revision: str = "unspecified",
    policy: ArtifactRef | None = None,
) -> EvaluationReport:
    """Validate supplied scenario metrics and conjunctively apply mandatory thresholds."""
    _validate_spec(spec)
    _text(evaluator_revision, "evaluator_revision")
    scenario_ids = [scenario.scenario_id for scenario in scenarios]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise EvaluationError("scenario IDs must be unique")
    missing_families = set(spec.held_out_scenarios) - {scenario.family for scenario in scenarios}
    if missing_families:
        raise EvaluationError(f"missing required scenario family {min(missing_families)!r}")
    unexpected_seeds = {scenario.seed for scenario in scenarios} - set(spec.evaluation_seeds)
    if unexpected_seeds:
        raise EvaluationError(f"scenario seed {min(unexpected_seeds)} is not an evaluation seed")

    evaluated: list[ScenarioResult] = []
    for scenario in sorted(scenarios, key=lambda item: item.scenario_id):
        results = tuple(_threshold_result(scenario, threshold) for threshold in spec.metrics)
        evaluated.append(replace(scenario, threshold_results=results))
    passed = all(result.passed or not result.mandatory for scenario in evaluated for result in scenario.threshold_results)
    return EvaluationReport(spec.id, spec.version, evaluator_revision, tuple(evaluated), passed, policy)


def select_worst_case(results: Iterable[ScenarioResult]) -> ScenarioResult:
    """Select the greatest normalized violation, resolving ties by scenario ID."""
    ordered = sorted(results, key=lambda result: (-result.normalized_violation, result.scenario_id))
    if not ordered:
        raise EvaluationError("cannot select a worst case from no scenarios")
    return ordered[0]


def preflight_onnx(policy: str | Path) -> ArtifactRef:
    """Apply the existing daemon shape and finite-output gates to an ONNX policy."""
    path = Path(policy)
    try:
        check_onnx(path)
        smoke_run_onnx(path)
    except ManifestError as error:
        raise EvaluationError(f"ONNX preflight failed: {error}") from error
    return ArtifactRef(str(path), "onnx", sha256_file(path))


def build_report(
    *,
    policy: str | Path,
    spec: SkillSpec,
    scenarios: Sequence[ScenarioResult],
    evaluator_revision: str,
    report_path: str | Path | None = None,
) -> EvaluationReport:
    """Preflight a policy, aggregate supplied metrics, and optionally persist immutable evidence."""
    report = evaluate_thresholds(
        spec,
        scenarios,
        evaluator_revision=evaluator_revision,
        policy=preflight_onnx(policy),
    )
    if report_path is not None:
        report.write(report_path)
    return report
