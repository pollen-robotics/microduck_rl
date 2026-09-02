"""Immutable visual evidence required before a skill can be learned."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .artifacts import canonical_json, sha256_file
from .evaluation import ArtifactIntegrityError, EvaluationReport


class ReviewError(ValueError):
    """Raised when a visual-review bundle is incomplete or no longer trustworthy."""


_MANDATORY_ROLES = ("nominal", "entry", "exit", "stress", "worst_case")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _regular_nonempty(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ReviewError(f"{label} must be a non-empty regular file: {path}")


@dataclass
class ReviewClip:
    """A rendered clip supplied for one role in a human review."""

    role: str
    scenario_id: str
    seed: int
    path: Path
    policy_digest: str


@dataclass(frozen=True)
class BoundReviewClip:
    """A clip manifest bound to the exact bytes presented to the reviewer."""

    role: str
    scenario_id: str
    seed: int
    path: Path
    digest: str
    policy_digest: str

    def as_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "path": str(self.path),
            "sha256": self.digest,
            "policy_sha256": self.policy_digest,
        }


@dataclass(frozen=True)
class BaselineComparison:
    """Optional, seed-matched deterministic evaluation supplied for context."""

    evaluation_digest: str
    metric_summary: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {"evaluation_digest": self.evaluation_digest, "metric_summary": self.metric_summary}


def _metric_summary(report: EvaluationReport) -> dict[str, float]:
    summary: dict[str, float] = {}
    for result in report.metric_results:
        summary[result.metric_name] = max(summary.get(result.metric_name, result.value), result.value)
    return dict(sorted(summary.items()))


def _baseline_comparison(report: EvaluationReport, baseline: EvaluationReport | None) -> BaselineComparison | None:
    if baseline is None:
        return None
    candidate_seeds = {(scenario.family, scenario.seed) for scenario in report.scenarios}
    baseline_seeds = {(scenario.family, scenario.seed) for scenario in baseline.scenarios}
    if candidate_seeds != baseline_seeds:
        raise ReviewError("baseline scenarios must be seed-matched to the evaluation")
    baseline_json = canonical_json(baseline.as_dict())
    return BaselineComparison(_digest(baseline_json), _metric_summary(baseline))


@dataclass(frozen=True)
class ReviewBundle:
    """Canonical evidence package that a named human reviewer must approve."""

    evaluation_json: str
    evaluation_digest: str
    policy_digest: str
    clips: tuple[BoundReviewClip, ...]
    metric_summary: dict[str, float]
    passed: bool
    baseline: BaselineComparison | None = None
    policy_path: Path | None = None

    @classmethod
    def build(
        cls,
        report: EvaluationReport,
        clip_files: Mapping[str, ReviewClip],
        *,
        baseline: EvaluationReport | None = None,
    ) -> ReviewBundle:
        if report.passed is not True:
            raise ReviewError("review requires a passing evaluation")
        if report.policy is None:
            raise ReviewError("review requires an evaluation bound to an ONNX policy")
        try:
            report.verify_policy(report.policy.path)
        except ArtifactIntegrityError as error:
            raise ReviewError(str(error)) from error

        bound: list[BoundReviewClip] = []
        for role in _MANDATORY_ROLES:
            try:
                clip = clip_files[role]
            except KeyError as error:
                raise ReviewError(f"missing mandatory {role} clip") from error
            if clip.role != role:
                raise ReviewError(f"clip role mismatch for {role}")
            if not isinstance(clip.scenario_id, str) or not clip.scenario_id.strip():
                raise ReviewError(f"{role} clip scenario must be non-empty")
            if isinstance(clip.seed, bool) or not isinstance(clip.seed, int):
                raise ReviewError(f"{role} clip seed must be an integer")
            if clip.policy_digest != report.policy.sha256:
                raise ReviewError(f"{role} clip policy digest does not match evaluated policy digest")
            path = Path(clip.path)
            _regular_nonempty(path, f"{role} clip")
            bound.append(BoundReviewClip(role, clip.scenario_id, clip.seed, path, sha256_file(path), clip.policy_digest))

        evaluation_json = canonical_json(report.as_dict())
        return cls(
            evaluation_json,
            _digest(evaluation_json),
            report.policy.sha256,
            tuple(bound),
            _metric_summary(report),
            report.passed,
            _baseline_comparison(report, baseline),
            Path(report.policy.path),
        )

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "evaluation_digest": self.evaluation_digest,
            "evaluation": self.evaluation_json,
            "policy_sha256": self.policy_digest,
            "clips": [clip.as_dict() for clip in self.clips],
            "metric_summary": self.metric_summary,
            "passed": self.passed,
        }
        if self.baseline is not None:
            result["baseline"] = self.baseline.as_dict()
        return result

    @property
    def digest(self) -> str:
        return _digest(canonical_json(self.as_dict()))

    def verify(self) -> None:
        """Raise unless the policy and every reviewed clip retain their exact bytes."""
        if _digest(self.evaluation_json) != self.evaluation_digest:
            raise ReviewError("evaluation digest mismatch")
        try:
            evaluation = json.loads(self.evaluation_json)
        except json.JSONDecodeError as error:
            raise ReviewError("evaluation evidence must be JSON") from error
        if not isinstance(evaluation, dict) or canonical_json(evaluation) != self.evaluation_json:
            raise ReviewError("evaluation evidence must be canonical JSON")
        if evaluation.get("passed") is not True or self.passed is not True:
            raise ReviewError("review requires a passing evaluation")
        policy = evaluation.get("policy")
        if not isinstance(policy, dict) or policy.get("sha256") != self.policy_digest:
            raise ReviewError("evaluation policy digest does not match review bundle")
        if self.policy_path is not None:
            _regular_nonempty(self.policy_path, "policy")
            if sha256_file(self.policy_path) != self.policy_digest:
                raise ReviewError("policy digest mismatch")
        for clip in self.clips:
            _regular_nonempty(clip.path, f"{clip.role} clip")
            if sha256_file(clip.path) != clip.digest:
                raise ReviewError(f"{clip.role} clip digest mismatch")
