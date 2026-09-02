"""Canonical visual evidence that remains trustworthy after deserialization."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import canonical_json, sha256_file
from .evaluation import ArtifactIntegrityError, EvaluationReport
from .schema import SkillSpec


class ReviewError(ValueError):
    """Review evidence is incomplete, forged, or no longer byte-identical."""


_ROLES = ("nominal", "entry", "exit", "stress", "worst_case")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _file(path: Path | None, label: str) -> None:
    if not isinstance(path, Path):
        raise ReviewError(f"{label} path is required")
    if not path.is_file() or path.stat().st_size == 0:
        raise ReviewError(f"{label} must be a non-empty regular file")


@dataclass
class ReviewClip:
    role: str
    scenario_id: str
    seed: int
    path: Path
    policy_digest: str


@dataclass(frozen=True)
class BoundReviewClip:
    role: str
    scenario_id: str
    seed: int
    path: Path
    digest: str
    policy_digest: str

    def as_dict(self) -> dict[str, object]:
        return {"role": self.role, "scenario_id": self.scenario_id, "seed": self.seed,
                "path": str(self.path), "sha256": self.digest, "policy_sha256": self.policy_digest}


@dataclass(frozen=True)
class BaselineComparison:
    evaluation_json: str
    evaluation_digest: str
    policy_path: Path
    policy_digest: str
    metric_summary: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {"evaluation": self.evaluation_json, "evaluation_digest": self.evaluation_digest,
                "policy_path": str(self.policy_path), "policy_sha256": self.policy_digest,
                "metric_summary": self.metric_summary}


def _report(value: str, label: str, *, require_passing: bool = True) -> dict[str, Any]:
    try:
        raw = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ReviewError(f"{label} evaluation must be canonical JSON") from error
    if not isinstance(raw, dict) or canonical_json(raw) != value:
        raise ReviewError(f"{label} evaluation must be canonical JSON")
    required = {"skill_id", "spec_version", "evaluator_revision", "passed", "scenarios", "policy"}
    if set(raw) != required or not isinstance(raw["passed"], bool):
        raise ReviewError(f"{label} evaluation pass status is invalid")
    if not all(isinstance(raw[key], str) and raw[key] for key in ("skill_id", "spec_version", "evaluator_revision")):
        raise ReviewError(f"{label} evaluation identity is invalid")
    policy = raw["policy"]
    if not isinstance(policy, dict) or set(policy) - {"path", "kind", "sha256", "metadata"} or policy.get("kind") != "onnx":
        raise ReviewError(f"{label} evaluation requires an ONNX policy")
    if not all(isinstance(policy.get(key), str) and policy[key] for key in ("path", "sha256")):
        raise ReviewError(f"{label} evaluation policy is invalid")
    if not isinstance(raw["scenarios"], list) or not raw["scenarios"]:
        raise ReviewError(f"{label} evaluation scenarios are required")
    seen: set[str] = set()
    mandatory_passed = True
    threshold_contract: set[tuple[str, str, str, float, bool]] | None = None
    for scenario in raw["scenarios"]:
        if not isinstance(scenario, dict) or not {"scenario_id", "family", "seed", "policy_sha256", "metrics", "threshold_results"} <= set(scenario):
            raise ReviewError(f"{label} evaluation scenario is invalid")
        if not isinstance(scenario["scenario_id"], str) or not scenario["scenario_id"] or scenario["scenario_id"] in seen:
            raise ReviewError(f"{label} evaluation scenario IDs are invalid")
        seen.add(scenario["scenario_id"])
        if not isinstance(scenario["family"], str) or not scenario["family"] or isinstance(scenario["seed"], bool) or not isinstance(scenario["seed"], int):
            raise ReviewError(f"{label} evaluation scenario is invalid")
        if scenario["policy_sha256"] != policy["sha256"]:
            raise ReviewError(f"{label} scenario policy digest mismatch")
        metrics = scenario["metrics"]
        if not isinstance(metrics, dict):
            raise ReviewError(f"{label} scenario metrics are invalid")
        if not scenario["threshold_results"]:
            raise ReviewError(f"{label} threshold results are required")
        signatures: set[tuple[str, str, str, float, bool]] = set()
        metric_names: set[str] = set()
        for result in scenario["threshold_results"]:
            fields = {"scenario_id", "metric_name", "unit", "direction", "limit", "value", "mandatory", "passed", "normalized_violation"}
            if not isinstance(result, dict) or set(result) != fields:
                raise ReviewError(f"{label} threshold result is invalid")
            if result["scenario_id"] != scenario["scenario_id"] or result["metric_name"] not in metrics:
                raise ReviewError(f"{label} threshold scenario binding is invalid")
            if result["direction"] not in {"minimum", "maximum"} or not isinstance(result["unit"], str) or not result["unit"]:
                raise ReviewError(f"{label} threshold result is invalid")
            numeric = ("limit", "value", "normalized_violation")
            if any(isinstance(result[key], bool) or not isinstance(result[key], (int, float)) or not math.isfinite(result[key]) for key in numeric):
                raise ReviewError(f"{label} threshold result is invalid")
            if not isinstance(result["mandatory"], bool) or not isinstance(result["passed"], bool) or metrics[result["metric_name"]] != result["value"]:
                raise ReviewError(f"{label} threshold result is invalid")
            passed = result["value"] >= result["limit"] if result["direction"] == "minimum" else result["value"] <= result["limit"]
            violation = max(0.0, (result["limit"] - result["value"]) if result["direction"] == "minimum" else (result["value"] - result["limit"])) / max(1.0, abs(result["limit"]))
            if result["passed"] is not passed or result["normalized_violation"] != violation:
                raise ReviewError(f"{label} threshold result is inconsistent")
            signature = (result["metric_name"], result["unit"], result["direction"], float(result["limit"]), result["mandatory"])
            if result["metric_name"] in metric_names:
                raise ReviewError(f"{label} duplicate threshold metric name")
            metric_names.add(result["metric_name"])
            if signature in signatures:
                raise ReviewError(f"{label} threshold result keys are not unique")
            signatures.add(signature)
            mandatory_passed = mandatory_passed and (passed or not result["mandatory"])
        if threshold_contract is None:
            threshold_contract = signatures
        elif threshold_contract != signatures:
            raise ReviewError(f"{label} threshold contract differs across scenarios")
    if raw["passed"] is not mandatory_passed:
        raise ReviewError(f"{label} evaluation threshold pass status is inconsistent")
    if require_passing and raw["passed"] is not True:
        raise ReviewError(f"{label} evaluation must record a passing evaluation")
    return raw


def _spec(value: str) -> SkillSpec:
    try:
        raw = json.loads(value)
        if canonical_json(raw) != value:
            raise ValueError
        return SkillSpec.from_dict(raw)
    except (TypeError, ValueError) as error:
        raise ReviewError("skill spec evidence is invalid") from error


def _validate_spec(report: Mapping[str, Any], spec: SkillSpec) -> None:
    if (report["skill_id"], report["spec_version"]) != (spec.id, spec.version):
        raise ReviewError("evaluation does not match bound skill spec")
    families = {scenario["family"] for scenario in report["scenarios"]}
    if not set(spec.held_out_scenarios) <= families:
        raise ReviewError("evaluation omits bound skill spec scenario families")
    expected = {metric.name: (metric.unit, metric.direction, float(metric.limit), metric.mandatory) for metric in spec.metrics}
    for scenario in report["scenarios"]:
        if scenario["seed"] not in spec.evaluation_seeds:
            raise ReviewError("evaluation scenario seed is not in bound skill spec")
        results: dict[str, Mapping[str, Any]] = {}
        for result in scenario["threshold_results"]:
            name = result["metric_name"]
            if name in results:
                raise ReviewError("duplicate spec threshold metric name")
            results[name] = result
        if set(results) != set(expected):
            raise ReviewError("evaluation does not contain the complete spec threshold contract")
        for name, signature in expected.items():
            result = results[name]
            if (result["unit"], result["direction"], float(result["limit"]), result["mandatory"]) != signature:
                raise ReviewError("evaluation threshold does not match bound skill spec")


def _summary(report: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for scenario in report["scenarios"]:
        for result in scenario["threshold_results"]:
            if not isinstance(result, dict) or not isinstance(result.get("metric_name"), str):
                raise ReviewError("evaluation threshold results are invalid")
            value = result.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ReviewError("evaluation metric summary is invalid")
            values[result["metric_name"]] = max(values.get(result["metric_name"], float(value)), float(value))
    return dict(sorted(values.items()))


def _expected(report: Mapping[str, Any]) -> dict[str, tuple[str, int]]:
    families: dict[str, list[Mapping[str, Any]]] = {}
    for scenario in report["scenarios"]:
        families.setdefault(scenario["family"], []).append(scenario)
    selected: dict[str, tuple[str, int]] = {}
    for role in _ROLES[:-1]:
        candidates = families.get(role, [])
        if not candidates:
            raise ReviewError(f"evaluation requires a {role} scenario")
        candidate = min(candidates, key=lambda item: (item["seed"], item["scenario_id"]))
        selected[role] = (candidate["scenario_id"], candidate["seed"])
    def key(item: Mapping[str, Any]) -> tuple[float, str]:
        violations = [result.get("normalized_violation", 0.0) for result in item["threshold_results"]]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in violations):
            raise ReviewError("evaluation normalized violation is invalid")
        return (-max((float(value) for value in violations), default=0.0), item["scenario_id"])
    worst = sorted(report["scenarios"], key=key)[0]
    selected["worst_case"] = (worst["scenario_id"], worst["seed"])
    return selected


@dataclass(frozen=True)
class ReviewBundle:
    evaluation_json: str
    evaluation_digest: str
    policy_digest: str
    clips: tuple[BoundReviewClip, ...]
    metric_summary: dict[str, float]
    passed: bool
    baseline: BaselineComparison | None
    policy_path: Path
    spec_json: str
    spec_digest: str

    @classmethod
    def build(cls, report: EvaluationReport, clip_files: Mapping[str, ReviewClip], *, spec: SkillSpec, baseline: EvaluationReport | None = None) -> ReviewBundle:
        if report.policy is None or report.passed is not True:
            raise ReviewError("review requires a passing evaluation bound to an ONNX policy")
        try:
            report.verify_policy(report.policy.path)
        except ArtifactIntegrityError as error:
            raise ReviewError(str(error)) from error
        raw = _report(canonical_json(report.as_dict()), "candidate")
        spec_json = canonical_json(spec.as_dict())
        _validate_spec(raw, _spec(spec_json))
        expected = _expected(raw)
        clips: list[BoundReviewClip] = []
        for role in _ROLES:
            if role not in clip_files:
                raise ReviewError(f"missing mandatory {role} clip")
            clip = clip_files[role]
            if clip.role != role or (clip.scenario_id, clip.seed) != expected[role]:
                raise ReviewError(f"{role} clip does not bind its evaluated scenario")
            if clip.policy_digest != report.policy.sha256:
                raise ReviewError(f"{role} clip policy digest does not match evaluated policy digest")
            path = Path(clip.path)
            _file(path, f"{role} clip")
            clips.append(BoundReviewClip(role, clip.scenario_id, clip.seed, path, sha256_file(path), clip.policy_digest))
        evaluation_json = canonical_json(report.as_dict())
        baseline_data = cls._baseline(raw, baseline) if baseline else None
        bundle = cls(evaluation_json, _digest(evaluation_json), report.policy.sha256, tuple(clips), _summary(raw), True, baseline_data, Path(report.policy.path), spec_json, _digest(spec_json))
        bundle.verify()
        return bundle

    @staticmethod
    def _baseline(candidate: Mapping[str, Any], baseline: EvaluationReport) -> BaselineComparison:
        if baseline.policy is None:
            raise ReviewError("baseline requires an ONNX policy")
        try:
            baseline.verify_policy(baseline.policy.path)
        except ArtifactIntegrityError as error:
            raise ReviewError(f"baseline {error}") from error
        text = canonical_json(baseline.as_dict())
        raw = _report(text, "baseline", require_passing=False)
        if tuple(raw[key] for key in ("skill_id", "spec_version", "evaluator_revision")) != tuple(candidate[key] for key in ("skill_id", "spec_version", "evaluator_revision")):
            raise ReviewError("baseline evaluation identity does not match")
        identity = lambda item: (item["scenario_id"], item["family"], item["seed"])
        if Counter(map(identity, raw["scenarios"])) != Counter(map(identity, candidate["scenarios"])):
            raise ReviewError("baseline scenario multiset does not match")
        return BaselineComparison(text, _digest(text), Path(baseline.policy.path), baseline.policy.sha256, _summary(raw))

    @property
    def skill_id(self) -> str:
        return _report(self.evaluation_json, "candidate")["skill_id"]

    @property
    def spec_version(self) -> str:
        return _report(self.evaluation_json, "candidate")["spec_version"]

    @property
    def skill_spec(self) -> SkillSpec:
        """Return the canonical SkillSpec bound into this verified review bundle."""
        if _digest(self.spec_json) != self.spec_digest:
            raise ReviewError("skill spec digest mismatch")
        return _spec(self.spec_json)

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"evaluation": self.evaluation_json, "evaluation_digest": self.evaluation_digest,
            "policy_sha256": self.policy_digest, "policy_path": str(self.policy_path), "clips": [clip.as_dict() for clip in self.clips],
            "metric_summary": self.metric_summary, "passed": self.passed}
        result["skill_spec"] = self.spec_json
        result["skill_spec_digest"] = self.spec_digest
        if self.baseline is not None:
            result["baseline"] = self.baseline.as_dict()
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ReviewBundle:
        clips = tuple(BoundReviewClip(item["role"], item["scenario_id"], item["seed"], Path(item["path"]), item["sha256"], item["policy_sha256"]) for item in raw["clips"])
        base = raw.get("baseline")
        baseline = None if base is None else BaselineComparison(base["evaluation"], base["evaluation_digest"], Path(base["policy_path"]), base["policy_sha256"], dict(base["metric_summary"]))
        return cls(raw["evaluation"], raw["evaluation_digest"], raw["policy_sha256"], clips, dict(raw["metric_summary"]), raw["passed"], baseline, Path(raw["policy_path"]), raw["skill_spec"], raw["skill_spec_digest"])

    @property
    def digest(self) -> str:
        return _digest(canonical_json(self.as_dict()))

    def verify(self) -> None:
        if self.passed is not True or _digest(self.evaluation_json) != self.evaluation_digest:
            raise ReviewError("review requires a passing evaluation")
        report = _report(self.evaluation_json, "candidate")
        spec = self.skill_spec
        _validate_spec(report, spec)
        policy = report["policy"]
        if not isinstance(self.policy_path, Path):
            raise ReviewError("policy path is required")
        if policy["sha256"] != self.policy_digest or policy["path"] != str(self.policy_path):
            raise ReviewError("evaluation policy binding does not match review bundle")
        _file(self.policy_path, "policy")
        if sha256_file(self.policy_path) != self.policy_digest:
            raise ReviewError("policy digest mismatch")
        roles = [clip.role for clip in self.clips]
        if set(roles) != set(_ROLES) or len(roles) != len(_ROLES):
            raise ReviewError("mandatory clip roles must appear exactly once")
        expected = _expected(report)
        for clip in self.clips:
            if (clip.scenario_id, clip.seed) != expected[clip.role] or clip.policy_digest != self.policy_digest:
                raise ReviewError(f"{clip.role} clip scenario or policy binding is invalid")
            _file(clip.path, f"{clip.role} clip")
            if sha256_file(clip.path) != clip.digest:
                raise ReviewError(f"{clip.role} clip digest mismatch")
        if self.metric_summary != _summary(report):
            raise ReviewError("metric summary does not match evaluation")
        if self.baseline is not None:
            base = self.baseline
            if _digest(base.evaluation_json) != base.evaluation_digest:
                raise ReviewError("baseline evaluation digest mismatch")
            raw = _report(base.evaluation_json, "baseline", require_passing=False)
            _validate_spec(raw, spec)
            if base.metric_summary != _summary(raw):
                raise ReviewError("baseline metric summary does not match evaluation")
            if raw["policy"]["sha256"] != base.policy_digest or raw["policy"]["path"] != str(base.policy_path):
                raise ReviewError("baseline policy binding is invalid")
            _file(base.policy_path, "baseline policy")
            if sha256_file(base.policy_path) != base.policy_digest:
                raise ReviewError("baseline policy digest mismatch")
            identity = lambda item: (item["scenario_id"], item["family"], item["seed"])
            if tuple(raw[key] for key in ("skill_id", "spec_version", "evaluator_revision")) != tuple(report[key] for key in ("skill_id", "spec_version", "evaluator_revision")) or Counter(map(identity, raw["scenarios"])) != Counter(map(identity, report["scenarios"])):
                raise ReviewError("baseline comparable evaluation evidence does not match")
