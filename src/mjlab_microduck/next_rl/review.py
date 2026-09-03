"""Canonical visual evidence that remains trustworthy after deserialization."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio

from .artifacts import canonical_json, sha256_file
from .evaluation import (
    ArtifactIntegrityError,
    EvaluationError,
    EvaluationReport,
    preflight_onnx,
)
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


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReviewError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReviewError(f"{label} must be a non-empty string")
    return value


def _decode_video(path: Path, label: str) -> None:
    if path.suffix.lower() != ".mp4":
        raise ReviewError(f"{label} video must be an MP4 container")
    try:
        with path.open("rb") as video:
            container_header = video.read(12)
    except OSError as error:
        raise ReviewError(f"{label} video must be a readable MP4 container") from error
    if len(container_header) < 12 or container_header[4:8] != b"ftyp":
        raise ReviewError(f"{label} video must be an MP4 container")
    try:
        first_frame = iio.imread(path, index=0, plugin="FFMPEG")
        second_frame = iio.imread(path, index=1, plugin="FFMPEG")
    except Exception as error:
        raise ReviewError(f"{label} video must decode at least two temporal frames") from error
    if any(getattr(frame, "size", 0) <= 0 for frame in (first_frame, second_frame)):
        raise ReviewError(f"{label} video must decode at least two temporal frames")


def _write_json_once(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(canonical_json(value).encode("utf-8"))
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite immutable renderer evidence {path}") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@dataclass
class ReviewClip:
    role: str
    scenario_id: str
    seed: int
    path: Path
    policy_digest: str


@dataclass(frozen=True)
class RendererEvidence:
    """Create-once renderer sidecar bound to exact evaluation and video bytes."""

    role: str
    scenario_id: str
    seed: int
    policy_sha256: str
    evaluation_digest: str
    video_path: Path
    video_sha256: str
    renderer_revision: str
    evidence_path: Path
    evidence_sha256: str

    def contract_dict(self) -> dict[str, object]:
        return {
            "evaluation_digest": self.evaluation_digest,
            "policy_sha256": self.policy_sha256,
            "renderer_revision": self.renderer_revision,
            "role": self.role,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "video_path": str(self.video_path),
            "video_sha256": self.video_sha256,
        }

    @classmethod
    def load(cls, path: str | Path) -> RendererEvidence:
        sidecar = Path(path)
        _file(sidecar, "renderer evidence")
        try:
            raw = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ReviewError("renderer evidence must be canonical JSON") from error
        fields = {
            "evaluation_digest",
            "policy_sha256",
            "renderer_revision",
            "role",
            "scenario_id",
            "seed",
            "video_path",
            "video_sha256",
        }
        if not isinstance(raw, dict) or set(raw) != fields or canonical_json(raw) != sidecar.read_text(encoding="utf-8"):
            raise ReviewError("renderer evidence must be canonical JSON with exact fields")
        seed = raw["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ReviewError("renderer evidence seed must be an integer")
        evidence = cls(
            _text(raw["role"], "renderer evidence role"),
            _text(raw["scenario_id"], "renderer evidence scenario_id"),
            seed,
            _sha256(raw["policy_sha256"], "renderer evidence policy_sha256"),
            _sha256(raw["evaluation_digest"], "renderer evidence evaluation_digest"),
            Path(_text(raw["video_path"], "renderer evidence video_path")),
            _sha256(raw["video_sha256"], "renderer evidence video_sha256"),
            _text(raw["renderer_revision"], "renderer revision"),
            sidecar,
            sha256_file(sidecar),
        )
        evidence.verify()
        return evidence

    def verify(self) -> None:
        _file(self.evidence_path, "renderer evidence")
        try:
            raw = json.loads(self.evidence_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ReviewError("renderer evidence must be canonical JSON") from error
        if canonical_json(raw) != self.evidence_path.read_text(encoding="utf-8") or raw != self.contract_dict():
            raise ReviewError("renderer evidence sidecar binding is invalid")
        if sha256_file(self.evidence_path) != self.evidence_sha256:
            raise ReviewError("renderer evidence sidecar digest mismatch")
        _file(self.video_path, f"{self.role} clip")
        if sha256_file(self.video_path) != self.video_sha256:
            raise ReviewError(f"{self.role} clip digest mismatch")
        _decode_video(self.video_path, f"{self.role} clip")


def write_renderer_evidence(
    path: str | Path,
    *,
    role: str,
    scenario_id: str,
    seed: int,
    policy_sha256: str,
    evaluation_digest: str,
    video_path: str | Path,
    renderer_revision: str,
) -> RendererEvidence:
    """Atomically create immutable evidence after a renderer has completed a video."""
    video = Path(video_path)
    _file(video, f"{role} clip")
    _decode_video(video, f"{role} clip")
    raw = {
        "evaluation_digest": _sha256(evaluation_digest, "evaluation_digest"),
        "policy_sha256": _sha256(policy_sha256, "policy_sha256"),
        "renderer_revision": _text(renderer_revision, "renderer revision"),
        "role": _text(role, "renderer evidence role"),
        "scenario_id": _text(scenario_id, "renderer evidence scenario_id"),
        "seed": seed,
        "video_path": str(video),
        "video_sha256": sha256_file(video),
    }
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ReviewError("renderer evidence seed must be an integer")
    _write_json_once(Path(path), raw)
    return RendererEvidence.load(path)


@dataclass(frozen=True)
class BoundReviewClip:
    role: str
    scenario_id: str
    seed: int
    path: Path
    digest: str
    policy_digest: str
    evaluation_digest: str
    renderer_revision: str
    evidence_path: Path
    evidence_digest: str

    def as_dict(self) -> dict[str, object]:
        return {"role": self.role, "scenario_id": self.scenario_id, "seed": self.seed,
                "path": str(self.path), "sha256": self.digest, "policy_sha256": self.policy_digest,
                "evaluation_digest": self.evaluation_digest, "renderer_revision": self.renderer_revision,
                "evidence_path": str(self.evidence_path), "evidence_sha256": self.evidence_digest}


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
    directions: dict[str, str] = {}
    for scenario in report["scenarios"]:
        for result in scenario["threshold_results"]:
            if not isinstance(result, dict) or not isinstance(result.get("metric_name"), str):
                raise ReviewError("evaluation threshold results are invalid")
            direction = result.get("direction")
            if direction not in {"minimum", "maximum"}:
                raise ReviewError("evaluation metric direction is invalid")
            previous_direction = directions.setdefault(result["metric_name"], direction)
            if previous_direction != direction:
                raise ReviewError("evaluation metric direction differs across scenarios")
            value = result.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ReviewError("evaluation metric summary is invalid")
            aggregate = min if direction == "minimum" else max
            values[result["metric_name"]] = aggregate(
                values.get(result["metric_name"], float(value)),
                float(value),
            )
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


def _preflight_policy(path: Path, digest: str, label: str) -> None:
    try:
        artifact = preflight_onnx(path)
    except (ArtifactIntegrityError, EvaluationError) as error:
        raise ReviewError(f"{label} ONNX preflight failed: {error}") from error
    if artifact.sha256 != digest:
        raise ReviewError(f"{label} policy digest mismatch")


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
    def build(cls, report: EvaluationReport, clip_files: Mapping[str, RendererEvidence], *, spec: SkillSpec, baseline: EvaluationReport | None = None) -> ReviewBundle:
        if report.policy is None or report.passed is not True:
            raise ReviewError("review requires a passing evaluation bound to an ONNX policy")
        _preflight_policy(Path(report.policy.path), report.policy.sha256, "candidate")
        raw = _report(canonical_json(report.as_dict()), "candidate")
        spec_json = canonical_json(spec.as_dict())
        _validate_spec(raw, _spec(spec_json))
        evaluation_json = canonical_json(report.as_dict())
        evaluation_digest = _digest(evaluation_json)
        expected = _expected(raw)
        clips: list[BoundReviewClip] = []
        for role in _ROLES:
            if role not in clip_files:
                raise ReviewError(f"missing mandatory {role} clip")
            clip = clip_files[role]
            if not isinstance(clip, RendererEvidence):
                raise ReviewError(f"{role} clip requires immutable renderer evidence sidecar")
            clip.verify()
            if clip.role != role or (clip.scenario_id, clip.seed) != expected[role]:
                raise ReviewError(f"{role} clip does not bind its evaluated scenario")
            if clip.policy_sha256 != report.policy.sha256:
                raise ReviewError(f"{role} clip policy digest does not match evaluated policy digest")
            if clip.evaluation_digest != evaluation_digest:
                raise ReviewError(f"{role} clip evaluation digest does not match evaluated report")
            clips.append(BoundReviewClip(
                role, clip.scenario_id, clip.seed, clip.video_path, clip.video_sha256,
                clip.policy_sha256, clip.evaluation_digest, clip.renderer_revision,
                clip.evidence_path, clip.evidence_sha256,
            ))
        baseline_data = cls._baseline(raw, baseline) if baseline else None
        bundle = cls(evaluation_json, evaluation_digest, report.policy.sha256, tuple(clips), _summary(raw), True, baseline_data, Path(report.policy.path), spec_json, _digest(spec_json))
        bundle.verify()
        return bundle

    @staticmethod
    def _baseline(candidate: Mapping[str, Any], baseline: EvaluationReport) -> BaselineComparison:
        if baseline.policy is None:
            raise ReviewError("baseline requires an ONNX policy")
        _preflight_policy(Path(baseline.policy.path), baseline.policy.sha256, "baseline")
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
        clips = tuple(BoundReviewClip(
            item["role"], item["scenario_id"], item["seed"], Path(item["path"]), item["sha256"],
            item["policy_sha256"], item["evaluation_digest"], item["renderer_revision"],
            Path(item["evidence_path"]), item["evidence_sha256"],
        ) for item in raw["clips"])
        base = raw.get("baseline")
        baseline = None if base is None else BaselineComparison(base["evaluation"], base["evaluation_digest"], Path(base["policy_path"]), base["policy_sha256"], dict(base["metric_summary"]))
        bundle = cls(raw["evaluation"], raw["evaluation_digest"], raw["policy_sha256"], clips, dict(raw["metric_summary"]), raw["passed"], baseline, Path(raw["policy_path"]), raw["skill_spec"], raw["skill_spec_digest"])
        bundle.verify()
        return bundle

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
        _preflight_policy(self.policy_path, self.policy_digest, "candidate")
        roles = [clip.role for clip in self.clips]
        if set(roles) != set(_ROLES) or len(roles) != len(_ROLES):
            raise ReviewError("mandatory clip roles must appear exactly once")
        expected = _expected(report)
        for clip in self.clips:
            if (
                (clip.scenario_id, clip.seed) != expected[clip.role]
                or clip.policy_digest != self.policy_digest
                or clip.evaluation_digest != self.evaluation_digest
            ):
                raise ReviewError(f"{clip.role} clip scenario or policy binding is invalid")
            evidence = RendererEvidence(
                clip.role, clip.scenario_id, clip.seed, clip.policy_digest,
                clip.evaluation_digest, clip.path, clip.digest, clip.renderer_revision,
                clip.evidence_path, clip.evidence_digest,
            )
            evidence.verify()
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
            _preflight_policy(base.policy_path, base.policy_digest, "baseline")
            identity = lambda item: (item["scenario_id"], item["family"], item["seed"])
            if tuple(raw[key] for key in ("skill_id", "spec_version", "evaluator_revision")) != tuple(report[key] for key in ("skill_id", "spec_version", "evaluator_revision")) or Counter(map(identity, raw["scenarios"])) != Counter(map(identity, report["scenarios"])):
                raise ReviewError("baseline comparable evaluation evidence does not match")
