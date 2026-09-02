"""Immutable, validated records shared by the Next RL workspace.

These records intentionally depend only on the standard library and the runtime
policy constants.  JSON is the interchange format, so every record provides an
explicit ``from_dict`` boundary rather than accepting unvalidated dictionaries.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar

from mjlab_microduck.publish.manifest import (
    ACTION_LEN,
    MODEL_API,
    OBS_LEN,
    ROBOT,
)

_SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SchemaError(ValueError):
    """A workspace schema document is malformed or semantically invalid."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{name} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise SchemaError(f"{name} keys must be strings")
    return value


def _fields(raw: Any, required: set[str], optional: set[str], name: str) -> Mapping[str, Any]:
    value = _mapping(raw, name)
    missing = required - value.keys()
    if missing:
        raise SchemaError(f"{name} missing required field {min(missing)!r}")
    unknown = value.keys() - required - optional
    if unknown:
        raise SchemaError(f"{name} has unknown field {min(unknown)!r}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{name} must be an integer")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SchemaError(f"{name} must be finite")
    return float(value)


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SchemaError(f"{name} must be a sequence of strings")
    result = tuple(_text(item, name) for item in value)
    if len(set(result)) != len(result):
        raise SchemaError(f"{name} contains duplicates")
    return result


def _integers(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SchemaError(f"{name} must be a sequence of integers")
    result = tuple(_integer(item, name) for item in value)
    if len(set(result)) != len(result):
        raise SchemaError(f"{name} contains duplicates")
    return result


def _metadata(value: Any) -> Mapping[str, Any]:
    data = _mapping(value, "metadata")
    return MappingProxyType({key: _freeze(item) for key, item in data.items()})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in _mapping(value, "metadata").items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise SchemaError("metadata must contain only finite JSON-compatible values")


def _as_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _as_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_as_json(item) for item in value]
    if hasattr(value, "as_dict"):
        return value.as_dict()
    return value


@dataclass(frozen=True)
class PolicyContract:
    model_api: int
    obs_len: int
    action_len: int
    robot_model: str
    control_hz: int
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}), repr=False)

    @classmethod
    def microduck(cls) -> PolicyContract:
        return cls(MODEL_API, OBS_LEN, ACTION_LEN, str(ROBOT["model"]), int(ROBOT["control_hz"]))

    @classmethod
    def from_dict(cls, raw: Any) -> PolicyContract:
        value = _fields(
            raw,
            {"model_api", "obs_len", "action_len", "robot_model", "control_hz"},
            {"metadata"},
            "policy contract",
        )
        contract = cls(
            _integer(value["model_api"], "model_api"),
            _integer(value["obs_len"], "obs_len"),
            _integer(value["action_len"], "action_len"),
            _text(value["robot_model"], "robot_model"),
            _integer(value["control_hz"], "control_hz"),
            _metadata(value.get("metadata", {})),
        )
        if min(contract.model_api, contract.obs_len, contract.action_len, contract.control_hz) <= 0:
            raise SchemaError("policy contract dimensions and control_hz must be positive")
        return contract

    def as_dict(self) -> dict[str, Any]:
        result = {
            "model_api": self.model_api,
            "obs_len": self.obs_len,
            "action_len": self.action_len,
            "robot_model": self.robot_model,
            "control_hz": self.control_hz,
        }
        if self.metadata:
            result["metadata"] = _as_json(self.metadata)
        return result


@dataclass(frozen=True)
class MetricThreshold:
    name: str
    unit: str
    direction: str
    limit: float
    mandatory: bool = True
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}), repr=False)

    DIRECTIONS: ClassVar[tuple[str, ...]] = ("minimum", "maximum")

    @classmethod
    def from_dict(cls, raw: Any) -> MetricThreshold:
        value = _fields(raw, {"name", "unit", "direction", "limit"}, {"mandatory", "metadata"}, "metric")
        direction = value["direction"]
        if direction not in cls.DIRECTIONS:
            raise SchemaError(f"metric direction must be one of {cls.DIRECTIONS}")
        mandatory = value.get("mandatory", True)
        if not isinstance(mandatory, bool):
            raise SchemaError("metric mandatory must be a boolean")
        return cls(
            _text(value["name"], "metric name"),
            _text(value["unit"], "metric unit"),
            direction,
            _number(value["limit"], "metric limit"),
            mandatory,
            _metadata(value.get("metadata", {})),
        )

    def as_dict(self) -> dict[str, Any]:
        result = {"name": self.name, "unit": self.unit, "direction": self.direction, "limit": self.limit}
        if not self.mandatory:
            result["mandatory"] = False
        if self.metadata:
            result["metadata"] = _as_json(self.metadata)
        return result


@dataclass(frozen=True)
class SkillSpec:
    id: str
    version: str
    description: str
    contract: PolicyContract
    metrics: tuple[MetricThreshold, ...]
    training_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    aliases: tuple[str, ...] = ()
    entry_states: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}), repr=False)
    exit_states: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}), repr=False)
    commands: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}), repr=False)
    held_out_scenarios: tuple[str, ...] = ()
    curriculum_stages: tuple[Mapping[str, Any], ...] = ()
    allowed_parent_capabilities: tuple[str, ...] = ()
    rendering_views: tuple[str, ...] = ()
    minimum_review_clips: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}), repr=False)

    @classmethod
    def from_dict(cls, raw: Any) -> SkillSpec:
        optional = {
            "aliases", "entry_states", "exit_states", "commands", "held_out_scenarios", "curriculum_stages",
            "allowed_parent_capabilities", "rendering_views", "minimum_review_clips", "metadata",
        }
        value = _fields(raw, {"id", "version", "description", "contract", "metrics", "training_seeds", "evaluation_seeds"}, optional, "skill")
        version = _text(value["version"], "skill version")
        if not _SEMVER.fullmatch(version):
            raise SchemaError("skill version must be semantic version X.Y.Z")
        raw_metrics = value["metrics"]
        if isinstance(raw_metrics, (str, bytes)) or not isinstance(raw_metrics, Sequence) or not raw_metrics:
            raise SchemaError("metrics must be a non-empty sequence")
        metrics = tuple(MetricThreshold.from_dict(item) for item in raw_metrics)
        if len({metric.name for metric in metrics}) != len(metrics):
            raise SchemaError("duplicate metric names")
        training_seeds = _integers(value["training_seeds"], "training_seeds")
        evaluation_seeds = _integers(value["evaluation_seeds"], "evaluation_seeds")
        if set(training_seeds) & set(evaluation_seeds):
            raise SchemaError("training_seeds and evaluation_seeds overlap")
        raw_stages = value.get("curriculum_stages", ())
        if isinstance(raw_stages, (str, bytes)) or not isinstance(raw_stages, Sequence):
            raise SchemaError("curriculum_stages must be a sequence of mappings")
        return cls(
            _text(value["id"], "skill id"), version, _text(value["description"], "skill description"),
            PolicyContract.from_dict(value["contract"]), metrics, training_seeds, evaluation_seeds,
            _strings(value.get("aliases", ()), "aliases"),
            _metadata(value.get("entry_states", {})), _metadata(value.get("exit_states", {})),
            _metadata(value.get("commands", {})), _strings(value.get("held_out_scenarios", ()), "held_out_scenarios"),
            tuple(_metadata(stage) for stage in raw_stages),
            _strings(value.get("allowed_parent_capabilities", ()), "allowed_parent_capabilities"),
            _strings(value.get("rendering_views", ()), "rendering_views"),
            _strings(value.get("minimum_review_clips", ()), "minimum_review_clips"),
            _metadata(value.get("metadata", {})),
        )

    def as_dict(self) -> dict[str, Any]:
        result = {
            "id": self.id, "version": self.version, "description": self.description,
            "contract": self.contract.as_dict(), "metrics": [metric.as_dict() for metric in self.metrics],
            "training_seeds": list(self.training_seeds), "evaluation_seeds": list(self.evaluation_seeds),
        }
        for key, item in (
            ("aliases", self.aliases), ("entry_states", self.entry_states), ("exit_states", self.exit_states),
            ("commands", self.commands), ("held_out_scenarios", self.held_out_scenarios),
            ("curriculum_stages", self.curriculum_stages), ("allowed_parent_capabilities", self.allowed_parent_capabilities),
            ("rendering_views", self.rendering_views), ("minimum_review_clips", self.minimum_review_clips),
            ("metadata", self.metadata),
        ):
            if item:
                result[key] = _as_json(item)
        return result


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    kind: str
    sha256: str
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}), repr=False)

    KINDS: ClassVar[tuple[str, ...]] = ("onnx", "policy", "checkpoint", "video", "manifest", "report")

    @classmethod
    def from_dict(cls, raw: Any) -> ArtifactRef:
        value = _fields(raw, {"path", "kind", "sha256"}, {"metadata"}, "artifact")
        kind = value["kind"]
        if kind not in cls.KINDS:
            raise SchemaError(f"artifact kind must be one of {cls.KINDS}")
        digest = _text(value["sha256"], "artifact sha256")
        if not _SHA256.fullmatch(digest):
            raise SchemaError("artifact sha256 must be a lowercase SHA-256 digest")
        return cls(_text(value["path"], "artifact path"), kind, digest, _metadata(value.get("metadata", {})))

    def as_dict(self) -> dict[str, Any]:
        result = {"path": self.path, "kind": self.kind, "sha256": self.sha256}
        if self.metadata:
            result["metadata"] = _as_json(self.metadata)
        return result


@dataclass(frozen=True)
class EvaluationRef:
    kind: str
    policy_sha256: str
    report_path: str | None = None
    passed: bool | None = None
    metric_results: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}), repr=False)
    runtime_repository: str | None = None
    runtime_commit: str | None = None
    approval_provenance: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}), repr=False)

    KINDS: ClassVar[tuple[str, ...]] = ("evaluation_report", "legacy_runtime_shipped")

    @classmethod
    def from_dict(cls, raw: Any) -> EvaluationRef:
        value = _mapping(raw, "evaluation")
        kind = value["kind"]
        if kind not in cls.KINDS:
            raise SchemaError(f"evaluation kind must be one of {cls.KINDS}")
        if kind == "evaluation_report":
            value = _fields(
                value,
                {"kind", "policy_sha256", "report_path", "passed", "metric_results"},
                {"approval_provenance", "metadata"},
                "evaluation report",
            )
        else:
            value = _fields(
                value,
                {"kind", "policy_sha256", "runtime_repository", "runtime_commit", "approval_provenance"},
                {"metadata"},
                "legacy evaluation",
            )
        digest = _text(value["policy_sha256"], "evaluation policy_sha256")
        if not _SHA256.fullmatch(digest):
            raise SchemaError("evaluation policy_sha256 must be a lowercase SHA-256 digest")
        if kind == "evaluation_report":
            report_path = _text(value["report_path"], "report_path")
            if not isinstance(value["passed"], bool):
                raise SchemaError("evaluation passed must be a boolean")
            metric_results = _mapping(value["metric_results"], "metric_results")
            parsed_metrics = {name: _number(result, f"metric_results.{name}") for name, result in metric_results.items()}
            approval = value.get("approval_provenance")
            if approval is not None:
                approval = _text(approval, "approval_provenance")
            return cls(
                kind,
                digest,
                report_path,
                value["passed"],
                MappingProxyType(parsed_metrics),
                approval_provenance=approval,
                metadata=_metadata(value.get("metadata", {})),
            )
        return cls(
            kind, digest, runtime_repository=_text(value["runtime_repository"], "runtime_repository"),
            runtime_commit=_text(value["runtime_commit"], "runtime_commit"),
            approval_provenance=_text(value["approval_provenance"], "approval_provenance"),
            metadata=_metadata(value.get("metadata", {})),
        )

    def has_metric_evidence(self, metric_name: str) -> bool:
        return self.kind == "evaluation_report" and metric_name in self.metric_results

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind, "policy_sha256": self.policy_sha256}
        if self.kind == "evaluation_report":
            result.update({"report_path": self.report_path, "passed": self.passed, "metric_results": _as_json(self.metric_results)})
        else:
            result.update({"runtime_repository": self.runtime_repository, "runtime_commit": self.runtime_commit, "approval_provenance": self.approval_provenance})
        if self.metadata:
            result["metadata"] = _as_json(self.metadata)
        return result


@dataclass(frozen=True)
class Capability:
    id: str
    version: str
    aliases: tuple[str, ...]
    robot_model: str
    contract: PolicyContract
    status: str
    policy: ArtifactRef | None = None
    evaluation: EvaluationRef | None = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}), repr=False)

    STATUSES: ClassVar[tuple[str, ...]] = ("available", "validated", "review_pending", "learned", "superseded")

    @classmethod
    def from_dict(cls, raw: Any) -> Capability:
        value = _fields(raw, {"id", "version", "aliases", "robot_model", "contract", "status"}, {"policy", "evaluation", "metadata"}, "capability")
        version = _text(value["version"], "capability version")
        if not _SEMVER.fullmatch(version):
            raise SchemaError("capability version must be semantic version X.Y.Z")
        status = value["status"]
        if status not in cls.STATUSES:
            raise SchemaError(f"capability status must be one of {cls.STATUSES}")
        policy = ArtifactRef.from_dict(value["policy"]) if "policy" in value else None
        evaluation = EvaluationRef.from_dict(value["evaluation"]) if "evaluation" in value else None
        if status == "learned" and (policy is None or evaluation is None):
            raise SchemaError("learned capability requires policy and evaluation evidence")
        if status == "learned" and evaluation is not None:
            if evaluation.kind != "evaluation_report":
                raise SchemaError("learned capability requires an evaluation_report")
            if evaluation.passed is not True:
                raise SchemaError("learned capability requires a passing evaluation")
            if not evaluation.approval_provenance:
                raise SchemaError("learned capability requires approval evidence")
        if policy is not None and evaluation is not None and policy.sha256 != evaluation.policy_sha256:
            raise SchemaError("capability policy and evaluation digest mismatch")
        return cls(_text(value["id"], "capability id"), version, _strings(value["aliases"], "aliases"), _text(value["robot_model"], "robot_model"), PolicyContract.from_dict(value["contract"]), status, policy, evaluation, _metadata(value.get("metadata", {})))

    def as_dict(self) -> dict[str, Any]:
        result = {"id": self.id, "version": self.version, "aliases": list(self.aliases), "robot_model": self.robot_model, "contract": self.contract.as_dict(), "status": self.status}
        if self.policy is not None:
            result["policy"] = self.policy.as_dict()
        if self.evaluation is not None:
            result["evaluation"] = self.evaluation.as_dict()
        if self.metadata:
            result["metadata"] = _as_json(self.metadata)
        return result


@dataclass(frozen=True)
class ExperimentManifest:
    skill_id: str
    spec_version: str
    task_id: str
    contract: PolicyContract
    code_digest: str
    seed: int
    runner_id: str
    status: str
    environment_config: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}), repr=False)
    agent_config: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}), repr=False)
    parent_policy_digest: str | None = None
    dirty_patch_digest: str | None = None
    created_at: str | None = None
    output_dir: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}), repr=False)

    STATUSES: ClassVar[tuple[str, ...]] = ("planned", "pending", "running", "succeeded", "failed", "interrupted")

    @classmethod
    def from_dict(cls, raw: Any) -> ExperimentManifest:
        required = {"skill_id", "spec_version", "task_id", "contract", "code_digest", "seed", "runner_id", "status"}
        optional = {"environment_config", "agent_config", "parent_policy_digest", "dirty_patch_digest", "created_at", "output_dir", "metadata"}
        value = _fields(raw, required, optional, "experiment")
        version = _text(value["spec_version"], "spec_version")
        if not _SEMVER.fullmatch(version):
            raise SchemaError("spec_version must be semantic version X.Y.Z")
        status = value["status"]
        if status not in cls.STATUSES:
            raise SchemaError(f"experiment status must be one of {cls.STATUSES}")
        for key in ("code_digest", "parent_policy_digest", "dirty_patch_digest"):
            if key in value and value[key] is not None and not _SHA256.fullmatch(_text(value[key], key)):
                raise SchemaError(f"{key} must be a lowercase SHA-256 digest")
        created_at = value.get("created_at")
        if created_at is not None:
            created_at = _text(created_at, "created_at")
        output_dir = value.get("output_dir")
        if output_dir is not None:
            output_dir = _text(output_dir, "output_dir")
        return cls(_text(value["skill_id"], "skill_id"), version, _text(value["task_id"], "task_id"), PolicyContract.from_dict(value["contract"]), _text(value["code_digest"], "code_digest"), _integer(value["seed"], "seed"), _text(value["runner_id"], "runner_id"), status, _metadata(value.get("environment_config", {})), _metadata(value.get("agent_config", {})), value.get("parent_policy_digest"), value.get("dirty_patch_digest"), created_at, output_dir, _metadata(value.get("metadata", {})))

    def as_dict(self) -> dict[str, Any]:
        result = {"skill_id": self.skill_id, "spec_version": self.spec_version, "task_id": self.task_id, "contract": self.contract.as_dict(), "code_digest": self.code_digest, "seed": self.seed, "runner_id": self.runner_id, "status": self.status}
        for key, item in (("environment_config", self.environment_config), ("agent_config", self.agent_config), ("parent_policy_digest", self.parent_policy_digest), ("dirty_patch_digest", self.dirty_patch_digest), ("created_at", self.created_at), ("output_dir", self.output_dir), ("metadata", self.metadata)):
            if item:
                result[key] = _as_json(item)
        return result
