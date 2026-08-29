"""Deterministic qualification batteries and immutable ROM bundle promotion."""

from __future__ import annotations

import hashlib
import math
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import mujoco
from pydantic import Field, model_validator

from .action_specs import ACTION_RUNTIME_SPECS
from .contracts import (
    ActionDefinition,
    ContractModel,
    ModelArtifact,
    PolicyBundle,
    TaskCreateRequest,
    canonical_json,
    sha256_prefixed,
)
from .main import load_verified_bundle
from .mujoco_runtime import MicroduckMujocoRuntime

_REPORT_PATH = "qualification/qualification-v1.json"
_REPORT_BINDING = "VERIFIED_INPUT_BUNDLE_DIGEST_V1"
_RUNTIME_IDENTIFIER = (
    "mjlab_microduck.rom.mujoco_runtime.MicroduckMujocoRuntime"
)


class QualificationFailed(RuntimeError):
    """A mandatory release action did not pass its declared battery."""


class ReleaseConfigurationError(ValueError):
    """The release policy cannot be executed by the governed runtime."""


class QualificationThresholds(ContractModel):
    """Aggregate pass criteria for one deterministic action battery."""

    minSuccessRate: float = Field(ge=0.0, le=1.0)
    maxFallRate: float = Field(ge=0.0, le=1.0)
    maxMeanTrackingError: float = Field(ge=0.0)
    minMeanDistanceM: float = Field(ge=0.0)
    maxMeanEnergyProxy: float = Field(ge=0.0)
    maxLimitViolations: int = Field(ge=0)
    actionMetric: str = Field(min_length=1)
    actionMetricOperator: Literal["gte", "lte"]
    actionMetricThreshold: float


class ActionQualificationConfig(ContractModel):
    """Code-owned runtime scenario declaration for one release action."""

    actionCode: str = Field(min_length=1)
    mandatory: bool
    terrain: str = Field(min_length=1)
    resetProfile: str = Field(min_length=1)
    seeds: tuple[int, ...] = Field(min_length=1)
    maxSteps: int = Field(gt=0, le=100_000)
    parameters: dict[str, Any]
    thresholds: QualificationThresholds

    @model_validator(mode="after")
    def validate_seeds(self) -> ActionQualificationConfig:
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("qualification seeds must be unique")
        if any(
            isinstance(seed, bool) or not 0 <= seed < 2**32 for seed in self.seeds
        ):
            raise ValueError("qualification seeds must be unsigned 32-bit integers")
        return self


class ReleaseConfiguration(ContractModel):
    """Explicit mandatory/optional policy for one new immutable release."""

    schema_: Literal["MICRODUCK_ROM_RELEASE_V1"] = Field(
        default="MICRODUCK_ROM_RELEASE_V1",
        alias="schema",
        serialization_alias="schema",
    )
    release: str = Field(min_length=1)
    createdAt: datetime
    runtimeSourceCommit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    actions: tuple[ActionQualificationConfig, ...]

    @model_validator(mode="after")
    def validate_actions(self) -> ReleaseConfiguration:
        codes = [action.actionCode for action in self.actions]
        if len(codes) != len(set(codes)):
            raise ValueError("release action codes must be unique")
        if self.createdAt.tzinfo is None:
            raise ValueError("release createdAt must include a UTC offset")
        return self


class QualificationRollout(ContractModel):
    seed: int
    startedAt: datetime
    finishedAt: datetime
    steps: int
    success: bool
    fallen: bool
    trackingError: float | None
    distanceM: float
    energyProxy: float
    limitViolations: int
    maxAbsAction: float
    actionMetric: str
    actionMetricValue: float | None
    stopReason: str


class ActionQualificationResult(ContractModel):
    actionCode: str
    mandatory: bool
    status: Literal["PASSED", "FAILED", "UNAVAILABLE"]
    unavailableReason: str | None = None
    terrain: str
    resetProfile: str
    scenarioProfile: str
    seeds: tuple[int, ...]
    maxSteps: int
    thresholds: QualificationThresholds
    successRate: float
    fallRate: float
    meanTrackingError: float | None
    meanDistanceM: float
    meanEnergyProxy: float
    limitViolations: int
    actionMetricMean: float | None
    runtimeClass: Literal["MicroduckMujocoRuntime"]
    runtimeIdentifier: Literal[
        "mjlab_microduck.rom.mujoco_runtime.MicroduckMujocoRuntime"
    ]
    runtimeSourceCommit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    simulatorVersion: str
    policyDigest: str | None
    modelDigest: str
    sourceCommit: str
    checkpoint: str | None
    runIdentity: str | None
    rollouts: tuple[QualificationRollout, ...]


class QualificationReport(ContractModel):
    schema_: Literal["MICRODUCK_ROM_QUALIFICATION_V1"] = Field(
        default="MICRODUCK_ROM_QUALIFICATION_V1",
        alias="schema",
        serialization_alias="schema",
    )
    generatedAt: datetime
    binding: Literal["VERIFIED_INPUT_BUNDLE_DIGEST_V1"] = _REPORT_BINDING
    subjectBundleId: str
    subjectBundleVersion: str
    subjectBundleDigest: str
    sourceRepository: str
    sourceCommit: str
    runtimeIdentifier: Literal[
        "mjlab_microduck.rom.mujoco_runtime.MicroduckMujocoRuntime"
    ]
    runtimeSourceCommit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    modelDigest: str
    releaseConfigurationDigest: str
    actions: tuple[ActionQualificationResult, ...]


@dataclass(frozen=True)
class PromotedBundle:
    manifest: PolicyBundle
    report: QualificationReport
    output_zip: Path
    artifact_digests: dict[str, str]


def _number(metrics: Mapping[str, object], key: str) -> float | None:
    value = metrics.get(key)
    if (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        return float(value)
    return None


def _integer(metrics: Mapping[str, object], key: str) -> int:
    value = metrics.get(key)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _task_id(bundle_digest: str, action_code: str, seed: int) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "bundleDigest": bundle_digest,
                "actionCode": action_code,
                "seed": seed,
            }
        )
    ).hexdigest()[:32]


def _validate_release_configuration(
    bundle: PolicyBundle, configuration: ReleaseConfiguration
) -> None:
    if configuration.release == bundle.bundleVersion:
        raise ReleaseConfigurationError("promotion requires a new bundle version")
    definitions = {action.actionCode: action for action in bundle.actions}
    configured = {action.actionCode: action for action in configuration.actions}
    unknown = set(configured) - set(definitions)
    if unknown:
        raise ReleaseConfigurationError(f"unknown release actions: {sorted(unknown)}")
    uncovered = {
        code
        for code, action in definitions.items()
        if action.availability == "AVAILABLE" and code not in configured
    }
    if uncovered:
        raise ReleaseConfigurationError(
            f"available actions require explicit release policy: {sorted(uncovered)}"
        )
    for code, declaration in configured.items():
        definition = definitions[code]
        spec = ACTION_RUNTIME_SPECS.get(code)
        if declaration.mandatory and (
            definition.availability != "AVAILABLE" or spec is None or not spec.supported
        ):
            raise ReleaseConfigurationError(
                f"mandatory action {code} is not supported by candidate capabilities"
            )
        if spec is None:
            raise ReleaseConfigurationError(f"action {code} has no code-owned runtime spec")
        if declaration.resetProfile != spec.reset_profile:
            raise ReleaseConfigurationError(
                f"action {code} reset profile does not match code-owned semantics"
            )
        if declaration.terrain != bundle.qualification.get("modelTerrain"):
            raise ReleaseConfigurationError(
                f"action {code} terrain does not match the verified candidate model"
            )
        if definition.availability == "AVAILABLE" and (
            bundle.qualification.get("scenarioProfile") != spec.scenario_profile
        ):
            raise ReleaseConfigurationError(
                f"action {code} scenario profile does not match code-owned semantics"
            )


def _unavailable_result(
    bundle: PolicyBundle,
    declaration: ActionQualificationConfig,
    definition: ActionDefinition,
    runtime_source_commit: str,
) -> ActionQualificationResult:
    spec = ACTION_RUNTIME_SPECS[declaration.actionCode]
    return ActionQualificationResult(
        actionCode=declaration.actionCode,
        mandatory=declaration.mandatory,
        status="UNAVAILABLE",
        unavailableReason=definition.unavailableReason or "CANDIDATE_UNAVAILABLE",
        terrain=declaration.terrain,
        resetProfile=declaration.resetProfile,
        scenarioProfile=spec.scenario_profile,
        seeds=declaration.seeds,
        maxSteps=declaration.maxSteps,
        thresholds=declaration.thresholds,
        successRate=0.0,
        fallRate=0.0,
        meanTrackingError=None,
        meanDistanceM=0.0,
        meanEnergyProxy=0.0,
        limitViolations=0,
        actionMetricMean=None,
        runtimeClass="MicroduckMujocoRuntime",
        runtimeIdentifier=_RUNTIME_IDENTIFIER,
        runtimeSourceCommit=runtime_source_commit,
        simulatorVersion=mujoco.__version__,
        policyDigest=None,
        modelDigest=bundle.model.digest,
        sourceCommit=bundle.sourceCommit,
        checkpoint=None,
        runIdentity=None,
        rollouts=(),
    )


def _qualify_action(
    root: Path,
    bundle: PolicyBundle,
    declaration: ActionQualificationConfig,
    definition: ActionDefinition,
    runtime_source_commit: str,
    timestamp: Callable[[], datetime],
) -> ActionQualificationResult:
    spec = ACTION_RUNTIME_SPECS[declaration.actionCode]
    policy = next(
        item for item in bundle.policies if item.policyRef == definition.policyRef
    )
    rollouts: list[QualificationRollout] = []
    for seed in declaration.seeds:
        started_at = timestamp()
        runtime = MicroduckMujocoRuntime(root, bundle, realtime=False)
        request = TaskCreateRequest(
            schema="MICRODUCK_SIM_TASK_V1",
            taskId=_task_id(bundle.bundleDigest or "", declaration.actionCode, seed),
            actionCode=declaration.actionCode,
            bundleVersion=bundle.bundleVersion,
            bundleDigest=bundle.bundleDigest,
            parameters=declaration.parameters,
            scenario={"terrain": declaration.terrain, "seed": seed},
            leaseMs=(definition.lease.defaultLeaseMs if definition.lease else None),
            requestedBy="rom-qualification",
        )
        runtime.validate(definition, request)
        handle = runtime.start(definition, request)
        sample = None
        try:
            for _ in range(declaration.maxSteps):
                sample = runtime.sample(handle)
                if not sample.running:
                    break
            evidence = runtime.safe_stop(handle, "QUALIFICATION_BATTERY_COMPLETE")
        except Exception:
            runtime.safe_stop(handle, "QUALIFICATION_RUNTIME_ERROR")
            raise
        metrics = dict(evidence.metrics)
        steps = _integer(metrics, "steps")
        fallen = bool(metrics.get("fallen", False))
        terminal_failed = sample is not None and sample.terminalState == "FAILED"
        action_metric_value = _number(metrics, declaration.thresholds.actionMetric)
        rollouts.append(
            QualificationRollout(
                seed=seed,
                startedAt=started_at,
                finishedAt=timestamp(),
                steps=steps,
                success=steps == declaration.maxSteps
                and not fallen
                and not terminal_failed,
                fallen=fallen,
                trackingError=_number(metrics, "trackingError"),
                distanceM=_number(metrics, "baseTravelM") or 0.0,
                energyProxy=_number(metrics, "energyProxy") or 0.0,
                limitViolations=_integer(metrics, "limitViolations"),
                maxAbsAction=_number(metrics, "maxAbsAction") or 0.0,
                actionMetric=declaration.thresholds.actionMetric,
                actionMetricValue=action_metric_value,
                stopReason=sample.stopReason
                if sample is not None and sample.stopReason
                else "MAX_STEPS_REACHED",
            )
        )

    success_rate = _mean([1.0 if item.success else 0.0 for item in rollouts])
    fall_rate = _mean([1.0 if item.fallen else 0.0 for item in rollouts])
    tracking_values = [
        item.trackingError for item in rollouts if item.trackingError is not None
    ]
    action_values = [
        item.actionMetricValue
        for item in rollouts
        if item.actionMetricValue is not None
    ]
    mean_tracking = _mean(tracking_values) if tracking_values else None
    action_mean = _mean(action_values) if len(action_values) == len(rollouts) else None
    mean_distance = _mean([item.distanceM for item in rollouts])
    mean_energy = _mean([item.energyProxy for item in rollouts])
    limit_violations = sum(item.limitViolations for item in rollouts)
    thresholds = declaration.thresholds
    action_metric_passed = action_mean is not None and (
        action_mean >= thresholds.actionMetricThreshold
        if thresholds.actionMetricOperator == "gte"
        else action_mean <= thresholds.actionMetricThreshold
    )
    passed = (
        success_rate >= thresholds.minSuccessRate
        and fall_rate <= thresholds.maxFallRate
        and mean_tracking is not None
        and mean_tracking <= thresholds.maxMeanTrackingError
        and mean_distance >= thresholds.minMeanDistanceM
        and mean_energy <= thresholds.maxMeanEnergyProxy
        and limit_violations <= thresholds.maxLimitViolations
        and action_metric_passed
    )
    return ActionQualificationResult(
        actionCode=declaration.actionCode,
        mandatory=declaration.mandatory,
        status="PASSED" if passed else "FAILED",
        unavailableReason=None if passed else "QUALIFICATION_FAILED",
        terrain=declaration.terrain,
        resetProfile=declaration.resetProfile,
        scenarioProfile=spec.scenario_profile,
        seeds=declaration.seeds,
        maxSteps=declaration.maxSteps,
        thresholds=thresholds,
        successRate=success_rate,
        fallRate=fall_rate,
        meanTrackingError=mean_tracking,
        meanDistanceM=mean_distance,
        meanEnergyProxy=mean_energy,
        limitViolations=limit_violations,
        actionMetricMean=action_mean,
        runtimeClass="MicroduckMujocoRuntime",
        runtimeIdentifier=_RUNTIME_IDENTIFIER,
        runtimeSourceCommit=runtime_source_commit,
        simulatorVersion=mujoco.__version__,
        policyDigest=policy.digest,
        modelDigest=bundle.model.digest,
        sourceCommit=bundle.sourceCommit,
        checkpoint=policy.checkpoint,
        runIdentity=policy.experimentRef,
        rollouts=tuple(rollouts),
    )


def qualify_bundle(
    bundle_root: Path,
    configuration: ReleaseConfiguration,
    *,
    timestamp: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> tuple[PolicyBundle, QualificationReport]:
    """Run bounded batteries through the exact installed runtime implementation."""
    root = Path(bundle_root).resolve()
    bundle = load_verified_bundle(root)
    _validate_release_configuration(bundle, configuration)
    definitions = {action.actionCode: action for action in bundle.actions}
    results: list[ActionQualificationResult] = []
    for declaration in configuration.actions:
        definition = definitions[declaration.actionCode]
        if definition.availability != "AVAILABLE":
            result = _unavailable_result(
                bundle,
                declaration,
                definition,
                configuration.runtimeSourceCommit,
            )
        else:
            result = _qualify_action(
                root,
                bundle,
                declaration,
                definition,
                configuration.runtimeSourceCommit,
                timestamp,
            )
        results.append(result)
        if declaration.mandatory and result.status != "PASSED":
            raise QualificationFailed(
                f"mandatory action {declaration.actionCode} failed qualification"
            )
    report = QualificationReport(
        generatedAt=timestamp(),
        subjectBundleId=bundle.bundleId,
        subjectBundleVersion=bundle.bundleVersion,
        subjectBundleDigest=bundle.bundleDigest,
        sourceRepository=bundle.sourceRepository,
        sourceCommit=bundle.sourceCommit,
        runtimeIdentifier=_RUNTIME_IDENTIFIER,
        runtimeSourceCommit=configuration.runtimeSourceCommit,
        modelDigest=bundle.model.digest,
        releaseConfigurationDigest=sha256_prefixed(configuration),
        actions=tuple(results),
    )
    return bundle, report


def _declared_artifacts(bundle: PolicyBundle) -> list[ModelArtifact]:
    artifacts = [
        bundle.model,
        *(ModelArtifact(path=item.path, digest=item.digest) for item in bundle.policies),
    ]
    for container, key in (
        (bundle.qualification, "artifacts"),
        (bundle.qualification, "modelClosure"),
        (bundle.license, "artifacts"),
    ):
        raw = container.get(key, [])
        if not isinstance(raw, list):
            raise TypeError("bundle artifact declaration must be a list")
        artifacts.extend(ModelArtifact.model_validate(item) for item in raw)
    return artifacts


def promote_qualified_bundle(
    bundle_root: Path,
    output_zip: Path,
    configuration: ReleaseConfiguration,
    bundle: PolicyBundle,
    report: QualificationReport,
) -> PromotedBundle:
    """Create a new deterministic ZIP; never modify or overwrite release inputs."""
    root = Path(bundle_root).resolve()
    output = Path(output_zip).resolve()
    if output.exists():
        raise FileExistsError(f"bundle output already exists: {output}")
    if output.is_relative_to(root):
        raise ValueError("promoted output must remain outside the source bundle")
    if report.subjectBundleDigest != bundle.bundleDigest:
        raise ValueError("qualification report is not bound to the source bundle")

    report_bytes = canonical_json(report)
    report_artifact = ModelArtifact(
        path=_REPORT_PATH,
        digest=f"sha256:{hashlib.sha256(report_bytes).hexdigest()}",
    )
    result_by_code = {item.actionCode: item for item in report.actions}
    promoted_actions: list[ActionDefinition] = []
    for action in bundle.actions:
        result = result_by_code.get(action.actionCode)
        if result is None or result.status == "UNAVAILABLE":
            promoted_actions.append(action)
        elif result.status == "PASSED":
            promoted_actions.append(
                action.model_copy(update={"qualificationRefs": [_REPORT_PATH]})
            )
        else:
            promoted_actions.append(
                action.model_copy(
                    update={
                        "availability": "UNAVAILABLE",
                        "unavailableReason": "QUALIFICATION_FAILED",
                        "qualificationRefs": [_REPORT_PATH],
                    }
                )
            )

    existing_qualification_artifacts = bundle.qualification.get("artifacts", [])
    if not isinstance(existing_qualification_artifacts, list):
        raise TypeError("qualification artifacts must be a list")
    qualification = bundle.qualification | {
        "artifacts": [*existing_qualification_artifacts, report_artifact.model_dump()],
        "binding": _REPORT_BINDING,
        "subjectBundleDigest": bundle.bundleDigest,
        "reportDigest": report_artifact.digest,
    }
    unsigned = bundle.model_copy(
        update={
            "bundleVersion": configuration.release,
            "createdAt": configuration.createdAt,
            "bundleDigest": None,
            "actions": promoted_actions,
            "qualification": qualification,
        }
    )
    contents: dict[str, bytes] = {}
    artifact_digests: dict[str, str] = {}
    for artifact in _declared_artifacts(bundle):
        if artifact.path in contents:
            raise ValueError("bundle declares duplicate artifact paths")
        source = (root / artifact.path).resolve()
        if not source.is_file() or not source.is_relative_to(root):
            raise ValueError("bundle artifact path is invalid")
        content = source.read_bytes()
        digest = f"sha256:{hashlib.sha256(content).hexdigest()}"
        if digest != artifact.digest:
            raise ValueError("bundle artifact changed after qualification")
        contents[artifact.path] = content
        artifact_digests[artifact.path] = digest
    if _REPORT_PATH in contents:
        raise ValueError("qualification report path already exists")
    contents[_REPORT_PATH] = report_bytes
    artifact_digests[_REPORT_PATH] = report_artifact.digest
    digest = sha256_prefixed(
        {
            "manifest": unsigned.model_dump(
                mode="json", by_alias=True, exclude={"bundleDigest"}
            ),
            "artifacts": artifact_digests,
        }
    )
    manifest = unsigned.model_copy(update={"bundleDigest": digest})
    contents["microduck-policy-bundle.json"] = canonical_json(manifest)

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED) as archive:
        for archive_path, content in sorted(contents.items()):
            info = zipfile.ZipInfo(archive_path, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, content)
    return PromotedBundle(manifest, report, output, artifact_digests)


def qualify_and_promote(
    bundle_root: Path,
    output_zip: Path,
    configuration: ReleaseConfiguration,
    *,
    timestamp: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> PromotedBundle:
    """Qualify a verified installed candidate and promote it to a new release."""
    output = Path(output_zip).resolve()
    if output.exists():
        raise FileExistsError(f"bundle output already exists: {output}")
    root = Path(bundle_root).resolve()
    if output.is_relative_to(root):
        raise ValueError("promoted output must remain outside the source bundle")
    bundle, report = qualify_bundle(root, configuration, timestamp=timestamp)
    return promote_qualified_bundle(root, output, configuration, bundle, report)
