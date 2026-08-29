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

from .action_specs import ACTION_RUNTIME_SPECS, STAND_SETTLEMENT_LIMITS
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
from .runtime import canonical_tracking_mean
from .runtime_identity import runtime_revision

QUALIFICATION_REPORT_PATH = "qualification/qualification-v1.json"
RELEASE_CONFIGURATION_PATH = "qualification/release-v1.json"
SUBJECT_MANIFEST_PATH = "qualification/subject-manifest-v1.json"
_REPORT_BINDING = "VERIFIED_INPUT_BUNDLE_DIGEST_V1"
_RUNTIME_IDENTIFIER = "mjlab_microduck.rom.mujoco_runtime.MicroduckMujocoRuntime"


class QualificationFailed(RuntimeError):
    """A mandatory release action did not pass its declared battery."""


class ReleaseConfigurationError(ValueError):
    """The release policy cannot be executed by the governed runtime."""


class QualificationThresholds(ContractModel):
    """Aggregate pass criteria for one deterministic action battery."""

    minSuccessRate: float = Field(ge=0.8, le=1.0, allow_inf_nan=False)
    maxFallRate: float = Field(ge=0.0, le=0.2, allow_inf_nan=False)
    maxMeanTrackingError: float = Field(ge=0.0, le=10.0, allow_inf_nan=False)
    minMeanDistanceM: float = Field(ge=0.0, le=100.0, allow_inf_nan=False)
    maxMeanEnergyProxy: float = Field(ge=0.0, le=10_000.0, allow_inf_nan=False)
    maxActuatorClampSteps: int = Field(ge=0, le=100)
    maxPhysicalJointLimitViolations: int = Field(ge=0, le=0)
    actionMetric: str = Field(min_length=1)
    actionMetricOperator: Literal["gte", "lte"]
    actionMetricThreshold: float = Field(ge=0.0, le=100.0, allow_inf_nan=False)


class ActionQualificationConfig(ContractModel):
    """Code-owned runtime scenario declaration for one release action."""

    actionCode: str = Field(min_length=1)
    mandatory: bool
    terrain: str = Field(min_length=1)
    resetProfile: str = Field(min_length=1)
    seeds: tuple[int, ...] = Field(min_length=3, max_length=16)
    maxSteps: int = Field(ge=100, le=2_000)
    parameters: dict[str, Any]
    thresholds: QualificationThresholds

    @model_validator(mode="after")
    def validate_seeds(self) -> ActionQualificationConfig:
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("qualification seeds must be unique")
        if any(isinstance(seed, bool) or not 0 <= seed < 2**32 for seed in self.seeds):
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
    actionCode: str
    bundleDigest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    policyDigest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    modelDigest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    sourceCommit: str
    checkpoint: str | None = None
    runIdentity: str | None = None
    runtimeIdentifier: Literal[
        "mjlab_microduck.rom.mujoco_runtime.MicroduckMujocoRuntime"
    ]
    runtimeRevision: str = Field(
        pattern=r"^mjlab-microduck@[^+]+\+sha256:[0-9a-f]{64}$"
    )
    simulatorVersion: str
    terrain: str
    resetProfile: str
    scenarioProfile: str
    seed: int
    requestedParameters: dict[str, Any]
    requestedMotion: dict[str, tuple[float, ...]]
    appliedMotion: dict[str, tuple[float, ...]]
    startedAt: datetime
    finishedAt: datetime
    steps: int
    terminalState: Literal["RUNNING", "SUCCEEDED", "FAILED"]
    success: bool
    fallen: bool
    trackingError: float | None
    trackingErrorSum: float | None
    trackingErrorMax: float | None
    trackingSampleCount: int
    distanceM: float
    energyProxy: float
    actuatorClampSteps: int
    physicalJointLimitViolations: int
    settledSteps: int
    uprightSteps: int
    maxAbsAction: float
    actionMetric: str
    actionMetricValue: float | None
    yawRotationRad: float | None = None
    standPoseError: float | None = None
    settledPoseErrorMax: float | None = None
    settledTrunkHeightMinM: float | None = None
    settledTrunkHeightMaxM: float | None = None
    settledTrunkTiltMaxRad: float | None = None
    settledJointSpeedMaxRadps: float | None = None
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
    parameters: dict[str, Any]
    thresholds: QualificationThresholds
    successRate: float
    fallRate: float
    meanTrackingError: float | None = None
    meanDistanceM: float
    meanEnergyProxy: float
    actuatorClampSteps: int
    physicalJointLimitViolations: int
    actionMetricMean: float | None = None
    runtimeClass: Literal["MicroduckMujocoRuntime"]
    runtimeIdentifier: Literal[
        "mjlab_microduck.rom.mujoco_runtime.MicroduckMujocoRuntime"
    ]
    runtimeRevision: str = Field(
        pattern=r"^mjlab-microduck@[^+]+\+sha256:[0-9a-f]{64}$"
    )
    simulatorVersion: str
    policyDigest: str | None = None
    modelDigest: str
    sourceCommit: str
    checkpoint: str | None = None
    runIdentity: str | None = None
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
    runtimeRevision: str = Field(
        pattern=r"^mjlab-microduck@[^+]+\+sha256:[0-9a-f]{64}$"
    )
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


_ACTION_METRIC_DERIVATIONS = {
    "baseTravelM": "DISTANCE",
    "trackingError": "TRACKING_MEAN",
    "standFraction": "UPRIGHT_FRACTION",
    "yawRotationRad": "YAW_ACCUMULATOR",
    "standPoseError": "STAND_FINAL_POSE_ERROR",
}


def _derive_action_metric_value(rollout: QualificationRollout) -> float:
    derivation = _ACTION_METRIC_DERIVATIONS.get(rollout.actionMetric)
    if derivation == "DISTANCE":
        return rollout.distanceM
    if derivation == "TRACKING_MEAN" and rollout.trackingErrorSum is not None:
        return canonical_tracking_mean(
            rollout.trackingErrorSum,
            rollout.trackingSampleCount,
        )
    if derivation == "UPRIGHT_FRACTION":
        return round(rollout.uprightSteps / rollout.steps, 6)
    if derivation == "YAW_ACCUMULATOR" and rollout.yawRotationRad is not None:
        return rollout.yawRotationRad
    if derivation == "STAND_FINAL_POSE_ERROR" and rollout.standPoseError is not None:
        return rollout.standPoseError
    raise ValueError("qualification rollout action metric derivation is undefined")


def validate_release_configuration(
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
        for code, definition in definitions.items()
        if definition.availability == "AVAILABLE" and code not in configured
    }
    if uncovered:
        raise ReleaseConfigurationError(
            f"available bundle actions require explicit release policy: {sorted(uncovered)}"
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
            raise ReleaseConfigurationError(
                f"action {code} has no code-owned runtime spec"
            )
        if declaration.resetProfile != spec.reset_profile:
            raise ReleaseConfigurationError(
                f"action {code} reset profile does not match code-owned semantics"
            )
        if declaration.terrain != spec.qualification_terrain:
            raise ReleaseConfigurationError(
                f"action {code} terrain does not match code-owned qualification"
            )
        if declaration.parameters != dict(spec.qualification_parameters):
            raise ReleaseConfigurationError(
                f"action {code} command does not match code-owned qualification"
            )
        if (
            not spec.qualification_min_seeds
            <= len(declaration.seeds)
            <= spec.qualification_max_seeds
        ):
            raise ReleaseConfigurationError(
                f"action {code} seed count does not match code-owned bounds"
            )
        if (
            not spec.qualification_min_steps
            <= declaration.maxSteps
            <= (spec.qualification_max_steps)
        ):
            raise ReleaseConfigurationError(
                f"action {code} step count does not match code-owned bounds"
            )
        metric_operators = dict(spec.qualification_metric_operators)
        if (
            declaration.thresholds.actionMetric not in spec.metric_keys
            or metric_operators.get(declaration.thresholds.actionMetric)
            != declaration.thresholds.actionMetricOperator
        ):
            raise ReleaseConfigurationError(
                f"action {code} metric does not match code-owned qualification"
            )
        if (
            definition.availability == "AVAILABLE"
            and declaration.thresholds.actionMetric not in _ACTION_METRIC_DERIVATIONS
        ):
            raise ReleaseConfigurationError(
                f"action {code} metric has no code-owned evidence derivation"
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


def _code_owned_unavailable_declaration(action_code: str) -> ActionQualificationConfig:
    """Carry a pre-unavailable catalog entry without caller-selected rollout policy."""
    spec = ACTION_RUNTIME_SPECS[action_code]
    metric, operator = spec.qualification_metric_operators[0]
    return ActionQualificationConfig(
        actionCode=action_code,
        mandatory=False,
        terrain=spec.qualification_terrain,
        resetProfile=spec.reset_profile,
        seeds=(0, 1, 2),
        maxSteps=spec.qualification_min_steps,
        parameters=dict(spec.qualification_parameters),
        thresholds=QualificationThresholds(
            minSuccessRate=1.0,
            maxFallRate=0.0,
            maxMeanTrackingError=10.0,
            minMeanDistanceM=0.0,
            maxMeanEnergyProxy=10_000.0,
            maxActuatorClampSteps=100,
            maxPhysicalJointLimitViolations=0,
            actionMetric=metric,
            actionMetricOperator=operator,
            actionMetricThreshold=0.0 if operator == "gte" else 100.0,
        ),
    )


def release_action_declarations(
    bundle: PolicyBundle, configuration: ReleaseConfiguration
) -> tuple[ActionQualificationConfig, ...]:
    """Expand sparse release policy to exact catalog coverage using only code-owned carryovers."""
    validate_release_configuration(bundle, configuration)
    configured = {action.actionCode: action for action in configuration.actions}
    return tuple(
        configured.get(action.actionCode)
        or _code_owned_unavailable_declaration(action.actionCode)
        for action in bundle.actions
    )


def _unavailable_result(
    bundle: PolicyBundle,
    declaration: ActionQualificationConfig,
    definition: ActionDefinition,
    installed_runtime_revision: str,
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
        parameters=declaration.parameters,
        thresholds=declaration.thresholds,
        successRate=0.0,
        fallRate=0.0,
        meanTrackingError=None,
        meanDistanceM=0.0,
        meanEnergyProxy=0.0,
        actuatorClampSteps=0,
        physicalJointLimitViolations=0,
        actionMetricMean=None,
        runtimeClass="MicroduckMujocoRuntime",
        runtimeIdentifier=_RUNTIME_IDENTIFIER,
        runtimeRevision=installed_runtime_revision,
        simulatorVersion=mujoco.__version__,
        policyDigest=None,
        modelDigest=bundle.model.digest,
        sourceCommit=bundle.sourceCommit,
        checkpoint=None,
        runIdentity=None,
        rollouts=(),
    )


_QUALIFICATION_FAILURE_REASONS = {
    "CONTROL_LOOP_OVERRUN",
    "FALLEN",
    "JOINT_LIMIT",
    "NON_FINITE_POLICY_OUTPUT",
    "NON_FINITE_STATE",
    "RUNTIME_EXCEPTION",
}


def _expected_qualification_motion(
    action_code: str, parameters: Mapping[str, Any]
) -> dict[str, tuple[float, ...]]:
    spec = ACTION_RUNTIME_SPECS[action_code]
    zero = {
        "twist": (0.0, 0.0, 0.0),
        "headPose": (0.0, 0.0, 0.0, 0.0),
        "bodyPose": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    }
    if spec.command_profile == "TWIST_VELOCITY":
        return zero | {
            "twist": (
                float(parameters["vxMps"]),
                float(parameters["vyMps"]),
                float(parameters["yawRateRadps"]),
            )
        }
    if spec.command_profile in {"SIT_FLAG_ZERO", "ZERO_TWIST_LEASE"}:
        return zero
    raise ValueError(f"action {action_code} has no governed qualification motion")


def _validate_rollout_semantics(
    bundle: PolicyBundle,
    declaration: ActionQualificationConfig,
    definition: ActionDefinition,
    rollout: QualificationRollout,
    installed_runtime_revision: str,
) -> tuple[float, float]:
    spec = ACTION_RUNTIME_SPECS[declaration.actionCode]
    policy = next(
        item for item in bundle.policies if item.policyRef == definition.policyRef
    )
    expected_identity = {
        "actionCode": declaration.actionCode,
        "bundleDigest": bundle.bundleDigest,
        "policyDigest": policy.digest,
        "modelDigest": bundle.model.digest,
        "sourceCommit": bundle.sourceCommit,
        "checkpoint": policy.checkpoint,
        "runIdentity": policy.experimentRef,
        "runtimeIdentifier": _RUNTIME_IDENTIFIER,
        "runtimeRevision": installed_runtime_revision,
        "simulatorVersion": mujoco.__version__,
        "terrain": declaration.terrain,
        "resetProfile": declaration.resetProfile,
        "scenarioProfile": spec.scenario_profile,
        "requestedParameters": declaration.parameters,
    }
    if any(getattr(rollout, key) != value for key, value in expected_identity.items()):
        raise ValueError("qualification rollout identity is invalid")
    expected_motion = _expected_qualification_motion(
        declaration.actionCode, declaration.parameters
    )
    if (
        rollout.requestedMotion != expected_motion
        or rollout.appliedMotion != expected_motion
    ):
        raise ValueError("qualification rollout command identity is invalid")

    integer_counts = (
        rollout.steps,
        rollout.trackingSampleCount,
        rollout.actuatorClampSteps,
        rollout.physicalJointLimitViolations,
        rollout.settledSteps,
        rollout.uprightSteps,
    )
    numeric_values = (
        rollout.trackingError,
        rollout.trackingErrorSum,
        rollout.trackingErrorMax,
        rollout.distanceM,
        rollout.energyProxy,
        rollout.maxAbsAction,
        rollout.actionMetricValue,
        rollout.yawRotationRad,
        rollout.standPoseError,
        rollout.settledPoseErrorMax,
        rollout.settledTrunkHeightMinM,
        rollout.settledTrunkHeightMaxM,
        rollout.settledTrunkTiltMaxRad,
        rollout.settledJointSpeedMaxRadps,
    )
    if any(
        value is not None and not math.isfinite(float(value))
        for value in numeric_values
    ):
        raise ValueError("qualification rollout numeric evidence must be finite")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in integer_counts
    ):
        raise ValueError("qualification rollout counts are invalid")
    if (
        not 1 <= rollout.steps <= declaration.maxSteps
        or rollout.actuatorClampSteps > rollout.steps
        or rollout.physicalJointLimitViolations
        > rollout.steps * bundle.actionContract.dimension
        or rollout.settledSteps > rollout.steps
        or rollout.uprightSteps > rollout.steps
    ):
        raise ValueError("qualification rollout counts exceed step bounds")
    nonnegative_values = (
        rollout.trackingError,
        rollout.trackingErrorSum,
        rollout.trackingErrorMax,
        rollout.distanceM,
        rollout.energyProxy,
        rollout.maxAbsAction,
        rollout.standPoseError,
        rollout.settledPoseErrorMax,
        rollout.settledTrunkHeightMinM,
        rollout.settledTrunkHeightMaxM,
        rollout.settledTrunkTiltMaxRad,
        rollout.settledJointSpeedMaxRadps,
    )
    if any(value is not None and value < 0.0 for value in nonnegative_values):
        raise ValueError("qualification rollout norm evidence must be nonnegative")
    if any(
        value is None
        for value in (
            rollout.trackingError,
            rollout.trackingErrorSum,
            rollout.trackingErrorMax,
        )
    ):
        raise ValueError("qualification rollout tracking evidence is incomplete")
    assert rollout.trackingError is not None
    assert rollout.trackingErrorSum is not None
    assert rollout.trackingErrorMax is not None
    if rollout.trackingSampleCount != rollout.steps:
        raise ValueError("qualification rollout tracking evidence is incomplete")
    derived_tracking_mean = canonical_tracking_mean(
        rollout.trackingErrorSum,
        rollout.trackingSampleCount,
    )
    if (
        rollout.trackingError != derived_tracking_mean
        or rollout.trackingErrorMax + 1e-6 < derived_tracking_mean
    ):
        raise ValueError("qualification rollout tracking evidence is incomplete")

    if rollout.actionMetric != declaration.thresholds.actionMetric:
        raise ValueError("qualification rollout action metric identity is invalid")
    if rollout.actionMetricValue is None:
        raise ValueError("qualification rollout action metric is missing")
    derived_action_metric = _derive_action_metric_value(rollout)
    if rollout.actionMetricValue != derived_action_metric:
        raise ValueError("qualification rollout action metric evidence is inconsistent")
    metric_domain = dict(spec.qualification_metric_domains).get(rollout.actionMetric)
    if metric_domain == "NONNEGATIVE" and derived_action_metric < 0.0:
        raise ValueError("qualification rollout action metric must be nonnegative")
    if metric_domain == "UNIT_INTERVAL" and not (0.0 <= derived_action_metric <= 1.0):
        raise ValueError("qualification rollout action metric must be a fraction")
    if metric_domain not in {"NONNEGATIVE", "SIGNED", "UNIT_INTERVAL"}:
        raise ValueError("qualification rollout action metric domain is undefined")

    stand_fields = (
        rollout.standPoseError,
        rollout.settledPoseErrorMax,
        rollout.settledTrunkHeightMinM,
        rollout.settledTrunkHeightMaxM,
        rollout.settledTrunkTiltMaxRad,
        rollout.settledJointSpeedMaxRadps,
    )
    if declaration.actionCode != "VELSTAND_VELOCITY" and rollout.uprightSteps:
        raise ValueError("qualification rollout upright accumulator is invalid")
    if declaration.actionCode != "SWIZZLE" and rollout.yawRotationRad is not None:
        raise ValueError("qualification rollout yaw accumulator is invalid")
    if declaration.actionCode != "STAND":
        if rollout.settledSteps or any(value is not None for value in stand_fields):
            raise ValueError("qualification rollout settled evidence is invalid")
    else:
        if rollout.standPoseError is None:
            raise ValueError("qualification rollout settled pose evidence is missing")
        settled_window = stand_fields[1:]
        if rollout.settledSteps == 0:
            if any(value is not None for value in settled_window):
                raise ValueError("qualification rollout settled evidence is invalid")
        else:
            if any(value is None for value in settled_window):
                raise ValueError("qualification rollout settled evidence is incomplete")
            assert rollout.settledPoseErrorMax is not None
            assert rollout.settledTrunkHeightMinM is not None
            assert rollout.settledTrunkHeightMaxM is not None
            assert rollout.settledTrunkTiltMaxRad is not None
            assert rollout.settledJointSpeedMaxRadps is not None
            limits = STAND_SETTLEMENT_LIMITS
            if (
                rollout.settledSteps > limits.required_consecutive_steps
                or rollout.settledPoseErrorMax > limits.pose_error_max_rad
                or rollout.settledTrunkHeightMinM < limits.trunk_height_min_m
                or rollout.settledTrunkHeightMaxM > limits.trunk_height_max_m
                or rollout.settledTrunkHeightMinM > rollout.settledTrunkHeightMaxM
                or rollout.settledTrunkTiltMaxRad > limits.trunk_tilt_max_rad
                or rollout.settledJointSpeedMaxRadps > limits.joint_speed_max_radps
                or rollout.standPoseError > rollout.settledPoseErrorMax + 1e-6
                or rollout.settledPoseErrorMax > rollout.trackingErrorMax + 1e-6
                or (
                    rollout.settledSteps == rollout.trackingSampleCount
                    and not math.isclose(
                        rollout.settledPoseErrorMax,
                        rollout.trackingErrorMax,
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    )
                )
            ):
                raise ValueError("qualification rollout settled evidence is invalid")

    if (
        rollout.startedAt.tzinfo is None
        or rollout.finishedAt.tzinfo is None
        or rollout.finishedAt < rollout.startedAt
    ):
        raise ValueError("qualification rollout timestamps are invalid")
    if rollout.fallen and (
        rollout.success
        or rollout.terminalState != "FAILED"
        or rollout.stopReason != "FALLEN"
    ):
        raise ValueError("qualification rollout fall evidence is inconsistent")

    if spec.execution_mode == "CONTINUOUS_LEASE":
        successful = (
            rollout.steps == declaration.maxSteps
            and rollout.terminalState == "RUNNING"
            and rollout.stopReason == spec.qualification_success_stop_reason
            and not rollout.fallen
        )
        if rollout.success != successful:
            raise ValueError("qualification continuous success evidence is invalid")
        if not successful and (
            rollout.terminalState != "FAILED"
            or rollout.stopReason not in _QUALIFICATION_FAILURE_REASONS
            or (rollout.stopReason == "FALLEN") != rollout.fallen
        ):
            raise ValueError("qualification continuous terminal evidence is invalid")
        return derived_tracking_mean, derived_action_metric

    if rollout.terminalState == "SUCCEEDED":
        completion_valid = (
            spec.completion_profile == "STAND_POSE_SETTLED"
            and rollout.stopReason == spec.qualification_success_stop_reason
            and rollout.settledSteps == spec.qualification_min_settled_steps
            and spec.qualification_completion_metric_max is not None
            and derived_action_metric <= spec.qualification_completion_metric_max
            and not rollout.fallen
        )
        if not rollout.success or not completion_valid:
            raise ValueError("qualification discrete completion evidence is invalid")
    elif rollout.terminalState == "RUNNING":
        if (
            rollout.success
            or rollout.fallen
            or rollout.steps != declaration.maxSteps
            or rollout.stopReason != "MAX_STEPS_REACHED"
        ):
            raise ValueError("qualification discrete timeout evidence is invalid")
    elif (
        rollout.success
        or rollout.stopReason not in _QUALIFICATION_FAILURE_REASONS
        or (rollout.stopReason == "FALLEN") != rollout.fallen
    ):
        raise ValueError("qualification discrete failure evidence is invalid")
    return derived_tracking_mean, derived_action_metric


def recompute_action_qualification(
    bundle: PolicyBundle,
    declaration: ActionQualificationConfig,
    definition: ActionDefinition,
    rollouts: tuple[QualificationRollout, ...],
    installed_runtime_revision: str,
) -> ActionQualificationResult:
    """Derive the only valid result from governed configuration and raw rollouts."""
    if definition.availability != "AVAILABLE":
        if rollouts:
            raise ValueError(
                "unavailable action must not contain qualification rollouts"
            )
        return _unavailable_result(
            bundle, declaration, definition, installed_runtime_revision
        )

    spec = ACTION_RUNTIME_SPECS[declaration.actionCode]
    policy = next(
        item for item in bundle.policies if item.policyRef == definition.policyRef
    )
    rollout_seeds = tuple(item.seed for item in rollouts)
    if rollout_seeds != declaration.seeds or len(set(rollout_seeds)) != len(
        rollout_seeds
    ):
        raise ValueError("qualification rollouts do not cover exact unique seeds")
    tracking_values = []
    action_values = []
    for rollout in rollouts:
        derived_tracking_mean, derived_action_metric = _validate_rollout_semantics(
            bundle,
            declaration,
            definition,
            rollout,
            installed_runtime_revision,
        )
        tracking_values.append(derived_tracking_mean)
        action_values.append(derived_action_metric)

    success_rate = _mean([1.0 if item.success else 0.0 for item in rollouts])
    fall_rate = _mean([1.0 if item.fallen else 0.0 for item in rollouts])
    mean_tracking = _mean(tracking_values)
    action_mean = _mean(action_values)
    mean_distance = _mean([item.distanceM for item in rollouts])
    mean_energy = _mean([item.energyProxy for item in rollouts])
    actuator_clamp_steps = sum(item.actuatorClampSteps for item in rollouts)
    physical_joint_limit_violations = sum(
        item.physicalJointLimitViolations for item in rollouts
    )
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
        and actuator_clamp_steps <= thresholds.maxActuatorClampSteps
        and physical_joint_limit_violations
        <= thresholds.maxPhysicalJointLimitViolations
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
        parameters=declaration.parameters,
        thresholds=thresholds,
        successRate=success_rate,
        fallRate=fall_rate,
        meanTrackingError=mean_tracking,
        meanDistanceM=mean_distance,
        meanEnergyProxy=mean_energy,
        actuatorClampSteps=actuator_clamp_steps,
        physicalJointLimitViolations=physical_joint_limit_violations,
        actionMetricMean=action_mean,
        runtimeClass="MicroduckMujocoRuntime",
        runtimeIdentifier=_RUNTIME_IDENTIFIER,
        runtimeRevision=installed_runtime_revision,
        simulatorVersion=mujoco.__version__,
        policyDigest=policy.digest,
        modelDigest=bundle.model.digest,
        sourceCommit=bundle.sourceCommit,
        checkpoint=policy.checkpoint,
        runIdentity=policy.experimentRef,
        rollouts=rollouts,
    )


def _qualify_action(
    root: Path,
    bundle: PolicyBundle,
    declaration: ActionQualificationConfig,
    definition: ActionDefinition,
    installed_runtime_revision: str,
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
            status = runtime.status()
            evidence = runtime.safe_stop(handle, "QUALIFICATION_BATTERY_COMPLETE")
        except Exception:
            runtime.safe_stop(handle, "QUALIFICATION_RUNTIME_ERROR")
            raise
        metrics = dict(evidence.metrics)
        expected_identities = {
            "actionCode": declaration.actionCode,
            "bundleDigest": bundle.bundleDigest,
            "onnxDigest": policy.digest,
            "mjcfDigest": bundle.model.digest,
            "sourceCommit": bundle.sourceCommit,
            "checkpoint": policy.checkpoint,
            "runIdentity": policy.experimentRef,
            "terrainIdentity": declaration.terrain,
            "rngSeed": seed,
            "scenarioProfile": spec.scenario_profile,
            "resetProfile": declaration.resetProfile,
        }
        if any(metrics.get(key) != value for key, value in expected_identities.items()):
            raise QualificationFailed(
                f"runtime evidence identity mismatch for {declaration.actionCode}"
            )
        steps = _integer(metrics, "steps")
        fallen = bool(metrics.get("fallen", False))
        terminal_failed = sample is not None and sample.terminalState == "FAILED"
        terminal_succeeded = sample is not None and sample.terminalState == "SUCCEEDED"
        succeeded = (
            terminal_succeeded
            if spec.execution_mode == "DISCRETE"
            else steps == declaration.maxSteps and not terminal_failed
        )
        action_metric_value = _number(metrics, declaration.thresholds.actionMetric)
        rollouts.append(
            QualificationRollout(
                actionCode=declaration.actionCode,
                bundleDigest=bundle.bundleDigest,
                policyDigest=policy.digest,
                modelDigest=bundle.model.digest,
                sourceCommit=bundle.sourceCommit,
                checkpoint=policy.checkpoint,
                runIdentity=policy.experimentRef,
                runtimeIdentifier=_RUNTIME_IDENTIFIER,
                runtimeRevision=installed_runtime_revision,
                simulatorVersion=mujoco.__version__,
                terrain=declaration.terrain,
                resetProfile=declaration.resetProfile,
                scenarioProfile=spec.scenario_profile,
                seed=seed,
                requestedParameters=declaration.parameters,
                requestedMotion=status.requestedMotion,
                appliedMotion=status.appliedMotion,
                startedAt=started_at,
                finishedAt=timestamp(),
                steps=steps,
                terminalState=(
                    sample.terminalState
                    if sample is not None and sample.terminalState is not None
                    else "RUNNING"
                ),
                success=succeeded and not fallen,
                fallen=fallen,
                trackingError=_number(metrics, "trackingError"),
                trackingErrorSum=_number(metrics, "trackingErrorSum"),
                trackingErrorMax=_number(metrics, "trackingErrorMax"),
                trackingSampleCount=_integer(metrics, "trackingErrorSamples"),
                distanceM=_number(metrics, "baseTravelM") or 0.0,
                energyProxy=_number(metrics, "energyProxy") or 0.0,
                actuatorClampSteps=_integer(metrics, "actuatorClampSteps"),
                physicalJointLimitViolations=_integer(
                    metrics, "physicalJointLimitViolations"
                ),
                settledSteps=_integer(metrics, "standSettledSteps"),
                uprightSteps=_integer(metrics, "uprightSteps"),
                maxAbsAction=_number(metrics, "maxAbsAction") or 0.0,
                actionMetric=declaration.thresholds.actionMetric,
                actionMetricValue=action_metric_value,
                yawRotationRad=_number(metrics, "yawRotationRad"),
                standPoseError=_number(metrics, "standPoseError"),
                settledPoseErrorMax=_number(metrics, "settledPoseErrorMax"),
                settledTrunkHeightMinM=_number(metrics, "settledHeightMinM"),
                settledTrunkHeightMaxM=_number(metrics, "settledHeightMaxM"),
                settledTrunkTiltMaxRad=_number(metrics, "settledTiltMaxRad"),
                settledJointSpeedMaxRadps=_number(metrics, "settledJointSpeedMaxRadps"),
                stopReason=sample.stopReason
                if sample is not None and sample.stopReason
                else "MAX_STEPS_REACHED",
            )
        )

    return recompute_action_qualification(
        bundle,
        declaration,
        definition,
        tuple(rollouts),
        installed_runtime_revision,
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
    declarations = release_action_declarations(bundle, configuration)
    installed_runtime_revision = runtime_revision()
    definitions = {action.actionCode: action for action in bundle.actions}
    results: list[ActionQualificationResult] = []
    for declaration in declarations:
        definition = definitions[declaration.actionCode]
        if definition.availability != "AVAILABLE":
            result = _unavailable_result(
                bundle,
                declaration,
                definition,
                installed_runtime_revision,
            )
        else:
            result = _qualify_action(
                root,
                bundle,
                declaration,
                definition,
                installed_runtime_revision,
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
        runtimeRevision=installed_runtime_revision,
        modelDigest=bundle.model.digest,
        releaseConfigurationDigest=sha256_prefixed(configuration),
        actions=tuple(results),
    )
    return bundle, report


def _declared_artifacts(bundle: PolicyBundle) -> list[ModelArtifact]:
    artifacts = [
        bundle.model,
        *(
            ModelArtifact(path=item.path, digest=item.digest)
            for item in bundle.policies
        ),
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


def _validate_qualification_correspondence(
    root: Path,
    configuration: ReleaseConfiguration,
    bundle: PolicyBundle,
    report: QualificationReport,
) -> None:
    installed = load_verified_bundle(root)
    if installed != bundle:
        raise ValueError("qualification bundle does not match installed candidate")
    effective_declarations = release_action_declarations(bundle, configuration)
    if (
        report.subjectBundleId != bundle.bundleId
        or report.subjectBundleVersion != bundle.bundleVersion
        or report.subjectBundleDigest != bundle.bundleDigest
        or report.sourceRepository != bundle.sourceRepository
        or report.sourceCommit != bundle.sourceCommit
        or report.modelDigest != bundle.model.digest
        or report.releaseConfigurationDigest != sha256_prefixed(configuration)
        or report.runtimeIdentifier != _RUNTIME_IDENTIFIER
        or report.runtimeRevision != runtime_revision()
    ):
        raise ValueError("qualification report identity does not match candidate")
    results = {item.actionCode: item for item in report.actions}
    declarations = {item.actionCode: item for item in effective_declarations}
    definitions = {item.actionCode: item for item in bundle.actions}
    if (
        len(results) != len(report.actions)
        or len(declarations) != len(effective_declarations)
        or set(results) != set(definitions)
        or set(declarations) != set(definitions)
    ):
        raise ValueError("qualification report action coverage is not exact")
    for code, definition in definitions.items():
        result = results[code]
        declaration = declarations[code]
        expected = recompute_action_qualification(
            bundle,
            declaration,
            definition,
            result.rollouts,
            report.runtimeRevision,
        )
        if canonical_json(result) != canonical_json(expected):
            raise ValueError("qualification result does not match release policy")
        if declaration.mandatory and expected.status != "PASSED":
            raise ValueError("qualification mandatory action did not pass")


def promoted_action_definition(
    subject_action: ActionDefinition,
    result: ActionQualificationResult,
) -> ActionDefinition:
    """Reconstruct the only promoted action contract allowed by qualification."""
    from .action_catalog import code_owned_action_definition

    availability = subject_action.availability
    unavailable_reason = subject_action.unavailableReason
    if result.status == "FAILED":
        availability = "UNAVAILABLE"
        unavailable_reason = "QUALIFICATION_FAILED"
    return code_owned_action_definition(
        subject_action.actionCode,
        availability=availability,
        policy_ref=subject_action.policyRef,
        unavailable_reason=unavailable_reason,
        qualification_refs=[QUALIFICATION_REPORT_PATH],
    )


def _promote_qualified_bundle(
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
    _validate_qualification_correspondence(root, configuration, bundle, report)

    report_bytes = canonical_json(report)
    configuration_bytes = canonical_json(configuration)
    subject_manifest_bytes = canonical_json(bundle)
    report_artifact = ModelArtifact(
        path=QUALIFICATION_REPORT_PATH,
        digest=f"sha256:{hashlib.sha256(report_bytes).hexdigest()}",
    )
    configuration_artifact = ModelArtifact(
        path=RELEASE_CONFIGURATION_PATH,
        digest=f"sha256:{hashlib.sha256(configuration_bytes).hexdigest()}",
    )
    subject_manifest_artifact = ModelArtifact(
        path=SUBJECT_MANIFEST_PATH,
        digest=f"sha256:{hashlib.sha256(subject_manifest_bytes).hexdigest()}",
    )
    result_by_code = {item.actionCode: item for item in report.actions}
    promoted_actions: list[ActionDefinition] = []
    for action in bundle.actions:
        result = result_by_code.get(action.actionCode)
        if result is None:
            raise ValueError("qualification report does not cover every bundle action")
        promoted_actions.append(promoted_action_definition(action, result))

    existing_qualification_artifacts = bundle.qualification.get("artifacts", [])
    if not isinstance(existing_qualification_artifacts, list):
        raise TypeError("qualification artifacts must be a list")
    qualification = bundle.qualification | {
        "artifacts": [
            *existing_qualification_artifacts,
            report_artifact.model_dump(),
            configuration_artifact.model_dump(),
            subject_manifest_artifact.model_dump(),
        ],
        "binding": _REPORT_BINDING,
        "reportPath": QUALIFICATION_REPORT_PATH,
        "subjectBundleDigest": bundle.bundleDigest,
        "subjectBundleId": bundle.bundleId,
        "subjectBundleVersion": bundle.bundleVersion,
        "reportDigest": report_artifact.digest,
        "releaseConfigurationPath": RELEASE_CONFIGURATION_PATH,
        "releaseConfigurationDigest": configuration_artifact.digest,
        "subjectManifestPath": SUBJECT_MANIFEST_PATH,
        "subjectManifestDigest": subject_manifest_artifact.digest,
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
    new_artifacts = (
        (QUALIFICATION_REPORT_PATH, report_bytes, report_artifact.digest),
        (
            RELEASE_CONFIGURATION_PATH,
            configuration_bytes,
            configuration_artifact.digest,
        ),
        (
            SUBJECT_MANIFEST_PATH,
            subject_manifest_bytes,
            subject_manifest_artifact.digest,
        ),
    )
    for artifact_path, artifact_bytes, artifact_digest in new_artifacts:
        if artifact_path in contents:
            raise ValueError("qualification artifact path already exists")
        contents[artifact_path] = artifact_bytes
        artifact_digests[artifact_path] = artifact_digest
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
    protected_source_roots: tuple[Path, ...] = (),
) -> PromotedBundle:
    """Qualify a verified installed candidate and promote it to a new release."""
    output = Path(output_zip).resolve()
    if output.exists():
        raise FileExistsError(f"bundle output already exists: {output}")
    root = Path(bundle_root).resolve()
    if output.is_relative_to(root):
        raise ValueError("promoted output must remain outside the source bundle")
    for protected_root in protected_source_roots:
        if output.is_relative_to(Path(protected_root).resolve()):
            raise ValueError(
                "promoted output must remain outside protected source roots"
            )
    bundle, report = qualify_bundle(root, configuration, timestamp=timestamp)
    return _promote_qualified_bundle(root, output, configuration, bundle, report)
