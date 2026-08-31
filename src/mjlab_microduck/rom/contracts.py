"""Stable wire contracts shared by MicroDuck policy bundles and the ROM API."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Any, Literal, get_origin

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    StringConstraints,
    field_validator,
    model_validator,
)

POLICY_BUNDLE_SCHEMA = "MICRODUCK_POLICY_BUNDLE_V1"
SIM_TASK_SCHEMA = "MICRODUCK_SIM_TASK_V1"
BIPED_POSE_SCHEMA = "BIPED_POSE_V1"
OBSERVATION_CONTRACT = "MICRODUCK_OBS_61_V1"
ACTION_CONTRACT = "MICRODUCK_ACTION_14_V1"

CONTROLLED_SERVO_JOINTS = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)
OBSERVATION_FIELDS = (
    "base_ang_vel.roll",
    "base_ang_vel.pitch",
    "base_ang_vel.yaw",
    "projected_gravity.x",
    "projected_gravity.y",
    "projected_gravity.z",
    *(f"joint_pos_rel.{joint}" for joint in CONTROLLED_SERVO_JOINTS),
    *(f"joint_vel_rel.{joint}" for joint in CONTROLLED_SERVO_JOINTS),
    *(f"last_action.{joint}" for joint in CONTROLLED_SERVO_JOINTS),
    "twist.lin_vel_x",
    "twist.lin_vel_y",
    "twist.ang_vel_z",
    "head_pose.neck_pitch",
    "head_pose.head_pitch",
    "head_pose.head_yaw",
    "head_pose.head_roll",
    "body_pose.x",
    "body_pose.y",
    "body_pose.z",
    "body_pose.roll",
    "body_pose.pitch",
    "body_pose.yaw",
)

_TASK_ID_PATTERN = r"^[0-9a-f]{32}$"
_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_RAW_CONTROL_KEY = re.compile(r"joint|torque|pwm|policypath|policyname", re.IGNORECASE)
_RFC3339_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)

_MAX_IDENTIFIER_LENGTH = 128
_MAX_PATH_LENGTH = 512
_MAX_DESCRIPTION_LENGTH = 1_024
_MAX_MANIFEST_COLLECTION_ITEMS = 256
_MAX_MANIFEST_MAP_ITEMS = 128
_MAX_MANIFEST_DEPTH = 8
_MAX_MANIFEST_JSON_BYTES = 65_536
_MAX_PARAMETER_ITEMS = 16
_MAX_PARAMETER_JSON_BYTES = 16_384
_MAX_EVENT_PAYLOAD_ITEMS = 16
_MAX_EVENT_PAYLOAD_JSON_BYTES = 4_096
_MAX_EVIDENCE_METRICS = 32
_MAX_EVIDENCE_JSON_BYTES = 1_024
_MAX_STATUS_MAP_ITEMS = 32
_MAX_STATUS_JSON_BYTES = 4_096
_SIGNED_INT_MIN = -(2**63)
_SIGNED_INT_MAX = 2**63 - 1

_RAW_CONTROL_PROPERTY_PATTERN = (
    "[jJ][oO][iI][nN][tT]|[tT][oO][rR][qQ][uU][eE]|[pP][wW][mM]|"
    "[pP][oO][lL][iI][cC][yY][pP][aA][tT][hH]|"
    "[pP][oO][lL][iI][cC][yY][nN][aA][mM][eE]"
)

BoundedIdentifier = Annotated[
    str, StringConstraints(strict=True, min_length=1, max_length=_MAX_IDENTIFIER_LENGTH)
]
BoundedPath = Annotated[
    str, StringConstraints(strict=True, min_length=1, max_length=_MAX_PATH_LENGTH)
]
BoundedDescription = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=_MAX_DESCRIPTION_LENGTH),
]
BoundedJsonKey = Annotated[
    str, StringConstraints(strict=True, min_length=1, max_length=128)
]
BoundedJsonString = Annotated[str, StringConstraints(strict=True, max_length=1_024)]
LicenseIdentifier = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=128,
        pattern=r"^(?:[A-Za-z0-9][A-Za-z0-9.-]*|LicenseRef-[A-Za-z0-9.-]+)$",
    ),
]

SPDX_LICENSE_IDENTIFIERS = frozenset({"Apache-2.0", "CC-BY-NC-SA-4.0"})


def _strict_finite_float(value: Any) -> float:
    if not isinstance(value, float):
        raise ValueError(  # noqa: TRY004 - Pydantic wraps ValueError as wire validation.
            "wire float values must use the JSON number representation"
        )
    if not math.isfinite(value):
        raise ValueError("wire float values must be finite")
    return value


FiniteFloat = Annotated[
    float, BeforeValidator(_strict_finite_float), Field(allow_inf_nan=False)
]
SignedJsonInteger = Annotated[
    int, Field(strict=True, ge=_SIGNED_INT_MIN, le=_SIGNED_INT_MAX)
]
JsonScalar = BoundedJsonString | SignedJsonInteger | FiniteFloat | StrictBool | None
type JsonValue = (
    BoundedJsonString
    | SignedJsonInteger
    | FiniteFloat
    | StrictBool
    | None
    | Annotated[list["JsonValue"], Field(max_length=_MAX_MANIFEST_COLLECTION_ITEMS)]
    | Annotated[
        dict[BoundedJsonKey, "JsonValue"], Field(max_length=_MAX_MANIFEST_MAP_ITEMS)
    ]
)


def _validate_bounded_json(
    value: Mapping[str, JsonValue],
    *,
    max_items: int,
    max_depth: int,
    max_bytes: int,
) -> Mapping[str, JsonValue]:
    def visit(candidate: JsonValue, depth: int) -> None:
        if depth > max_depth:
            raise ValueError("JSON nesting exceeds the wire limit")
        if isinstance(candidate, Mapping):
            if len(candidate) > _MAX_MANIFEST_MAP_ITEMS:
                raise ValueError("JSON object exceeds the wire item limit")
            for key, nested in candidate.items():
                if not isinstance(key, str) or not key or len(key) > 128:
                    raise ValueError("JSON keys must be bounded non-empty strings")
                visit(nested, depth + 1)
        elif isinstance(candidate, list):
            if len(candidate) > _MAX_MANIFEST_COLLECTION_ITEMS:
                raise ValueError("JSON array exceeds the wire item limit")
            for nested in candidate:
                visit(nested, depth + 1)
        elif isinstance(candidate, float) and not math.isfinite(candidate):
            raise ValueError("JSON numbers must be finite")

    if len(value) > max_items:
        raise ValueError("JSON object exceeds the wire item limit")
    visit(value, 0)
    if len(canonical_json(value)) > max_bytes:
        raise ValueError("JSON object exceeds the encoded wire limit")
    return value


def _manifest_object(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return _validate_bounded_json(
        value,
        max_items=_MAX_MANIFEST_MAP_ITEMS,
        max_depth=_MAX_MANIFEST_DEPTH,
        max_bytes=_MAX_MANIFEST_JSON_BYTES,
    )


def _status_object(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return _validate_bounded_json(
        value,
        max_items=_MAX_STATUS_MAP_ITEMS,
        max_depth=4,
        max_bytes=_MAX_STATUS_JSON_BYTES,
    )


def _reject_raw_control_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            if isinstance(key, str) and _RAW_CONTROL_KEY.search(key):
                raise ValueError(f"raw control key is not permitted: {key}")
            _reject_raw_control_keys(nested_value)
    elif isinstance(value, list):
        for nested_value in value:
            _reject_raw_control_keys(nested_value)


def _parameter_object(value: Mapping[str, JsonScalar]) -> Mapping[str, JsonScalar]:
    _reject_raw_control_keys(value)
    return _validate_bounded_json(
        value,
        max_items=_MAX_PARAMETER_ITEMS,
        max_depth=1,
        max_bytes=_MAX_PARAMETER_JSON_BYTES,
    )


def _event_payload(value: Mapping[str, JsonScalar]) -> Mapping[str, JsonScalar]:
    return _validate_bounded_json(
        value,
        max_items=_MAX_EVENT_PAYLOAD_ITEMS,
        max_depth=1,
        max_bytes=_MAX_EVENT_PAYLOAD_JSON_BYTES,
    )


def _evidence_metrics(value: Mapping[str, JsonScalar]) -> Mapping[str, JsonScalar]:
    return _validate_bounded_json(
        value,
        max_items=_MAX_EVIDENCE_METRICS,
        max_depth=1,
        max_bytes=_MAX_EVIDENCE_JSON_BYTES,
    )


ManifestObject = Annotated[
    dict[BoundedJsonKey, JsonValue],
    Field(
        max_length=_MAX_MANIFEST_MAP_ITEMS,
        json_schema_extra={
            "x-unergy-invariants": {
                "finiteNumbers": True,
                "maxCanonicalJsonBytes": _MAX_MANIFEST_JSON_BYTES,
                "maxDepth": _MAX_MANIFEST_DEPTH,
            }
        },
    ),
    AfterValidator(_manifest_object),
]
StatusObject = Annotated[
    dict[BoundedJsonKey, JsonValue],
    Field(
        max_length=_MAX_STATUS_MAP_ITEMS,
        json_schema_extra={
            "x-unergy-invariants": {
                "finiteNumbers": True,
                "maxCanonicalJsonBytes": _MAX_STATUS_JSON_BYTES,
                "maxDepth": 4,
            }
        },
    ),
    AfterValidator(_status_object),
]
ParameterKey = Annotated[
    str, StringConstraints(strict=True, min_length=1, max_length=64)
]
ParameterObject = Annotated[
    dict[ParameterKey, JsonScalar],
    Field(
        max_length=_MAX_PARAMETER_ITEMS,
        json_schema_extra={
            "propertyNames": {
                "maxLength": 64,
                "minLength": 1,
                "not": {"pattern": _RAW_CONTROL_PROPERTY_PATTERN},
            },
            "x-unergy-invariants": {
                "finiteScalarsOnly": True,
                "maxCanonicalJsonBytes": _MAX_PARAMETER_JSON_BYTES,
                "maxDepth": 1,
                "rejectPropertyNameSubstringsCaseInsensitive": [
                    "joint",
                    "torque",
                    "pwm",
                    "policyPath",
                    "policyName",
                ],
            },
        },
    ),
    AfterValidator(_parameter_object),
]
EventPayload = Annotated[
    dict[ParameterKey, JsonScalar],
    Field(
        max_length=_MAX_EVENT_PAYLOAD_ITEMS,
        json_schema_extra={
            "x-unergy-invariants": {
                "finiteScalarsOnly": True,
                "maxCanonicalJsonBytes": _MAX_EVENT_PAYLOAD_JSON_BYTES,
                "maxDepth": 1,
            }
        },
    ),
    AfterValidator(_event_payload),
]
EvidenceMetrics = Annotated[
    dict[ParameterKey, JsonScalar],
    Field(
        max_length=_MAX_EVIDENCE_METRICS,
        json_schema_extra={
            "x-unergy-invariants": {
                "finiteScalarsOnly": True,
                "maxCanonicalJsonBytes": _MAX_EVIDENCE_JSON_BYTES,
                "maxDepth": 1,
            }
        },
    ),
    AfterValidator(_evidence_metrics),
]


class ContractModel(BaseModel):
    """Base for strict, camel-case wire records."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        validate_by_alias=True,
        validate_by_name=True,
        serialize_by_alias=True,
    )

    @field_validator("*", mode="before")
    @classmethod
    def validate_wire_types(cls, value: Any, info) -> Any:
        field = cls.model_fields.get(info.field_name)
        if field is None:
            return value
        if get_origin(field.annotation) is tuple and isinstance(value, list):
            return tuple(value)
        if field.annotation is not datetime:
            return value
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("date-time values must include an RFC3339 timezone")
            return value
        if not isinstance(value, str) or _RFC3339_PATTERN.fullmatch(value) is None:
            raise ValueError("date-time values must be RFC3339 strings with timezone")
        parsed = datetime.fromisoformat(
            value.removesuffix("Z") + "+00:00" if value.endswith("Z") else value
        )
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("date-time values must include an RFC3339 timezone")
        return parsed


class CompletionContract(ContractModel):
    terminalConditions: list[BoundedIdentifier] = Field(min_length=1, max_length=16)
    maxDurationMs: int = Field(gt=0, le=3_600_000)


class LeaseContract(ContractModel):
    model_config = ConfigDict(
        json_schema_extra={
            "x-unergy-invariants": [
                "minLeaseMs <= defaultLeaseMs <= maxLeaseMs",
                "commandCadenceMs <= minLeaseMs",
            ]
        }
    )

    minLeaseMs: int = Field(gt=0, le=60_000)
    defaultLeaseMs: int = Field(gt=0, le=60_000)
    maxLeaseMs: int = Field(gt=0, le=60_000)
    commandCadenceMs: int = Field(gt=0, le=60_000)
    zeroCommand: ParameterObject
    safeStopBehavior: Literal["ZERO_TWIST"]

    @model_validator(mode="after")
    def validate_lease_bounds(self) -> LeaseContract:
        if not self.minLeaseMs <= self.defaultLeaseMs <= self.maxLeaseMs:
            raise ValueError(
                "lease bounds must satisfy minLeaseMs <= defaultLeaseMs <= maxLeaseMs"
            )
        if self.commandCadenceMs > self.minLeaseMs:
            raise ValueError("commandCadenceMs must not exceed minLeaseMs")
        return self


class ObservationContract(ContractModel):
    identifier: Literal["MICRODUCK_OBS_61_V1"]
    dimension: Literal[61]
    fields: list[BoundedIdentifier] = Field(
        min_length=61,
        max_length=61,
        json_schema_extra={
            "prefixItems": [{"const": field} for field in OBSERVATION_FIELDS]
        },
    )
    units: dict[BoundedJsonKey, BoundedIdentifier] = Field(max_length=61)
    normalization: Literal["BAKED_IN_ONNX", "DECLARED"]

    @model_validator(mode="after")
    def validate_shared_layout(self) -> ObservationContract:
        if self.fields != list(OBSERVATION_FIELDS):
            raise ValueError("observation fields must use the exact shared 61D layout")
        return self


class ActionContract(ContractModel):
    identifier: Literal["MICRODUCK_ACTION_14_V1"]
    dimension: Literal[14]
    joints: list[BoundedIdentifier] = Field(
        min_length=14,
        max_length=14,
        json_schema_extra={
            "prefixItems": [{"const": joint} for joint in CONTROLLED_SERVO_JOINTS]
        },
    )
    units: BoundedIdentifier
    scaling: ManifestObject
    clipping: ManifestObject

    @model_validator(mode="after")
    def validate_controlled_servo_order(self) -> ActionContract:
        if self.joints != list(CONTROLLED_SERVO_JOINTS):
            raise ValueError("action joints must use the exact controlled-servo order")
        return self


class PolicyArtifact(ContractModel):
    policyRef: BoundedIdentifier
    path: BoundedPath
    digest: str = Field(pattern=_DIGEST_PATTERN)
    taskId: BoundedIdentifier | None = None
    runtimeRequirements: ManifestObject = Field(default_factory=dict)
    checkpoint: BoundedPath | None = None
    experimentRef: BoundedPath | None = None


class ModelArtifact(ContractModel):
    path: BoundedPath
    digest: str = Field(pattern=_DIGEST_PATTERN)


class LicenseDeclaration(ContractModel):
    identifier: LicenseIdentifier
    artifactPaths: list[BoundedPath] = Field(min_length=1, max_length=32)

    @field_validator("artifactPaths")
    @classmethod
    def validate_artifact_paths(cls, paths: list[BoundedPath]) -> list[BoundedPath]:
        if len(paths) != len(set(paths)):
            raise ValueError("license artifactPaths must be unique")
        for path in paths:
            parts = path.split("/")
            if (
                not path.startswith("licenses/")
                or "\\" in path
                or any(part in {"", ".", ".."} for part in parts)
            ):
                raise ValueError("license artifactPaths must be safe licenses-relative paths")
        return paths


class ModelAssetLicenseDeclaration(LicenseDeclaration):
    distributionStatus: Literal["DEVELOPMENT_ONLY", "DISTRIBUTION_CLEARED"]


class BundleLicense(ContractModel):
    software: LicenseDeclaration
    modelAssets: ModelAssetLicenseDeclaration
    artifacts: list[ModelArtifact] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_license_contract(self) -> BundleLicense:
        if self.software.identifier not in SPDX_LICENSE_IDENTIFIERS:
            raise ValueError("software license identifier must be an allowed SPDX identifier")
        if (
            self.modelAssets.identifier not in SPDX_LICENSE_IDENTIFIERS
            and not re.fullmatch(r"LicenseRef-[A-Za-z0-9.-]+", self.modelAssets.identifier)
        ):
            raise ValueError(
                "model asset license identifier must be an allowed SPDX identifier or LicenseRef"
            )
        artifact_paths = [artifact.path for artifact in self.artifacts]
        if len(artifact_paths) != len(set(artifact_paths)):
            raise ValueError("license artifact paths must be unique")
        referenced_paths = {
            *self.software.artifactPaths,
            *self.modelAssets.artifactPaths,
        }
        if referenced_paths != set(artifact_paths):
            raise ValueError("license artifact references must exactly match declared artifacts")
        return self


class ActionDefinition(ContractModel):
    model_config = ConfigDict(
        json_schema_extra={
            "allOf": [
                {
                    "if": {
                        "properties": {"availability": {"const": "AVAILABLE"}},
                        "required": ["availability"],
                    },
                    "then": {
                        "required": ["policyRef"],
                        "properties": {"policyRef": {"type": "string", "minLength": 1}},
                    },
                },
                {
                    "if": {
                        "properties": {"executionMode": {"const": "CONTINUOUS_LEASE"}},
                        "required": ["executionMode"],
                    },
                    "then": {
                        "required": ["lease"],
                        "properties": {"lease": {"not": {"type": "null"}}},
                    },
                },
            ]
        }
    )

    actionCode: BoundedIdentifier
    executionMode: Literal["DISCRETE", "CONTINUOUS_LEASE"]
    availability: Literal["AVAILABLE", "UNAVAILABLE"]
    policyRef: BoundedIdentifier | None = None
    unavailableReason: BoundedIdentifier | None = None
    parameterSchema: ManifestObject
    completion: CompletionContract | None = None
    lease: LeaseContract | None = None
    displayName: BoundedIdentifier | None = None
    description: BoundedDescription | None = None
    localizedLabels: dict[BoundedIdentifier, BoundedDescription] | None = Field(
        default=None, max_length=32
    )
    preconditions: ManifestObject | None = None
    safety: ManifestObject | None = None
    qualificationRefs: list[BoundedPath] | None = Field(default=None, max_length=32)

    @model_validator(mode="after")
    def validate_mode_and_artifact(self) -> ActionDefinition:
        if self.availability == "AVAILABLE" and not self.policyRef:
            raise ValueError("available action requires policyRef")
        if self.executionMode == "CONTINUOUS_LEASE" and self.lease is None:
            raise ValueError("continuous action requires lease contract")
        return self


class UnsignedPolicyBundleManifest(ContractModel):
    """Builder-only manifest representation which cannot be published as signed V1."""

    model_config = ConfigDict(
        json_schema_extra={
            "x-unergy-invariants": {
                "finiteNumbers": True,
                "maxCanonicalJsonBytes": _MAX_MANIFEST_JSON_BYTES,
                "maxDepth": _MAX_MANIFEST_DEPTH,
                "semanticValidatorRequired": True,
            }
        }
    )

    schema_: Literal["MICRODUCK_POLICY_BUNDLE_V1"] = Field(
        ..., alias="schema", serialization_alias="schema"
    )
    bundleId: BoundedIdentifier
    bundleVersion: Annotated[
        str,
        StringConstraints(
            strict=True,
            min_length=1,
            max_length=64,
            pattern=r"^(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$",
        ),
    ]
    createdAt: datetime
    sourceRepository: BoundedPath
    sourceCommit: BoundedIdentifier
    robotModel: Literal["MICRODUCK"]
    observationContract: ObservationContract
    actionContract: ActionContract
    model: ModelArtifact
    policies: list[PolicyArtifact] = Field(max_length=_MAX_MANIFEST_COLLECTION_ITEMS)
    actions: list[ActionDefinition] = Field(max_length=_MAX_MANIFEST_COLLECTION_ITEMS)
    qualification: ManifestObject
    license: BundleLicense

    @property
    def schema(self) -> str:
        return self.schema_

    @model_validator(mode="after")
    def validate_unique_references(self) -> UnsignedPolicyBundleManifest:
        policy_refs = [policy.policyRef for policy in self.policies]
        if len(policy_refs) != len(set(policy_refs)):
            raise ValueError("policyRef values must be unique")
        action_codes = [action.actionCode for action in self.actions]
        if len(action_codes) != len(set(action_codes)):
            raise ValueError("actionCode values must be unique")
        known_refs = set(policy_refs)
        for action in self.actions:
            if action.policyRef and action.policyRef not in known_refs:
                raise ValueError("action policyRef must reference a declared policy")
        if len(canonical_json(self)) > _MAX_MANIFEST_JSON_BYTES:
            raise ValueError("policy manifest exceeds the canonical encoded wire limit")
        return self


class PolicyBundle(UnsignedPolicyBundleManifest):
    """Published V1 manifest; a digest is always present and non-null on the wire."""

    bundleDigest: str = Field(pattern=_DIGEST_PATTERN)


def unsigned_policy_bundle_manifest(
    bundle: PolicyBundle | UnsignedPolicyBundleManifest,
) -> UnsignedPolicyBundleManifest:
    """Return the explicit digest-free manifest used as the bundle hash input."""
    if isinstance(bundle, UnsignedPolicyBundleManifest) and not isinstance(
        bundle, PolicyBundle
    ):
        return bundle
    return UnsignedPolicyBundleManifest.model_validate(
        bundle.model_dump(mode="python", by_alias=True, exclude={"bundleDigest"})
    )


def publish_policy_bundle(
    unsigned: UnsignedPolicyBundleManifest,
    artifact_digests: Mapping[str, str],
) -> PolicyBundle:
    """Bind an unsigned manifest to the canonical artifact map and publish strict V1."""
    digest = sha256_prefixed(
        {
            "manifest": unsigned.model_dump(mode="json", by_alias=True),
            "artifacts": artifact_digests,
        }
    )
    return PolicyBundle.model_validate(
        unsigned.model_dump(mode="python", by_alias=True) | {"bundleDigest": digest}
    )


class Scenario(ContractModel):
    """Exact seeded reset scenario supported by the governed V1 runtime."""

    terrain: Literal["flat", "ramp"]
    seed: int = Field(ge=0, le=4_294_967_295)


class TaskCreateRequest(ContractModel):
    schema_: Literal["MICRODUCK_SIM_TASK_V1"] = Field(
        ..., alias="schema", serialization_alias="schema"
    )
    taskId: str = Field(pattern=_TASK_ID_PATTERN)
    actionCode: BoundedIdentifier
    bundleVersion: BoundedIdentifier
    bundleDigest: str = Field(pattern=_DIGEST_PATTERN)
    parameters: ParameterObject
    scenario: Scenario
    leaseMs: int | None = Field(default=None, gt=0, le=60_000)
    requestedBy: Annotated[
        str, StringConstraints(strict=True, min_length=1, max_length=256)
    ]

    @property
    def schema(self) -> str:
        return self.schema_

    @field_validator("parameters", mode="before")
    @classmethod
    def reject_raw_control_intent(cls, value: Any) -> Any:
        _reject_raw_control_keys(value)
        return value


class TaskCommandRequest(ContractModel):
    commandSequence: int = Field(ge=0, le=9_223_372_036_854_775_807)
    parameters: ParameterObject
    leaseMs: int = Field(gt=0, le=60_000)

    @field_validator("parameters", mode="before")
    @classmethod
    def reject_raw_control_intent(cls, value: Any) -> Any:
        _reject_raw_control_keys(value)
        return value


class TaskEvidence(ContractModel):
    bundleDigest: str = Field(pattern=_DIGEST_PATTERN)
    policyDigest: str = Field(pattern=_DIGEST_PATTERN)
    modelDigest: str | None = Field(default=None, pattern=_DIGEST_PATTERN)
    metrics: EvidenceMetrics = Field(default_factory=dict)
    stopReason: BoundedIdentifier | None = None


class TaskEvent(ContractModel):
    sequence: int = Field(ge=0, le=9_223_372_036_854_775_807)
    eventType: BoundedIdentifier
    payload: EventPayload = Field(default_factory=dict)
    createdAt: datetime


class TaskSnapshot(ContractModel):
    taskId: str = Field(pattern=_TASK_ID_PATTERN)
    state: Literal[
        "ACCEPTED",
        "VALIDATING",
        "RUNNING",
        "SUCCEEDED",
        "FAILED",
        "CANCELLED",
        "TIMED_OUT",
        "UNKNOWN",
    ]
    actionCode: BoundedIdentifier
    bundleVersion: BoundedIdentifier
    bundleDigest: str = Field(pattern=_DIGEST_PATTERN)
    requestedAt: datetime
    updatedAt: datetime
    evidence: TaskEvidence | None = None
    stopReason: BoundedIdentifier | None = None


class RobotStatus(ContractModel):
    schema_: Literal["BIPED_POSE_V1"] = Field(
        ..., alias="schema", serialization_alias="schema"
    )
    timestamp: datetime
    basePositionM: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    baseOrientationXyzw: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
    baseLinearVelocityMps: tuple[FiniteFloat, FiniteFloat, FiniteFloat] = Field(
        description="World-frame trunk-base linear velocity."
    )
    baseAngularVelocityRadps: tuple[FiniteFloat, FiniteFloat, FiniteFloat] = Field(
        description="Trunk-body-frame angular velocity, matching the training IMU gyro."
    )
    jointPositionsRad: tuple[
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
    ]
    jointVelocitiesRadps: tuple[
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
    ]
    policyTarget: StatusObject
    requestedMotion: StatusObject
    appliedMotion: StatusObject
    limitingReason: BoundedIdentifier | None = None
    activePolicyRef: BoundedIdentifier | None = None
    activeActionCode: BoundedIdentifier | None = None
    activeTaskId: str | None = Field(default=None, pattern=_TASK_ID_PATTERN)
    simulationTimeS: FiniteFloat = Field(ge=0)
    loopFrequencyHz: FiniteFloat = Field(ge=0)
    fallen: bool
    limp: bool
    health: StatusObject

    @property
    def schema(self) -> str:
        return self.schema_


def _canonical_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite floats are not valid canonical JSON")
    if isinstance(value, BaseModel):
        return _canonical_value(
            value.model_dump(mode="json", by_alias=True, exclude_none=True)
        )
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical JSON object keys must be strings")
        return {
            key: _canonical_value(nested_value) for key, nested_value in value.items()
        }
    if isinstance(value, tuple | list):
        return [_canonical_value(nested_value) for nested_value in value]
    return value


def canonical_json(value: BaseModel | Mapping[str, Any]) -> bytes:
    """Return UTF-8 JSON with a stable key order and no insignificant whitespace."""
    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()


def sha256_prefixed(value: BaseModel | Mapping[str, Any]) -> str:
    """Hash canonical JSON using the manifest/API ``sha256:<hex>`` wire form."""
    return f"sha256:{hashlib.sha256(canonical_json(value)).hexdigest()}"
