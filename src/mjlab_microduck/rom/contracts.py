"""Stable wire contracts shared by MicroDuck policy bundles and the ROM API."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class ContractModel(BaseModel):
    """Base for strict, camel-case wire records."""

    model_config = ConfigDict(
        extra="forbid", validate_by_alias=True, validate_by_name=True, serialize_by_alias=True
    )


class CompletionContract(ContractModel):
    terminalConditions: list[str] = Field(min_length=1)
    maxDurationMs: int = Field(gt=0)


class LeaseContract(ContractModel):
    minLeaseMs: int = Field(gt=0)
    defaultLeaseMs: int = Field(gt=0)
    maxLeaseMs: int = Field(gt=0)
    commandCadenceMs: int = Field(gt=0)
    safeStopBehavior: str | None = None

    @model_validator(mode="after")
    def validate_lease_bounds(self) -> LeaseContract:
        if not self.minLeaseMs <= self.defaultLeaseMs <= self.maxLeaseMs:
            raise ValueError("lease bounds must satisfy minLeaseMs <= defaultLeaseMs <= maxLeaseMs")
        if self.commandCadenceMs > self.minLeaseMs:
            raise ValueError("commandCadenceMs must not exceed minLeaseMs")
        return self


class ObservationContract(ContractModel):
    identifier: Literal["MICRODUCK_OBS_61_V1"]
    dimension: Literal[61]
    fields: list[str] = Field(min_length=1)
    units: dict[str, str]
    normalization: Literal["BAKED_IN_ONNX", "DECLARED"]

    @model_validator(mode="after")
    def validate_shared_layout(self) -> ObservationContract:
        if self.fields != list(OBSERVATION_FIELDS):
            raise ValueError("observation fields must use the exact shared 61D layout")
        return self


class ActionContract(ContractModel):
    identifier: Literal["MICRODUCK_ACTION_14_V1"]
    dimension: Literal[14]
    joints: list[str] = Field(min_length=14, max_length=14)
    units: str
    scaling: dict[str, Any]
    clipping: dict[str, Any]

    @model_validator(mode="after")
    def validate_controlled_servo_order(self) -> ActionContract:
        if self.joints != list(CONTROLLED_SERVO_JOINTS):
            raise ValueError("action joints must use the exact controlled-servo order")
        return self


class PolicyArtifact(ContractModel):
    policyRef: str = Field(min_length=1)
    path: str = Field(min_length=1)
    digest: str = Field(pattern=_DIGEST_PATTERN)
    taskId: str | None = None
    runtimeRequirements: dict[str, Any] = Field(default_factory=dict)
    checkpoint: str | None = None
    experimentRef: str | None = None


class ModelArtifact(ContractModel):
    path: str = Field(min_length=1)
    digest: str = Field(pattern=_DIGEST_PATTERN)


class ActionDefinition(ContractModel):
    actionCode: str = Field(min_length=1)
    executionMode: Literal["DISCRETE", "CONTINUOUS_LEASE"]
    availability: Literal["AVAILABLE", "UNAVAILABLE"]
    policyRef: str | None = None
    unavailableReason: str | None = None
    parameterSchema: dict[str, Any]
    completion: CompletionContract | None = None
    lease: LeaseContract | None = None
    displayName: str | None = None
    description: str | None = None
    localizedLabels: dict[str, str] | None = None
    preconditions: dict[str, Any] | None = None
    safety: dict[str, Any] | None = None
    qualificationRefs: list[str] | None = None

    @model_validator(mode="after")
    def validate_mode_and_artifact(self) -> ActionDefinition:
        if self.availability == "AVAILABLE" and not self.policyRef:
            raise ValueError("available action requires policyRef")
        if self.executionMode == "CONTINUOUS_LEASE" and self.lease is None:
            raise ValueError("continuous action requires lease contract")
        return self


class PolicyBundle(ContractModel):
    schema_: Literal["MICRODUCK_POLICY_BUNDLE_V1"] = Field(
        ..., alias="schema", serialization_alias="schema"
    )
    bundleId: str = Field(min_length=1)
    bundleVersion: str = Field(min_length=1)
    bundleDigest: str | None = Field(default=None, pattern=_DIGEST_PATTERN)
    createdAt: datetime
    sourceRepository: str = Field(min_length=1)
    sourceCommit: str = Field(min_length=1)
    robotModel: Literal["MICRODUCK"]
    observationContract: ObservationContract
    actionContract: ActionContract
    model: ModelArtifact
    policies: list[PolicyArtifact]
    actions: list[ActionDefinition]
    qualification: dict[str, Any]
    license: dict[str, Any]

    @property
    def schema(self) -> str:
        return self.schema_

    @model_validator(mode="after")
    def validate_unique_references(self) -> PolicyBundle:
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
        return self


def _reject_raw_control_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            if isinstance(key, str) and _RAW_CONTROL_KEY.search(key):
                raise ValueError(f"raw control key is not permitted: {key}")
            _reject_raw_control_keys(nested_value)
    elif isinstance(value, list):
        for nested_value in value:
            _reject_raw_control_keys(nested_value)


class TaskCreateRequest(ContractModel):
    schema_: Literal["MICRODUCK_SIM_TASK_V1"] = Field(
        ..., alias="schema", serialization_alias="schema"
    )
    taskId: str = Field(pattern=_TASK_ID_PATTERN)
    actionCode: str = Field(min_length=1)
    bundleVersion: str = Field(min_length=1)
    bundleDigest: str = Field(pattern=_DIGEST_PATTERN)
    parameters: dict[str, Any]
    scenario: dict[str, Any]
    leaseMs: int | None = Field(default=None, gt=0)
    requestedBy: str = Field(min_length=1)

    @property
    def schema(self) -> str:
        return self.schema_

    @model_validator(mode="after")
    def reject_raw_control_intent(self) -> TaskCreateRequest:
        _reject_raw_control_keys(self.parameters)
        return self


class TaskCommandRequest(ContractModel):
    commandSequence: int = Field(ge=0)
    parameters: dict[str, Any]
    leaseMs: int = Field(gt=0)

    @model_validator(mode="after")
    def reject_raw_control_intent(self) -> TaskCommandRequest:
        _reject_raw_control_keys(self.parameters)
        return self


class TaskEvidence(ContractModel):
    bundleDigest: str = Field(pattern=_DIGEST_PATTERN)
    policyDigest: str = Field(pattern=_DIGEST_PATTERN)
    modelDigest: str | None = Field(default=None, pattern=_DIGEST_PATTERN)
    metrics: dict[str, Any] = Field(default_factory=dict)
    stopReason: str | None = None


class TaskEvent(ContractModel):
    sequence: int = Field(ge=0)
    eventType: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
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
    actionCode: str = Field(min_length=1)
    bundleVersion: str = Field(min_length=1)
    bundleDigest: str = Field(pattern=_DIGEST_PATTERN)
    requestedAt: datetime
    updatedAt: datetime
    evidence: TaskEvidence | None = None
    stopReason: str | None = None


class RobotStatus(ContractModel):
    schema_: Literal["BIPED_POSE_V1"] = Field(
        ..., alias="schema", serialization_alias="schema"
    )
    timestamp: datetime
    basePositionM: tuple[float, float, float]
    baseOrientationXyzw: tuple[float, float, float, float]
    baseLinearVelocityMps: tuple[float, float, float]
    baseAngularVelocityRadps: tuple[float, float, float]
    jointPositionsRad: tuple[float, float, float, float, float, float, float, float, float, float, float, float, float, float]
    jointVelocitiesRadps: tuple[float, float, float, float, float, float, float, float, float, float, float, float, float, float]
    policyTarget: dict[str, Any]
    requestedMotion: dict[str, Any]
    appliedMotion: dict[str, Any]
    limitingReason: str | None = None
    activePolicyRef: str | None = None
    activeActionCode: str | None = None
    activeTaskId: str | None = Field(default=None, pattern=_TASK_ID_PATTERN)
    simulationTimeS: float = Field(ge=0)
    loopFrequencyHz: float = Field(ge=0)
    fallen: bool
    limp: bool
    health: dict[str, Any]

    @property
    def schema(self) -> str:
        return self.schema_


def _canonical_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite floats are not valid canonical JSON")
    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="json", by_alias=True, exclude_none=True))
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(nested_value) for key, nested_value in value.items()}
    if isinstance(value, tuple | list):
        return [_canonical_value(nested_value) for nested_value in value]
    return value


def canonical_json(value: BaseModel | Mapping[str, Any]) -> bytes:
    """Return UTF-8 JSON with a stable key order and no insignificant whitespace."""
    return json.dumps(
        _canonical_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def sha256_prefixed(value: BaseModel | Mapping[str, Any]) -> str:
    """Hash canonical JSON using the manifest/API ``sha256:<hex>`` wire form."""
    return f"sha256:{hashlib.sha256(canonical_json(value)).hexdigest()}"
