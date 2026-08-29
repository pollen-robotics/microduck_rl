from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mjlab_microduck.rom.contracts import (
    ActionContract,
    ActionDefinition,
    CompletionContract,
    LeaseContract,
    ObservationContract,
    PolicyBundle,
    RobotStatus,
    Scenario,
    TaskCommandRequest,
    TaskCreateRequest,
    TaskEvent,
    TaskEvidence,
    UnsignedPolicyBundleManifest,
    canonical_json,
    sha256_prefixed,
    unsigned_policy_bundle_manifest,
)

CONTROLLED_JOINTS = (
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
    *(f"joint_pos_rel.{joint}" for joint in CONTROLLED_JOINTS),
    *(f"joint_vel_rel.{joint}" for joint in CONTROLLED_JOINTS),
    *(f"last_action.{joint}" for joint in CONTROLLED_JOINTS),
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


def valid_observation_contract() -> dict[str, object]:
    return {
        "identifier": "MICRODUCK_OBS_61_V1",
        "dimension": 61,
        "fields": list(OBSERVATION_FIELDS),
        "units": {},
        "normalization": "BAKED_IN_ONNX",
    }


def valid_action_contract() -> dict[str, object]:
    return {
        "identifier": "MICRODUCK_ACTION_14_V1",
        "dimension": 14,
        "joints": list(CONTROLLED_JOINTS),
        "units": "rad",
        "scaling": {},
        "clipping": {},
    }


def valid_bundle() -> dict[str, object]:
    return {
        "schema": "MICRODUCK_POLICY_BUNDLE_V1",
        "bundleId": "org.microduck.test",
        "bundleVersion": "1.0.0",
        "bundleDigest": "sha256:" + "a" * 64,
        "createdAt": "2026-08-29T00:00:00Z",
        "sourceRepository": "microduck_rl",
        "sourceCommit": "a" * 40,
        "robotModel": "MICRODUCK",
        "observationContract": valid_observation_contract(),
        "actionContract": valid_action_contract(),
        "model": {"path": "models/robot.xml", "digest": "sha256:" + "b" * 64},
        "policies": [],
        "actions": [],
        "qualification": {},
        "license": {},
    }


def valid_robot_status() -> dict[str, object]:
    return {
        "schema": "BIPED_POSE_V1",
        "timestamp": "2026-08-29T00:00:00Z",
        "basePositionM": [0.0, 0.0, 0.0],
        "baseOrientationXyzw": [0.0, 0.0, 0.0, 1.0],
        "baseLinearVelocityMps": [0.0, 0.0, 0.0],
        "baseAngularVelocityRadps": [0.0, 0.0, 0.0],
        "jointPositionsRad": [0.0] * 14,
        "jointVelocitiesRadps": [0.0] * 14,
        "policyTarget": {},
        "requestedMotion": {},
        "appliedMotion": {},
        "simulationTimeS": 0.0,
        "loopFrequencyHz": 50.0,
        "fallen": False,
        "limp": False,
        "health": {},
    }


def test_available_action_requires_policy_artifact():
    """Removing the availability/policy binding must reject an unsafe catalog entry."""
    with pytest.raises(ValidationError, match="available action requires policyRef"):
        ActionDefinition(
            actionCode="WALK_VELOCITY",
            executionMode="CONTINUOUS_LEASE",
            availability="AVAILABLE",
            parameterSchema={},
            policyRef=None,
        )


def test_continuous_action_requires_lease_contract():
    """Removing the target-side deadman contract must reject continuous motion."""
    with pytest.raises(
        ValidationError, match="continuous action requires lease contract"
    ):
        ActionDefinition(
            actionCode="WALK_VELOCITY",
            executionMode="CONTINUOUS_LEASE",
            availability="AVAILABLE",
            parameterSchema={},
            policyRef="walk-policy",
        )


def test_task_rejects_raw_joint_intent():
    """Removing recursive raw-control validation would expose servo targets to ROM."""
    with pytest.raises(ValidationError, match="raw control key is not permitted"):
        TaskCreateRequest.model_validate(
            {
                "schema": "MICRODUCK_SIM_TASK_V1",
                "taskId": "0" * 32,
                "actionCode": "STAND",
                "bundleVersion": "1.0.0",
                "bundleDigest": "sha256:" + "a" * 64,
                "parameters": {"jointTargets": [0] * 14},
                "scenario": {"terrain": "flat", "seed": 1},
                "requestedBy": "execution-1",
            }
        )


@pytest.mark.parametrize("raw_key", ["torque", "PWM", "policyPath", "policyName"])
def test_task_rejects_raw_control_keys_at_every_parameter_depth(raw_key: str):
    """Removing nested inspection would permit a raw control key below a typed envelope."""
    with pytest.raises(ValidationError, match="raw control key is not permitted"):
        TaskCreateRequest.model_validate(
            {
                "schema": "MICRODUCK_SIM_TASK_V1",
                "taskId": "1" * 32,
                "actionCode": "STAND",
                "bundleVersion": "1.0.0",
                "bundleDigest": "sha256:" + "b" * 64,
                "parameters": {"nested": [{raw_key: 0}]},
                "scenario": {"terrain": "flat", "seed": 1},
                "requestedBy": "execution-1",
            }
        )


def test_task_id_and_digest_are_lowercase_fixed_width_values():
    """Weakening identity patterns would make canonical task and bundle binding ambiguous."""
    payload = {
        "schema": "MICRODUCK_SIM_TASK_V1",
        "taskId": "A" * 32,
        "actionCode": "STAND",
        "bundleVersion": "1.0.0",
        "bundleDigest": "sha256:" + "A" * 64,
        "parameters": {},
        "scenario": {"terrain": "flat", "seed": 1},
        "requestedBy": "execution-1",
    }

    with pytest.raises(ValidationError):
        TaskCreateRequest.model_validate(payload)


def test_canonical_digest_ignores_mapping_insertion_order():
    """Dropping canonical key sorting would make a task/bundle hash depend on construction order."""
    assert sha256_prefixed({"b": 2, "a": 1}) == sha256_prefixed({"a": 1, "b": 2})
    assert canonical_json({"b": 2, "a": 1}) == b'{"a":1,"b":2}'


def test_canonical_json_normalizes_nested_models():
    """Failing to normalize embedded models would make manifest digest construction non-serializable."""
    lease = LeaseContract(
        minLeaseMs=100,
        defaultLeaseMs=200,
        maxLeaseMs=500,
        commandCadenceMs=50,
        zeroCommand={"vxMps": 0.0},
        safeStopBehavior="ZERO_TWIST",
    )
    assert canonical_json({"lease": lease}) == (
        b'{"lease":{"commandCadenceMs":50,"defaultLeaseMs":200,"maxLeaseMs":500,'
        b'"minLeaseMs":100,"safeStopBehavior":"ZERO_TWIST","zeroCommand":{"vxMps":0.0}}}'
    )


def test_observation_contract_rejects_any_order_other_than_the_shared_61d_layout():
    """A swapped observation coordinate would feed every hot-swappable policy the wrong value."""
    fields = list(OBSERVATION_FIELDS)
    fields[0], fields[1] = fields[1], fields[0]

    with pytest.raises(ValidationError, match="exact shared 61D layout"):
        ObservationContract(**(valid_observation_contract() | {"fields": fields}))


def test_action_contract_rejects_passive_or_reordered_servo_joints():
    """A passive/reordered action coordinate would send ONNX output to the wrong actuator."""
    joints = list(CONTROLLED_JOINTS)
    joints[4] = "passive_left_wheel"

    with pytest.raises(ValidationError, match="exact controlled-servo order"):
        ActionContract(**(valid_action_contract() | {"joints": joints}))


@pytest.mark.parametrize("non_finite", [math.nan, math.inf, -math.inf])
def test_canonical_json_rejects_non_finite_values_at_any_depth(non_finite: float):
    """Allowing a non-finite number would produce non-standard JSON and unstable digests."""
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json({"nested": [0.0, {"value": non_finite}]})


def test_published_v1_models_require_explicit_schema_identifiers():
    """Omitting a schema identifier would permit ambiguous versionless wire messages."""
    task = {
        "taskId": "0" * 32,
        "actionCode": "STAND",
        "bundleVersion": "1.0.0",
        "bundleDigest": "sha256:" + "a" * 64,
        "parameters": {},
        "scenario": {"terrain": "flat", "seed": 1},
        "requestedBy": "execution-1",
    }
    bundle = valid_bundle()
    status = valid_robot_status()
    bundle.pop("schema")
    status.pop("schema")

    for model, payload in (
        (TaskCreateRequest, task),
        (PolicyBundle, bundle),
        (RobotStatus, status),
    ):
        with pytest.raises(ValidationError):
            model.model_validate(payload)


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (CompletionContract, {"terminalConditions": ["DONE"], "maxDurationMs": "1"}),
        (
            TaskCommandRequest,
            {"commandSequence": 1.0, "parameters": {}, "leaseMs": 100},
        ),
        (
            TaskCommandRequest,
            {"commandSequence": True, "parameters": {}, "leaseMs": 100},
        ),
        (
            RobotStatus,
            valid_robot_status() | {"loopFrequencyHz": "50.0"},
        ),
    ],
)
def test_wire_models_reject_numeric_and_string_coercion(model, payload):
    """Weakening strict validation would let distinct JSON types share one intent hash."""
    with pytest.raises(ValidationError):
        model.model_validate(payload)


@pytest.mark.parametrize(
    "timestamp",
    [0, "0", "2026-08-29", "2026-08-29 00:00:00Z", "2026-08-29T00:00:00"],
)
def test_wire_datetimes_accept_only_rfc3339_strings_with_timezone(timestamp):
    """Permissive datetime coercion would make event timestamps non-portable across clients."""
    payload = {
        "sequence": 0,
        "eventType": "TASK_VALIDATING",
        "payload": {},
        "createdAt": timestamp,
    }

    with pytest.raises(ValidationError):
        TaskEvent.model_validate(payload)

    accepted = TaskEvent.model_validate(payload | {"createdAt": "2026-08-29T00:00:00Z"})
    assert accepted.createdAt.isoformat() == "2026-08-29T00:00:00+00:00"


def test_published_bundle_requires_digest_and_unsigned_hashing_is_explicit():
    """Making the published digest nullable would let an unsigned manifest cross the trust boundary."""
    payload = valid_bundle()
    payload.pop("bundleDigest")
    with pytest.raises(ValidationError):
        PolicyBundle.model_validate(payload)
    with pytest.raises(ValidationError):
        PolicyBundle.model_validate(payload | {"bundleDigest": None})

    unsigned = UnsignedPolicyBundleManifest.model_validate(payload)
    published = PolicyBundle.model_validate(
        payload | {"bundleDigest": "sha256:" + "a" * 64}
    )
    assert unsigned_policy_bundle_manifest(published) == unsigned
    assert "bundleDigest" not in unsigned.model_dump(by_alias=True)


@pytest.mark.parametrize(
    "scenario",
    [
        {"terrain": "stairs", "seed": 1},
        {"terrain": "flat", "seed": -1},
        {"terrain": "flat", "seed": 2**32},
        {"terrain": "flat", "seed": "1"},
        {"terrain": "flat", "seed": 1, "friction": 0.1},
    ],
)
def test_scenario_is_the_exact_seeded_v1_runtime_contract(scenario):
    """A free-form scenario would let callers mutate reset/runtime semantics outside V1."""
    with pytest.raises(ValidationError):
        Scenario.model_validate(scenario)

    assert Scenario.model_validate({"terrain": "ramp", "seed": 2**32 - 1}) == Scenario(
        terrain="ramp", seed=2**32 - 1
    )


def test_task_parameters_evidence_and_event_payloads_are_flat_and_bounded():
    """Unbounded nested request/evidence maps would permit persistence and response amplification."""
    task = {
        "schema": "MICRODUCK_SIM_TASK_V1",
        "taskId": "0" * 32,
        "actionCode": "STAND",
        "bundleVersion": "1.0.0",
        "bundleDigest": "sha256:" + "a" * 64,
        "parameters": {},
        "scenario": {"terrain": "flat", "seed": 1},
        "requestedBy": "execution-1",
    }
    with pytest.raises(ValidationError):
        TaskCreateRequest.model_validate(task | {"parameters": {"nested": {"x": 1}}})
    with pytest.raises(ValidationError):
        TaskCreateRequest.model_validate(
            task | {"parameters": {f"p{index}": index for index in range(17)}}
        )
    with pytest.raises(ValidationError):
        TaskEvidence(
            bundleDigest="sha256:" + "a" * 64,
            policyDigest="sha256:" + "b" * 64,
            metrics={f"metric{index}": index for index in range(33)},
        )
    with pytest.raises(ValidationError):
        TaskEvent.model_validate(
            {
                "sequence": 0,
                "eventType": "TASK_EVENT",
                "payload": {f"field{index}": index for index in range(17)},
                "createdAt": "2026-08-29T00:00:00Z",
            }
        )
    with pytest.raises(ValidationError, match="encoded wire limit"):
        TaskEvent.model_validate(
            {
                "sequence": 0,
                "eventType": "TASK_EVENT",
                "payload": {f"field{index}": "x" * 1_024 for index in range(4)},
                "createdAt": "2026-08-29T00:00:00Z",
            }
        )
    with pytest.raises(ValidationError, match="encoded wire limit"):
        TaskEvidence(
            bundleDigest="sha256:" + "a" * 64,
            policyDigest="sha256:" + "b" * 64,
            metrics={"note": "x" * 1_024},
        )


def test_wire_float_and_parameter_integer_types_are_exact_and_bounded():
    """Float coercion and arbitrary-size JSON integers are not portable wire values."""
    with pytest.raises(ValidationError, match="wire float"):
        RobotStatus.model_validate(valid_robot_status() | {"loopFrequencyHz": 50})
    task = {
        "schema": "MICRODUCK_SIM_TASK_V1",
        "taskId": "0" * 32,
        "actionCode": "STAND",
        "bundleVersion": "1.0.0",
        "bundleDigest": "sha256:" + "a" * 64,
        "parameters": {"count": 2**63},
        "scenario": {"terrain": "flat", "seed": 1},
        "requestedBy": "execution-1",
    }
    with pytest.raises(ValidationError):
        TaskCreateRequest.model_validate(task)


def test_shared_portability_fixtures_match_python_semantic_validation():
    """Python and ROM consumers must share accept/reject examples for non-schema invariants."""
    fixture_path = (
        Path(__file__).parents[1] / "schemas/microduck-v1-portability-fixtures.json"
    )
    fixture = json.loads(fixture_path.read_text())
    models = {
        "PolicyBundle": PolicyBundle,
        "RobotStatus": RobotStatus,
        "TaskCommandRequest": TaskCommandRequest,
        "TaskCreateRequest": TaskCreateRequest,
        "TaskEvent": TaskEvent,
    }
    covered = {case["covers"] for case in fixture["cases"]}
    assert {
        "bytes",
        "coercion",
        "date-time",
        "depth",
        "digest",
        "raw-control",
        "scenario",
        "sequence",
    } <= covered

    for case in fixture["cases"]:
        payload = deepcopy(fixture["basePayloads"][case["base"]])
        for pointer in case.get("remove", []):
            _remove_json_pointer(payload, pointer)
        for pointer, value in case.get("set", {}).items():
            _set_json_pointer(payload, pointer, value)
        model = models[case["model"]]
        try:
            model.model_validate(payload)
        except ValidationError:
            accepted = False
        else:
            accepted = True
        assert accepted is case["accepted"], case["name"]


def _set_json_pointer(document, pointer: str, value) -> None:
    parts = [
        part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")
    ]
    target = document
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    final = parts[-1]
    if isinstance(target, list):
        target[int(final)] = value
    else:
        target[final] = value


def _remove_json_pointer(document, pointer: str) -> None:
    parts = [
        part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")
    ]
    target = document
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    final = parts[-1]
    if isinstance(target, list):
        del target[int(final)]
    else:
        del target[final]


def test_lease_contract_rejects_invalid_semantic_bounds():
    """Ignoring lease ordering would let a manifest declare an impossible deadman interval."""
    with pytest.raises(ValidationError, match="lease bounds"):
        LeaseContract(
            minLeaseMs=200,
            defaultLeaseMs=100,
            maxLeaseMs=500,
            commandCadenceMs=50,
            zeroCommand={"vxMps": 0.0},
            safeStopBehavior="ZERO_TWIST",
        )
    with pytest.raises(ValidationError, match="commandCadenceMs"):
        LeaseContract(
            minLeaseMs=100,
            defaultLeaseMs=200,
            maxLeaseMs=500,
            commandCadenceMs=101,
            zeroCommand={"vxMps": 0.0},
            safeStopBehavior="ZERO_TWIST",
        )


def test_checked_in_schemas_lock_layouts_error_codes_and_portable_lease_invariants():
    """A relaxed checked-in schema would let non-Python consumers bypass the wire contract."""
    repository = Path(__file__).parents[1]
    bundle_schema = json.loads(
        (repository / "schemas/microduck-policy-bundle-v1.schema.json").read_text()
    )
    openapi = yaml.safe_load(
        (repository / "schemas/microduck-simulator-api-v1.openapi.yaml").read_text()
    )

    observation_fields = bundle_schema["$defs"]["ObservationContract"]["properties"][
        "fields"
    ]
    action_joints = bundle_schema["$defs"]["ActionContract"]["properties"]["joints"]
    assert "bundleDigest" in bundle_schema["required"]
    assert [entry["const"] for entry in observation_fields["prefixItems"]] == list(
        OBSERVATION_FIELDS
    )
    assert observation_fields["minItems"] == observation_fields["maxItems"] == 61
    assert [entry["const"] for entry in action_joints["prefixItems"]] == list(
        CONTROLLED_JOINTS
    )
    assert action_joints["minItems"] == action_joints["maxItems"] == 14
    lease_schema = bundle_schema["$defs"]["LeaseContract"]
    assert lease_schema["x-unergy-invariants"] == [
        "minLeaseMs <= defaultLeaseMs <= maxLeaseMs",
        "commandCadenceMs <= minLeaseMs",
    ]
    assert {"zeroCommand", "safeStopBehavior"} <= set(lease_schema["required"])
    assert lease_schema["properties"]["safeStopBehavior"]["const"] == "ZERO_TWIST"
    openapi_lease = openapi["components"]["schemas"]["LeaseContract"]
    assert {"zeroCommand", "safeStopBehavior"} <= set(openapi_lease["required"])
    assert openapi_lease["properties"]["safeStopBehavior"]["const"] == "ZERO_TWIST"
    assert bundle_schema["$defs"]["ActionDefinition"]["allOf"] == [
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
    assert (
        "TASK_NOT_FOUND"
        in openapi["components"]["schemas"]["Error"]["properties"]["code"]["enum"]
    )
    assert (
        "COMMAND_SEQUENCE_CONFLICT"
        in openapi["components"]["schemas"]["Error"]["properties"]["code"]["enum"]
    )
    assert "schema" in bundle_schema["required"]
    assert "schema" in openapi["components"]["schemas"]["TaskCreateRequest"]["required"]
    assert "schema" in openapi["components"]["schemas"]["RobotStatus"]["required"]
    assert bundle_schema["x-unergy-invariants"] == {
        "finiteNumbers": True,
        "maxCanonicalJsonBytes": 65_536,
        "maxDepth": 8,
        "semanticValidatorRequired": True,
    }
    parameters = openapi["components"]["schemas"]["TaskCreateRequest"]["properties"][
        "parameters"
    ]
    assert parameters["x-unergy-invariants"] == {
        "finiteScalarsOnly": True,
        "maxCanonicalJsonBytes": 16_384,
        "maxDepth": 1,
        "rejectPropertyNameSubstringsCaseInsensitive": [
            "joint",
            "torque",
            "pwm",
            "policyPath",
            "policyName",
        ],
    }
    assert parameters["propertyNames"]["not"]["pattern"]
    assert (
        openapi["components"]["schemas"]["TaskEvent"]["properties"]["payload"][
            "x-unergy-invariants"
        ]["maxCanonicalJsonBytes"]
        == 4_096
    )
    assert (
        openapi["components"]["schemas"]["TaskEvidence"]["properties"]["metrics"][
            "x-unergy-invariants"
        ]["maxCanonicalJsonBytes"]
        == 1_024
    )


def test_tracked_bundle_schema_matches_the_complete_generated_contract():
    """Partial hand-maintained parity would leave bounds or required fields inconsistent."""
    tracked = json.loads(
        (
            Path(__file__).parents[1] / "schemas/microduck-policy-bundle-v1.schema.json"
        ).read_text()
    )
    tracked.pop("$schema")
    tracked.pop("$id")
    generated = PolicyBundle.model_json_schema(by_alias=True)

    assert _normalize_json_schema(tracked) == _normalize_json_schema(generated)


def _normalize_json_schema(value):
    if isinstance(value, list):
        return [_normalize_json_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: _normalize_json_schema(item)
        for key, item in value.items()
        if key != "title" and not (key == "additionalProperties" and item is True)
    }
