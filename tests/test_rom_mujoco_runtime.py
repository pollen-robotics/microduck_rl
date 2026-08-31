from __future__ import annotations

import hashlib
import math
import threading
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import mujoco
import numpy as np
import onnx
import pytest
from fastapi.testclient import TestClient
from onnx import TensorProto, helper
from pydantic import ValidationError

from mjlab_microduck.rom.action_catalog import (
    CODE_OWNED_ACTION_CODES,
    code_owned_action_definition,
)
from mjlab_microduck.rom.action_specs import ACTION_RUNTIME_SPECS
from mjlab_microduck.rom.contracts import (
    ACTION_CONTRACT,
    CONTROLLED_SERVO_JOINTS,
    OBSERVATION_CONTRACT,
    ActionContract,
    ModelArtifact,
    ObservationContract,
    PolicyArtifact,
    PolicyBundle,
    Scenario,
    TaskCreateRequest,
    UnsignedPolicyBundleManifest,
    publish_policy_bundle,
    sha256_prefixed,
    unsigned_policy_bundle_manifest,
)
from mjlab_microduck.rom.main import create_configured_app, load_verified_bundle
from mjlab_microduck.rom.mujoco_runtime import MicroduckMujocoRuntime
from mjlab_microduck.rom.observation import DEFAULT_JOINT_POSE
from mjlab_microduck.rom.onnx_policy import inspect_normalized_actor
from mjlab_microduck.rom.qualification import (
    ActionQualificationConfig,
    QualificationThresholds,
    ReleaseConfiguration,
    qualify_and_promote,
)
from mjlab_microduck.rom.runtime import RuntimeHandle, canonical_tracking_mean
from mjlab_microduck.rom.service import SimulatorTaskService
from mjlab_microduck.rom.store import SqliteTaskStore

SOURCE_COMMIT = "a" * 40
TASK_ID = "Mjlab-Velocity-Flat-MicroDuck"

_REPLACED_DIRECT_SERVICE_TESTS = {
    "test_realtime_stop_during_blocked_start_leaves_no_runtime_owner_or_control",
    "test_realtime_emergency_after_final_start_check_revokes_publication",
    "test_realtime_stop_after_runtime_start_return_uses_retained_cleanup_handle",
    "test_service_tick_observes_concrete_runtime_fault_and_zeros_applied_motion",
}


@pytest.fixture(autouse=True)
def _process_replaces_direct_runtime_service_tests(request):
    if request.node.name.split("[", 1)[0] in _REPLACED_DIRECT_SERVICE_TESTS:
        pytest.skip(
            "implementation-shape-only direct-runtime service test; exact child-process "
            "replacement is mapped in test_rom_service_process_integration.py"
        )


def _digest(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _write_policy(
    path: Path,
    *,
    output: np.ndarray | None = None,
    input_dimension: int = 61,
    tensor_type: int = TensorProto.FLOAT,
    normalizer_tensor_type: int | None = None,
    observation_dependent: bool = False,
    metadata_overrides: dict[str, str] | None = None,
    normalizer_mean_values: np.ndarray | None = None,
    normalizer_std_values: np.ndarray | None = None,
    identity_before_normalizer: bool = False,
    bypass_normalizer: bool = False,
    second_normalizer: bool = False,
    normalizer_constant_transform: str | None = None,
    second_normalizer_constant_transform: str | None = None,
    actor_post_normalization: bool = False,
    task_id: str = TASK_ID,
) -> None:
    output_values = (
        np.linspace(-0.13, 0.13, 14, dtype=np.float32)
        if output is None
        else np.asarray(output, dtype=np.float32)
    )
    observations = helper.make_tensor_value_info(
        "observations", tensor_type, [1, input_dimension]
    )
    actions = helper.make_tensor_value_info("actions", tensor_type, [1, 14])
    normalizer_mean = helper.make_tensor(
        "normalizer_mean",
        normalizer_tensor_type or tensor_type,
        [input_dimension],
        (
            np.zeros(input_dimension, dtype=np.float32)
            if normalizer_mean_values is None
            else np.asarray(normalizer_mean_values)
        ).ravel(),
    )
    normalizer_std = helper.make_tensor(
        "normalizer_std",
        normalizer_tensor_type or tensor_type,
        [input_dimension],
        (
            np.ones(input_dimension, dtype=np.float32)
            if normalizer_std_values is None
            else np.asarray(normalizer_std_values)
        ).ravel(),
    )
    weight_values = np.zeros((input_dimension, 14), dtype=np.float32)
    if observation_dependent and input_dimension == 61:
        weight_values[48, 0] = 0.5
    weights = helper.make_tensor(
        "weights",
        TensorProto.FLOAT,
        [input_dimension, 14],
        weight_values.ravel(),
    )
    bias = helper.make_tensor("bias", TensorProto.FLOAT, [14], output_values.ravel())
    second_mean = helper.make_tensor(
        "second_mean", TensorProto.FLOAT, [61], [0.25] * 61
    )
    second_std = helper.make_tensor("second_std", TensorProto.FLOAT, [61], [2.0] * 61)
    actor_mean = helper.make_tensor("actor_mean", TensorProto.FLOAT, [14], [0.25] * 14)
    actor_std = helper.make_tensor("actor_std", TensorProto.FLOAT, [14], [2.0] * 14)
    constant_nodes: list[onnx.NodeProto] = []
    constant_initializers: list[onnx.TensorProto] = []

    def transformed_constant(name: str, transform: str | None) -> str:
        if transform is None:
            return name
        output_name = f"{name}_{transform.lower()}"
        if transform == "Identity":
            constant_nodes.append(helper.make_node("Identity", [name], [output_name]))
        elif transform == "Cast":
            constant_nodes.append(
                helper.make_node("Cast", [name], [output_name], to=TensorProto.FLOAT)
            )
        elif transform == "Reshape":
            shape_name = f"{name}_shape"
            constant_initializers.append(
                helper.make_tensor(shape_name, TensorProto.INT64, [1], [61])
            )
            constant_nodes.append(
                helper.make_node("Reshape", [name, shape_name], [output_name])
            )
        elif transform == "Neg":
            constant_nodes.append(helper.make_node("Neg", [name], [output_name]))
        else:
            raise ValueError(f"unsupported test constant transform: {transform}")
        return output_name

    normalizer_mean_input = transformed_constant(
        "normalizer_mean", normalizer_constant_transform
    )
    normalizer_std_input = transformed_constant(
        "normalizer_std", normalizer_constant_transform
    )
    second_mean_input = transformed_constant(
        "second_mean", second_normalizer_constant_transform
    )
    second_std_input = transformed_constant(
        "second_std", second_normalizer_constant_transform
    )
    normalizer_input = "prefixed" if identity_before_normalizer else "observations"
    actor_input = "normalized_twice" if second_normalizer else "normalized"
    linear_output = "linear_raw" if actor_post_normalization else "linear"
    nodes = [
        *constant_nodes,
        *(
            [helper.make_node("Identity", ["observations"], ["prefixed"])]
            if identity_before_normalizer
            else []
        ),
        helper.make_node(
            "Sub", [normalizer_input, normalizer_mean_input], ["centered"]
        ),
        helper.make_node("Div", ["centered", normalizer_std_input], ["normalized"]),
        *(
            [
                helper.make_node(
                    "Sub", ["normalized", second_mean_input], ["centered_twice"]
                ),
                helper.make_node(
                    "Div",
                    ["centered_twice", second_std_input],
                    ["normalized_twice"],
                ),
            ]
            if second_normalizer
            else []
        ),
        helper.make_node(
            "MatMul",
            ["observations" if bypass_normalizer else actor_input, "weights"],
            [linear_output],
        ),
        *(
            [
                helper.make_node(
                    "Sub", ["linear_raw", "actor_mean"], ["actor_centered"]
                ),
                helper.make_node("Div", ["actor_centered", "actor_std"], ["linear"]),
            ]
            if actor_post_normalization
            else []
        ),
        helper.make_node("Add", ["linear", "bias"], ["actions"]),
    ]
    initializers = [
        normalizer_mean,
        normalizer_std,
        weights,
        bias,
        *constant_initializers,
    ]
    if second_normalizer:
        initializers.extend([second_mean, second_std])
    if actor_post_normalization:
        initializers.extend([actor_mean, actor_std])
    graph = helper.make_graph(
        nodes,
        "fixture-policy",
        [observations],
        [actions],
        initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
    )
    metadata = {
        "microduck.task_id": task_id,
        "microduck.source_commit": SOURCE_COMMIT,
        "microduck.observation_contract": OBSERVATION_CONTRACT,
        "microduck.action_contract": ACTION_CONTRACT,
        "microduck.checkpoint": "model_100.pt",
        "microduck.run_identity": "entity/project/run-id",
        "microduck.normalization": "EMPIRICAL_NORMALIZATION_V1",
        "microduck.normalization_graph_sha256": hashlib.sha256(
            model.graph.SerializeToString()
        ).hexdigest(),
    }
    metadata.update(metadata_overrides or {})
    for key, value in sorted(metadata.items()):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, path)


def _write_model(
    path: Path,
    *,
    backlash: bool = False,
    actuator_kind: str = "position",
    include_file: str | None = None,
    actuator_gear: float = 1.0,
    extra_passive_actuator: bool = False,
    freejoint_on_child: bool = False,
    gyro_kind: str = "gyro",
    imu_site_quat: str = "1 0 0 0",
    roller_topology: bool = False,
    floor_pos: str = "0 0 0",
    floor_quat: str = "1 0 0 0",
    floor_contype: int = 1,
    floor_conaffinity: int = 1,
    wheel_contype: int = 1,
    wheel_conaffinity: int = 1,
    trunk_contype: int = 1,
    trunk_conaffinity: int = 1,
    exact_foot_topology: bool = True,
    extra_passive_wheel_joint: bool = False,
    weld_trunk: bool = False,
) -> None:
    mesh_vertices = (
        "-0.01 -0.01 -0.002  0.01 -0.01 -0.002  "
        "0.01 0.01 -0.002  -0.01 0.01 -0.002  "
        "-0.01 -0.01 0.002  0.01 -0.01 0.002  "
        "0.01 0.01 0.002  -0.01 0.01 0.002"
    )
    mesh_faces = (
        "0 2 1 0 3 2 4 5 6 4 6 7 0 1 5 0 5 4 1 2 6 1 6 5 2 3 7 2 7 6 3 0 4 3 4 7"
    )
    bodies: list[str] = []
    closing: list[str] = []
    for index, joint in enumerate(CONTROLLED_SERVO_JOINTS):
        if joint == "left_ankle":
            body_name = "ankle_l_v1" if roller_topology else "ankle_left"
        elif joint == "right_ankle":
            body_name = "ankle_r_v1" if roller_topology else "ankle_right"
        else:
            body_name = f"link_{index}"
        contact_z = -0.12 - 0.005 * index
        foot_contact = []
        if not roller_topology and joint in {"left_ankle", "right_ankle"}:
            side = "left" if joint == "left_ankle" else "right"
            geom_name = (
                f"{side}_foot_collision"
                if exact_foot_topology
                else f"spoof_{side}_ankle_contact"
            )
            foot_contact = [
                (
                    f'<geom name="{geom_name}" type="mesh" mesh="sole_{side}" '
                    f'pos="0 {0.03 if side == "left" else -0.03} {contact_z}" '
                    'mass="0" contype="1" conaffinity="1"/>'
                )
            ]
        wheel_bodies = []
        if roller_topology and joint in {"left_ankle", "right_ankle"}:
            wheels = (
                (("LF", "tire"), ("LR", "tire_2"))
                if joint == "left_ankle"
                else (("RF", "tire_3"), ("RR", "tire_4"))
            )
            wheel_bodies = [
                f'<body name="{wheel_body}" '
                f'pos="{-0.03 if wheel in {"LF", "RF"} else 0.03} 0 {contact_z}">'
                f'<joint name="passive_{wheel}_wheel" type="hinge" axis="0 0 1"/>'
                f'<geom type="mesh" mesh="tire" mass="0.001" '
                f'contype="{wheel_contype}" conaffinity="{wheel_conaffinity}"/>'
                "</body>"
                for wheel, wheel_body in wheels
            ]
            if extra_passive_wheel_joint and joint == "left_ankle":
                wheel_bodies.append(
                    '<body name="spare_tire"><joint name="passive_spare_wheel" '
                    'type="hinge"/><geom type="sphere" size="0.001"/></body>'
                )
        bodies.extend(
            [
                f'<body name="{body_name}" pos="0 0 0.005">',
                f'<joint name="{joint}" type="hinge" axis="0 0 1" range="-2 2" armature="0.01" damping="0.1"/>',
                *(
                    [
                        f'<joint name="passive_{joint}_backlash" type="hinge" axis="0 0 1" range="-0.1 0.1"/>'
                    ]
                    if backlash
                    else []
                ),
                '<geom type="sphere" size="0.002" mass="0.01"/>',
                *foot_contact,
                *wheel_bodies,
            ]
        )
        closing.append("</body>")

    def actuator(joint: str) -> str:
        common = (
            f'name="servo_{joint}" joint="{joint}" '
            f'gear="{actuator_gear}" ctrlrange="-2 2"'
        )
        if actuator_kind == "position":
            return f'<position {common} kp="1"/>'
        if actuator_kind == "position_infinite_gain":
            return f'<position {common} kp="inf"/>'
        if actuator_kind == "general_user_gain":
            return (
                f'<general {common} gaintype="user" biastype="affine" '
                'gainprm="1" biasprm="0 -1 0"/>'
            )
        if actuator_kind == "general_dynamic":
            return (
                f'<general {common} dyntype="filter" dynprm="0.1" '
                'gaintype="fixed" biastype="affine" '
                'gainprm="1" biasprm="0 -1 0"/>'
            )
        if actuator_kind == "general_affine_offset":
            return (
                f'<general {common} gaintype="fixed" biastype="affine" '
                'gainprm="1" biasprm="0.25 -1 0"/>'
            )
        if actuator_kind == "general_velocity_bias":
            return (
                f'<general {common} gaintype="fixed" biastype="affine" '
                'gainprm="1" biasprm="0 -1 0.25"/>'
            )
        if actuator_kind == "general_negative_infinite_bias":
            return (
                f'<general {common} gaintype="fixed" biastype="affine" '
                'gainprm="1" biasprm="0 -1 -inf"/>'
            )
        return f"<{actuator_kind} {common}/>"

    actuators = "\n".join(
        actuator(joint) for joint in reversed(CONTROLLED_SERVO_JOINTS)
    )
    if extra_passive_actuator:
        actuators += (
            '\n<motor name="passive_drive" joint="passive_wheel" ctrlrange="-1 1"/>'
        )
    path.write_text(
        f"""
<mujoco model="microduck-runtime-fixture">
  <compiler angle="radian"/>
  {f'<include file="{include_file}"/>' if include_file else ""}
  <asset>
    <mesh name="sole_left" vertex="{mesh_vertices}" face="{mesh_faces}"/>
    <mesh name="sole_right" vertex="{mesh_vertices}" face="{mesh_faces}"/>
    <mesh name="tire" vertex="{mesh_vertices}" face="{mesh_faces}"/>
  </asset>
  <option timestep="0.005" gravity="0 0 0"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" pos="{floor_pos}" quat="{
            floor_quat
        }" contype="{floor_contype}" conaffinity="{floor_conaffinity}"/>
    <body name="trunk_base" pos="0 0 0.12">
      {"" if freejoint_on_child else '<freejoint name="trunk_base_freejoint"/>'}
      <geom type="sphere" size="0.01" mass="0.1" contype="{
            trunk_contype
        }" conaffinity="{trunk_conaffinity}"/>
      <site name="imu" quat="{imu_site_quat}"/>
      {
            '<body name="roller"><joint name="passive_wheel" type="hinge"/>'
            '<geom type="sphere" size="0.002" mass="0.001"/></body>'
            if not roller_topology or extra_passive_actuator
            else ""
        }
      {"".join(bodies)}
      {"".join(reversed(closing))}
    </body>
    {
            (
                '<body name="floating_sensor"><freejoint name="trunk_base_freejoint"/>'
                '<geom type="sphere" size="0.002" mass="0.001"/></body>'
            )
            if freejoint_on_child
            else ""
        }
  </worldbody>
  {'<equality><weld body1="trunk_base"/></equality>' if weld_trunk else ""}
  <actuator>{actuators}</actuator>
  <sensor><{gyro_kind} name="imu_ang_vel" site="imu"/></sensor>
</mujoco>
""".strip()
    )


def _write_verified_bundle(
    root: Path,
    *,
    policy_output: np.ndarray | None = None,
    input_dimension: int = 61,
    tensor_type: int = TensorProto.FLOAT,
    normalizer_tensor_type: int | None = None,
    backlash: bool = False,
    actuator_kind: str = "position",
    actuator_gear: float = 1.0,
    extra_passive_actuator: bool = False,
    freejoint_on_child: bool = False,
    gyro_kind: str = "gyro",
    imu_site_quat: str = "1 0 0 0",
    include_dependency: bool = False,
    declare_dependency: bool = True,
    observation_dependent: bool = False,
    metadata_overrides: dict[str, str] | None = None,
    runtime_requirements: dict[str, str] | None = None,
    normalizer_mean_values: np.ndarray | None = None,
    normalizer_std_values: np.ndarray | None = None,
    identity_before_normalizer: bool = False,
    bypass_normalizer: bool = False,
    action_code: str = "WALK_VELOCITY",
    task_id: str = TASK_ID,
    roller_topology: bool = False,
    floor_pos: str = "0 0 0",
    floor_quat: str = "1 0 0 0",
    floor_contype: int = 1,
    floor_conaffinity: int = 1,
    wheel_contype: int = 1,
    wheel_conaffinity: int = 1,
    trunk_contype: int = 1,
    trunk_conaffinity: int = 1,
    exact_foot_topology: bool = True,
    extra_passive_wheel_joint: bool = False,
    weld_trunk: bool | None = None,
    model_license_status: str = "DISTRIBUTION_CLEARED",
) -> PolicyBundle:
    model_path = root / "models" / "robot.xml"
    policy_path = root / "policies" / "walk.onnx"
    license_path = root / "licenses" / "Apache-2.0.txt"
    model_path.parent.mkdir(parents=True)
    policy_path.parent.mkdir(parents=True)
    license_path.parent.mkdir(parents=True)
    license_path.write_bytes((Path(__file__).parents[1] / "LICENSE").read_bytes())
    dependency_path = root / "models" / "extra.xml"
    if include_dependency:
        dependency_path.write_text("<mujoco><default/></mujoco>")
    _write_model(
        model_path,
        backlash=backlash,
        actuator_kind=actuator_kind,
        actuator_gear=actuator_gear,
        extra_passive_actuator=extra_passive_actuator,
        freejoint_on_child=freejoint_on_child,
        gyro_kind=gyro_kind,
        imu_site_quat=imu_site_quat,
        roller_topology=roller_topology,
        floor_pos=floor_pos,
        floor_quat=floor_quat,
        floor_contype=floor_contype,
        floor_conaffinity=floor_conaffinity,
        wheel_contype=wheel_contype,
        wheel_conaffinity=wheel_conaffinity,
        trunk_contype=trunk_contype,
        trunk_conaffinity=trunk_conaffinity,
        exact_foot_topology=exact_foot_topology,
        extra_passive_wheel_joint=extra_passive_wheel_joint,
        weld_trunk=action_code == "STAND" if weld_trunk is None else weld_trunk,
        include_file="extra.xml" if include_dependency else None,
    )
    _write_policy(
        policy_path,
        output=policy_output,
        input_dimension=input_dimension,
        tensor_type=tensor_type,
        normalizer_tensor_type=normalizer_tensor_type,
        observation_dependent=observation_dependent,
        metadata_overrides=metadata_overrides,
        normalizer_mean_values=normalizer_mean_values,
        normalizer_std_values=normalizer_std_values,
        identity_before_normalizer=identity_before_normalizer,
        bypass_normalizer=bypass_normalizer,
        task_id=task_id,
    )
    normalized_fingerprint: str | None = None
    try:
        normalized_fingerprint = inspect_normalized_actor(
            onnx.load(policy_path)
        ).fingerprint
    except ValueError:
        pass
    default_runtime_requirements = {
        "observationContract": OBSERVATION_CONTRACT,
        "actionContract": ACTION_CONTRACT,
        "normalization": "BAKED_IN_ONNX",
        "normalizedGraphFingerprint": normalized_fingerprint or "INVALID",
    }
    policy = PolicyArtifact(
        policyRef="walk-policy",
        path="policies/walk.onnx",
        digest=_digest(policy_path),
        taskId=task_id,
        checkpoint="model_100.pt",
        experimentRef="entity/project/run-id",
        runtimeRequirements=(
            runtime_requirements
            if runtime_requirements is not None
            else default_runtime_requirements
        ),
    )
    actions = [
        code_owned_action_definition(
            code,
            availability="AVAILABLE" if code == action_code else "UNAVAILABLE",
            policy_ref=policy.policyRef if code == action_code else None,
            unavailable_reason=(
                None if code == action_code else "POLICY_ARTIFACT_MISSING"
            ),
        )
        for code in CODE_OWNED_ACTION_CODES
    ]
    observation_contract = ObservationContract(
        identifier=OBSERVATION_CONTRACT,
        dimension=61,
        fields=[
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
        ],
        units={},
        normalization="BAKED_IN_ONNX",
    )
    action_contract = ActionContract(
        identifier=ACTION_CONTRACT,
        dimension=14,
        joints=list(CONTROLLED_SERVO_JOINTS),
        units="rad",
        scaling={},
        clipping={},
    )
    unsigned = UnsignedPolicyBundleManifest(
        schema="MICRODUCK_POLICY_BUNDLE_V1",
        bundleId="org.microduck.fixture",
        bundleVersion="1.0.0",
        createdAt=datetime(2026, 8, 29, tzinfo=UTC),
        sourceRepository="microduck-rl",
        sourceCommit=SOURCE_COMMIT,
        robotModel="MICRODUCK",
        observationContract=observation_contract,
        actionContract=action_contract,
        model=ModelArtifact(path="models/robot.xml", digest=_digest(model_path)),
        policies=[policy],
        actions=actions,
        qualification={
            "artifacts": [],
            "modelTerrain": "flat",
            "scenarioProfile": "SEEDED_SERVO_RESET_V1",
            "modelClosure": (
                [{"path": "models/extra.xml", "digest": _digest(dependency_path)}]
                if include_dependency and declare_dependency
                else []
            ),
        },
        license={
            "software": {
                "identifier": "Apache-2.0",
                "artifactPaths": ["licenses/Apache-2.0.txt"],
            },
            "modelAssets": {
                "identifier": "Apache-2.0",
                "distributionStatus": model_license_status,
                "artifactPaths": ["licenses/Apache-2.0.txt"],
            },
            "artifacts": [
                {
                    "path": "licenses/Apache-2.0.txt",
                    "digest": _digest(license_path),
                }
            ],
        },
    )
    artifact_digests = {
        unsigned.model.path: unsigned.model.digest,
        policy.path: policy.digest,
        "licenses/Apache-2.0.txt": _digest(license_path),
    }
    if include_dependency and declare_dependency:
        artifact_digests["models/extra.xml"] = _digest(dependency_path)
    bundle = publish_policy_bundle(unsigned, artifact_digests)
    (root / "microduck-policy-bundle.json").write_text(
        bundle.model_dump_json(by_alias=True, exclude_none=True)
    )
    return load_verified_bundle(root)


def _request() -> TaskCreateRequest:
    return TaskCreateRequest(
        schema="MICRODUCK_SIM_TASK_V1",
        taskId="1" * 32,
        actionCode="WALK_VELOCITY",
        bundleVersion="1.0.0",
        bundleDigest="sha256:" + "0" * 64,
        parameters={"vxMps": 0.1, "vyMps": -0.2, "yawRateRadps": 0.3},
        scenario={"terrain": "flat", "seed": 7},
        leaseMs=500,
        requestedBy="test",
    )


def _rewrite_as_stand_bundle(root: Path, source: PolicyBundle) -> PolicyBundle:
    policy = source.policies[0].model_copy(
        update={
            "runtimeRequirements": source.policies[0].runtimeRequirements
            | {"completionEvaluator": "os.system('must-not-run')"}
        }
    )
    actions = [
        code_owned_action_definition(
            code,
            availability="AVAILABLE" if code == "STAND" else "UNAVAILABLE",
            policy_ref=policy.policyRef if code == "STAND" else None,
            unavailable_reason=(None if code == "STAND" else "POLICY_ARTIFACT_MISSING"),
        )
        for code in CODE_OWNED_ACTION_CODES
    ]
    unsigned = unsigned_policy_bundle_manifest(source).model_copy(
        update={"policies": [policy], "actions": actions}
    )
    digests = {
        source.model.path: source.model.digest,
        policy.path: policy.digest,
        **{
            item.path: item.digest for item in source.license.artifacts
        },
    }
    rewritten = publish_policy_bundle(unsigned, digests)
    (root / "microduck-policy-bundle.json").write_text(
        rewritten.model_dump_json(by_alias=True, exclude_none=True)
    )
    return load_verified_bundle(root)


def test_runtime_contract_exposes_only_controlled_servos() -> None:
    """Including passive roller/backlash joints would misalign every policy output."""
    assert MicroduckMujocoRuntime.controlled_joint_names == CONTROLLED_SERVO_JOINTS
    assert len(MicroduckMujocoRuntime.controlled_joint_names) == 14
    assert all(
        not name.startswith("passive_")
        for name in MicroduckMujocoRuntime.controlled_joint_names
    )


def test_emergency_stop_is_independent_of_the_primary_runtime_lock(
    tmp_path: Path,
) -> None:
    """A wedged policy/control call must not prevent zero/disable emergency intent."""
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root)
    runtime = MicroduckMujocoRuntime(root, bundle, realtime=False)
    request = _request().model_copy(update={"bundleDigest": bundle.bundleDigest})
    runtime.start(bundle.actions[0], request)
    lock_held = threading.Event()
    release = threading.Event()

    def hold_primary_lock() -> None:
        with runtime._lock:
            lock_held.set()
            release.wait()

    holder = threading.Thread(target=hold_primary_lock, daemon=True)
    holder.start()
    assert lock_held.wait(timeout=0.2)
    started = time.monotonic()
    try:
        runtime.emergency_stop("RUNTIME_UNRESPONSIVE")
        assert time.monotonic() - started < 0.1
        assert runtime._stop_event.is_set()
        assert runtime._fatal_reason == "RUNTIME_UNRESPONSIVE"
        np.testing.assert_array_equal(
            runtime._data.ctrl[runtime._actuator_indices], np.zeros(14)
        )
        np.testing.assert_array_equal(
            runtime._model.actuator_gainprm[runtime._actuator_indices],
            np.zeros((14, 10)),
        )
    finally:
        release.set()
        holder.join(timeout=0.2)


def test_emergency_retains_handle_until_idempotent_handle_specific_cleanup(
    tmp_path: Path,
) -> None:
    """Clearing the emergency handle makes the only valid thread cleanup impossible."""
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root)
    runtime = MicroduckMujocoRuntime(root, bundle, realtime=True)
    request = _request().model_copy(update={"bundleDigest": bundle.bundleDigest})
    handle = runtime.start(bundle.actions[0], request)
    policy_thread = runtime._thread
    assert policy_thread is not None

    runtime.emergency_stop("RUNTIME_UNRESPONSIVE")

    assert runtime._active_handle == handle
    assert runtime._active_request == request
    assert runtime._thread is policy_thread
    assert runtime._emergency_cleanup_required is True
    replacement = request.model_copy(update={"taskId": "2" * 32})
    with pytest.raises(RuntimeError):
        runtime.start(bundle.actions[0], replacement)

    first = runtime.safe_stop(handle, "RUNTIME_UNRESPONSIVE")
    second = runtime.safe_stop(handle, "RUNTIME_UNRESPONSIVE")

    assert second == first
    assert first.stopReason == "RUNTIME_UNRESPONSIVE"
    assert runtime._active_handle is None
    assert runtime._active_action is None
    assert runtime._active_request is None
    assert runtime._active_policy is None
    assert runtime._active_session is None
    assert runtime._thread is None
    assert runtime._emergency_cleanup_required is False
    assert not policy_thread.is_alive()
    np.testing.assert_array_equal(runtime._requested_command.twist, np.zeros(3))
    np.testing.assert_array_equal(runtime._command.twist, np.zeros(3))
    np.testing.assert_array_equal(
        runtime._data.ctrl[runtime._actuator_indices], np.zeros(14)
    )
    np.testing.assert_array_equal(
        runtime._model.actuator_gainprm[runtime._actuator_indices],
        np.zeros((14, 10)),
    )
    np.testing.assert_array_equal(
        runtime._model.actuator_biasprm[runtime._actuator_indices],
        np.zeros((14, 10)),
    )
    with pytest.raises(RuntimeError, match="restart"):
        runtime.start(bundle.actions[0], replacement)


def test_late_blocked_command_cannot_overwrite_emergency_zero_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A command returning after emergency stop must not republish motion."""
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root)
    runtime = MicroduckMujocoRuntime(root, bundle, realtime=False)
    request = _request().model_copy(update={"bundleDigest": bundle.bundleDigest})
    handle = runtime.start(bundle.actions[0], request)
    original_command_for = runtime._command_for
    command_started = threading.Event()
    command_release = threading.Event()
    outcome: dict[str, BaseException] = {}

    def blocked_command_for(*args):
        command_started.set()
        command_release.wait()
        return original_command_for(*args)

    monkeypatch.setattr(runtime, "_command_for", blocked_command_for)

    def invoke_command() -> None:
        try:
            runtime.command(handle, {"vxMps": 0.2, "vyMps": 0.0, "yawRateRadps": 0.0})
        except BaseException as exc:  # noqa: BLE001 - assert the late-call outcome.
            outcome["error"] = exc

    command_thread = threading.Thread(target=invoke_command, daemon=True)
    command_thread.start()
    assert command_started.wait(timeout=0.2)
    try:
        runtime.emergency_stop("RUNTIME_UNRESPONSIVE")
    finally:
        command_release.set()
        command_thread.join(timeout=0.2)

    assert isinstance(outcome.get("error"), RuntimeError)
    np.testing.assert_array_equal(runtime._requested_command.twist, np.zeros(3))
    np.testing.assert_array_equal(runtime._command.twist, np.zeros(3))
    np.testing.assert_array_equal(
        runtime._data.ctrl[runtime._actuator_indices], np.zeros(14)
    )


def test_emergency_after_final_command_check_disables_before_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A guard-held emergency after the last command check must be observed."""
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root)
    runtime = MicroduckMujocoRuntime(root, bundle, realtime=True)
    request = _request().model_copy(update={"bundleDigest": bundle.bundleDigest})
    handle = runtime.start(bundle.actions[0], request)
    original_reject = runtime._reject_emergency_publication_locked
    final_check_passed = threading.Event()
    guard_release = threading.Event()
    check_count = 0
    outcome: dict[str, object] = {}

    def block_after_final_check(generation: int) -> None:
        nonlocal check_count
        original_reject(generation)
        check_count += 1
        if check_count == 2:
            final_check_passed.set()
            assert guard_release.wait(timeout=1.0)

    monkeypatch.setattr(
        runtime, "_reject_emergency_publication_locked", block_after_final_check
    )

    def invoke_command() -> None:
        try:
            runtime.command(
                handle,
                {"vxMps": 0.2, "vyMps": 0.0, "yawRateRadps": 0.0},
            )
        except BaseException as exc:  # noqa: BLE001 - assert late rejection.
            outcome["error"] = exc

    command_thread = threading.Thread(target=invoke_command, daemon=True)
    command_thread.start()
    assert final_check_passed.wait(timeout=0.2)
    try:
        runtime.emergency_stop("RUNTIME_UNRESPONSIVE")
    finally:
        guard_release.set()
        command_thread.join(timeout=0.2)

    assert isinstance(outcome.get("error"), RuntimeError)
    assert not command_thread.is_alive()
    assert runtime._active_handle is None
    assert runtime._thread is None
    np.testing.assert_array_equal(runtime._requested_command.twist, np.zeros(3))
    np.testing.assert_array_equal(runtime._command.twist, np.zeros(3))
    np.testing.assert_array_equal(
        runtime._data.ctrl[runtime._actuator_indices], np.zeros(14)
    )
    np.testing.assert_array_equal(
        runtime._model.actuator_gainprm[runtime._actuator_indices],
        np.zeros((14, 10)),
    )
    np.testing.assert_array_equal(
        runtime._model.actuator_biasprm[runtime._actuator_indices],
        np.zeros((14, 10)),
    )


def test_late_blocked_start_cannot_publish_runtime_ownership_after_emergency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A start returning after emergency stop must not revive an active handle."""
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root)
    runtime = MicroduckMujocoRuntime(root, bundle, realtime=False)
    request = _request().model_copy(update={"bundleDigest": bundle.bundleDigest})
    original_reset = runtime._reset_model_locked
    reset_started = threading.Event()
    reset_release = threading.Event()
    outcome: dict[str, object] = {}

    def blocked_reset(*args):
        reset_started.set()
        reset_release.wait()
        return original_reset(*args)

    monkeypatch.setattr(runtime, "_reset_model_locked", blocked_reset)

    def invoke_start() -> None:
        try:
            outcome["result"] = runtime.start(bundle.actions[0], request)
        except BaseException as exc:  # noqa: BLE001 - assert the late-call outcome.
            outcome["error"] = exc

    start_thread = threading.Thread(target=invoke_start, daemon=True)
    start_thread.start()
    assert reset_started.wait(timeout=0.2)
    try:
        runtime.emergency_stop("RUNTIME_UNRESPONSIVE")
    finally:
        reset_release.set()
        start_thread.join(timeout=0.2)

    assert isinstance(outcome.get("error"), RuntimeError)
    assert "result" not in outcome
    assert runtime._active_handle is None
    np.testing.assert_array_equal(
        runtime._data.ctrl[runtime._actuator_indices], np.zeros(14)
    )


def test_emergency_stop_signals_immediately_when_emergency_guard_is_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The emergency signal path cannot wait behind a stalled start publication."""
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root)
    runtime = MicroduckMujocoRuntime(root, bundle, realtime=False)
    request = _request().model_copy(update={"bundleDigest": bundle.bundleDigest})
    guard_held = threading.Event()
    guard_release = threading.Event()
    reset_completed = threading.Event()
    original_reset = runtime._reset_model_locked
    outcome: dict[str, object] = {}

    def hold_emergency_guard() -> None:
        with runtime._emergency_guard:
            guard_held.set()
            guard_release.wait()

    def observed_reset(*args):
        result = original_reset(*args)
        reset_completed.set()
        return result

    monkeypatch.setattr(runtime, "_reset_model_locked", observed_reset)
    holder = threading.Thread(target=hold_emergency_guard, daemon=True)
    holder.start()
    assert guard_held.wait(timeout=0.2)

    def invoke_start() -> None:
        try:
            outcome["result"] = runtime.start(bundle.actions[0], request)
        except BaseException as exc:  # noqa: BLE001 - assert late-start rejection.
            outcome["error"] = exc

    start_thread = threading.Thread(target=invoke_start, daemon=True)
    start_thread.start()
    assert reset_completed.wait(timeout=0.2)
    emergency_done = threading.Event()
    emergency_outcome: dict[str, object] = {}

    def invoke_emergency() -> None:
        try:
            runtime.emergency_stop("RUNTIME_UNRESPONSIVE")
        except BaseException as exc:  # noqa: BLE001 - assert the emergency outcome.
            emergency_outcome["error"] = exc
        finally:
            emergency_done.set()

    emergency_thread = threading.Thread(target=invoke_emergency, daemon=True)
    emergency_thread.start()

    try:
        assert emergency_done.wait(timeout=0.05)
        assert "error" not in emergency_outcome
        assert runtime._emergency_event.is_set()
        assert runtime._stop_event.is_set()
        assert runtime._fatal_reason == "RUNTIME_UNRESPONSIVE"
    finally:
        guard_release.set()
        holder.join(timeout=0.2)
        start_thread.join(timeout=0.2)
        emergency_thread.join(timeout=0.2)

    assert isinstance(outcome.get("error"), RuntimeError)
    assert "result" not in outcome
    assert runtime._active_handle is None
    np.testing.assert_array_equal(
        runtime._data.ctrl[runtime._actuator_indices], np.zeros(14)
    )


@pytest.mark.parametrize("stop_kind", ["cancel", "watchdog"])
def test_realtime_stop_during_blocked_start_leaves_no_runtime_owner_or_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop_kind: str
) -> None:
    """Discarding a late start handle leaves its realtime policy thread in motion."""
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root)
    runtime = MicroduckMujocoRuntime(root, bundle, realtime=True)
    service = SimulatorTaskService(
        bundle,
        SqliteTaskStore(tmp_path / "tasks.sqlite3"),
        runtime,
        runtimeCallTimeoutS=0.5,
    )
    request = _request().model_copy(update={"bundleDigest": bundle.bundleDigest})
    original_reset = runtime._reset_model_locked
    reset_started = threading.Event()
    reset_release = threading.Event()
    create_done = threading.Event()
    cancel_done = threading.Event()
    create_outcome: dict[str, object] = {}
    cancel_outcome: dict[str, object] = {}

    def blocked_reset(*args):
        reset_started.set()
        reset_release.wait()
        return original_reset(*args)

    monkeypatch.setattr(runtime, "_reset_model_locked", blocked_reset)

    def invoke_create() -> None:
        try:
            create_outcome["result"] = service.create_task(request)
        except BaseException as exc:  # noqa: BLE001 - assert concurrent outcome.
            create_outcome["error"] = exc
        finally:
            create_done.set()

    def invoke_cancel() -> None:
        try:
            cancel_outcome["result"] = (
                service.cancel_task(request.taskId)
                if stop_kind == "cancel"
                else service.watchdog_failed()
            )
        except BaseException as exc:  # noqa: BLE001 - assert concurrent outcome.
            cancel_outcome["error"] = exc
        finally:
            cancel_done.set()

    create_thread = threading.Thread(target=invoke_create, daemon=True)
    cancel_thread = threading.Thread(target=invoke_cancel, daemon=True)
    create_thread.start()
    assert reset_started.wait(timeout=0.2)
    cancel_thread.start()
    deadline = time.monotonic() + 0.2
    while time.monotonic() < deadline:
        with service._lock:
            active = service._active
            stop_claimed = active is not None and active.stop_claimed
        if stop_claimed:
            break
        time.sleep(0.002)
    assert stop_claimed

    try:
        reset_release.set()
        assert create_done.wait(timeout=0.5)
        assert cancel_done.wait(timeout=0.5)
        deadline = time.monotonic() + 0.5
        expected_state = "CANCELLED" if stop_kind == "cancel" else "FAILED"
        expected_reason = "CANCELLED" if stop_kind == "cancel" else "WATCHDOG_FAILURE"
        while time.monotonic() < deadline:
            terminal = service.get_task(request.taskId)
            if terminal.state == expected_state:
                break
            time.sleep(0.002)
        assert terminal.state == expected_state
        assert terminal.stopReason == expected_reason
        assert "error" not in create_outcome
        assert "error" not in cancel_outcome
        assert runtime._active_handle is None
        assert runtime._stop_event.is_set()
        assert runtime._thread is None or not runtime._thread.is_alive()
        np.testing.assert_array_equal(runtime._requested_command.twist, np.zeros(3))
        np.testing.assert_array_equal(runtime._command.twist, np.zeros(3))
        np.testing.assert_array_equal(
            runtime._data.ctrl[runtime._actuator_indices], np.zeros(14)
        )
        np.testing.assert_array_equal(
            runtime._model.actuator_gainprm[runtime._actuator_indices],
            np.zeros((14, 10)),
        )
    finally:
        reset_release.set()
        runtime.emergency_stop("TEST_CLEANUP")
        policy_thread = runtime._thread
        if policy_thread is not None:
            policy_thread.join(timeout=1.0)
        create_thread.join(timeout=0.2)
        cancel_thread.join(timeout=0.2)


@pytest.mark.parametrize("stop_kind", ["cancel", "watchdog"])
def test_realtime_emergency_after_final_start_check_revokes_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop_kind: str
) -> None:
    """An emergency after the final check must not strand enabled actuators."""
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root)
    runtime = MicroduckMujocoRuntime(root, bundle, realtime=True)
    service = SimulatorTaskService(
        bundle,
        SqliteTaskStore(tmp_path / "tasks.sqlite3"),
        runtime,
        runtimeCallTimeoutS=0.5,
    )
    request = _request().model_copy(update={"bundleDigest": bundle.bundleDigest})
    original_reject = runtime._reject_emergency_publication_locked
    final_check_passed = threading.Event()
    publication_release = threading.Event()
    check_count = 0

    def block_after_final_check(generation: int) -> None:
        nonlocal check_count
        original_reject(generation)
        check_count += 1
        if check_count == 2:
            final_check_passed.set()
            assert publication_release.wait(timeout=1.0)

    monkeypatch.setattr(
        runtime, "_reject_emergency_publication_locked", block_after_final_check
    )
    create_done = threading.Event()
    stop_done = threading.Event()
    create_outcome: dict[str, object] = {}
    stop_outcome: dict[str, object] = {}

    def invoke_create() -> None:
        try:
            create_outcome["result"] = service.create_task(request)
        except BaseException as exc:  # noqa: BLE001 - assert concurrent outcome.
            create_outcome["error"] = exc
        finally:
            create_done.set()

    def invoke_stop() -> None:
        try:
            stop_outcome["result"] = (
                service.cancel_task(request.taskId)
                if stop_kind == "cancel"
                else service.watchdog_failed()
            )
        except BaseException as exc:  # noqa: BLE001 - assert concurrent outcome.
            stop_outcome["error"] = exc
        finally:
            stop_done.set()

    create_thread = threading.Thread(target=invoke_create, daemon=True)
    stop_thread = threading.Thread(target=invoke_stop, daemon=True)
    create_thread.start()
    assert final_check_passed.wait(timeout=0.2)
    stop_thread.start()

    try:
        assert runtime._emergency_event.wait(timeout=0.2)
        assert runtime._stop_event.is_set()
        publication_release.set()
        assert create_done.wait(timeout=0.5)
        assert stop_done.wait(timeout=0.5)
        expected_state = "CANCELLED" if stop_kind == "cancel" else "FAILED"
        expected_reason = "CANCELLED" if stop_kind == "cancel" else "WATCHDOG_FAILURE"
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            terminal = service.get_task(request.taskId)
            if terminal.state == expected_state:
                break
            time.sleep(0.002)

        assert terminal.state == expected_state
        assert terminal.stopReason == expected_reason
        assert "error" not in create_outcome
        assert "error" not in stop_outcome
        assert service._active is None
        assert runtime._active_handle is None
        assert runtime._thread is None
        np.testing.assert_array_equal(runtime._requested_command.twist, np.zeros(3))
        np.testing.assert_array_equal(runtime._command.twist, np.zeros(3))
        np.testing.assert_array_equal(
            runtime._data.ctrl[runtime._actuator_indices], np.zeros(14)
        )
        np.testing.assert_array_equal(
            runtime._model.actuator_gainprm[runtime._actuator_indices],
            np.zeros((14, 10)),
        )
        np.testing.assert_array_equal(
            runtime._model.actuator_biasprm[runtime._actuator_indices],
            np.zeros((14, 10)),
        )
        mujoco.mj_forward(runtime._model, runtime._data)
        np.testing.assert_array_equal(
            runtime._data.actuator_force[runtime._actuator_indices], np.zeros(14)
        )
        np.testing.assert_array_equal(
            runtime._data.qfrc_actuator, np.zeros(runtime._model.nv)
        )
    finally:
        publication_release.set()
        runtime.emergency_stop("TEST_CLEANUP")
        policy_thread = runtime._thread
        if policy_thread is not None:
            policy_thread.join(timeout=1.0)
        create_thread.join(timeout=0.2)
        stop_thread.join(timeout=0.2)


@pytest.mark.parametrize("stop_kind", ["cancel", "watchdog"])
def test_realtime_stop_after_runtime_start_return_uses_retained_cleanup_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop_kind: str
) -> None:
    """The service must still clean the handle returned just before registration."""
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root)
    runtime = MicroduckMujocoRuntime(root, bundle, realtime=True)
    service = SimulatorTaskService(
        bundle,
        SqliteTaskStore(tmp_path / "tasks.sqlite3"),
        runtime,
        runtimeCallTimeoutS=0.5,
    )
    request = _request().model_copy(update={"bundleDigest": bundle.bundleDigest})
    original_start = runtime.start
    start_returned = threading.Event()
    registration_release = threading.Event()
    captured: dict[str, object] = {}

    def block_after_runtime_start(action, runtime_request):
        handle = original_start(action, runtime_request)
        captured["handle"] = handle
        captured["thread"] = runtime._thread
        start_returned.set()
        assert registration_release.wait(timeout=1.0)
        return handle

    monkeypatch.setattr(runtime, "start", block_after_runtime_start)
    create_done = threading.Event()
    stop_done = threading.Event()
    create_outcome: dict[str, object] = {}
    stop_outcome: dict[str, object] = {}

    def invoke_create() -> None:
        try:
            create_outcome["result"] = service.create_task(request)
        except BaseException as exc:  # noqa: BLE001 - assert concurrent outcome.
            create_outcome["error"] = exc
        finally:
            create_done.set()

    def invoke_stop() -> None:
        try:
            stop_outcome["result"] = (
                service.cancel_task(request.taskId)
                if stop_kind == "cancel"
                else service.watchdog_failed()
            )
        except BaseException as exc:  # noqa: BLE001 - assert concurrent outcome.
            stop_outcome["error"] = exc
        finally:
            stop_done.set()

    create_thread = threading.Thread(target=invoke_create, daemon=True)
    stop_thread = threading.Thread(target=invoke_stop, daemon=True)
    create_thread.start()
    assert start_returned.wait(timeout=0.2)
    handle = captured["handle"]
    policy_thread = captured["thread"]
    assert isinstance(handle, RuntimeHandle)
    assert isinstance(policy_thread, threading.Thread)
    with service._lock:
        first_generation = service._active
        assert first_generation is not None
        assert first_generation.handle is None
        assert first_generation.start_pending is True

    stop_thread.start()
    assert runtime._emergency_event.wait(timeout=0.2)

    try:
        with service._lock:
            assert service._active is first_generation
            assert service._active.handle is None
        assert runtime._active_handle == handle
        assert runtime._active_request == request
        assert runtime._emergency_cleanup_required is True

        registration_release.set()
        assert create_done.wait(timeout=0.5)
        assert stop_done.wait(timeout=0.5)
        expected_state = "CANCELLED" if stop_kind == "cancel" else "FAILED"
        expected_reason = "CANCELLED" if stop_kind == "cancel" else "WATCHDOG_FAILURE"
        terminal = service.get_task(request.taskId)
        assert terminal.state == expected_state
        assert terminal.stopReason == expected_reason
        assert terminal.evidence is not None
        assert terminal.evidence.metrics.get("safetyFailure") != "SAFE_STOP_FAILED"
        assert "safetyCode" not in service.events_after(request.taskId, -1)[-1].payload
        assert "error" not in create_outcome
        assert "error" not in stop_outcome
        assert service._active is None
        assert service._next_generation == first_generation.generation + 1

        repeated = runtime.safe_stop(handle, expected_reason)
        assert repeated == runtime._stopped_evidence[request.taskId]
        assert runtime._active_handle is None
        assert runtime._active_action is None
        assert runtime._active_request is None
        assert runtime._active_policy is None
        assert runtime._active_session is None
        assert runtime._thread is None
        assert runtime._emergency_cleanup_required is False
        assert not policy_thread.is_alive()
        np.testing.assert_array_equal(runtime._requested_command.twist, np.zeros(3))
        np.testing.assert_array_equal(runtime._command.twist, np.zeros(3))
        np.testing.assert_array_equal(
            runtime._data.ctrl[runtime._actuator_indices], np.zeros(14)
        )
        np.testing.assert_array_equal(
            runtime._model.actuator_gainprm[runtime._actuator_indices],
            np.zeros((14, 10)),
        )
        np.testing.assert_array_equal(
            runtime._model.actuator_biasprm[runtime._actuator_indices],
            np.zeros((14, 10)),
        )
    finally:
        registration_release.set()
        create_thread.join(timeout=0.2)
        stop_thread.join(timeout=0.2)


def test_runtime_readiness_rejects_available_action_with_wrong_policy_task_identity(
    tmp_path: Path,
) -> None:
    """Deferring task-family validation until POST would create false catalog availability."""
    root = tmp_path / "bundle"
    source = _write_verified_bundle(root)
    wrong_policy = source.policies[0].model_copy(
        update={"taskId": "Mjlab-VelStand-Flat-MicroDuck"}
    )
    unsigned = unsigned_policy_bundle_manifest(source).model_copy(
        update={"policies": [wrong_policy]}
    )
    artifact_digests = {
        source.model.path: source.model.digest,
        source.policies[0].path: source.policies[0].digest,
        **{
            item.path: item.digest for item in source.license.artifacts
        },
    }
    rewritten = publish_policy_bundle(unsigned, artifact_digests)
    (root / "microduck-policy-bundle.json").write_text(
        rewritten.model_dump_json(by_alias=True, exclude_none=True)
    )
    with pytest.raises(ValueError, match="policy identity"):
        load_verified_bundle(root)


@pytest.mark.parametrize(
    "fixture",
    [
        {"floor_quat": "0.70710678 0.70710678 0 0"},
        {"floor_pos": "0 0 2"},
        {
            "floor_contype": 4,
            "floor_conaffinity": 4,
            "trunk_contype": 4,
            "trunk_conaffinity": 4,
        },
    ],
)
def test_runtime_rejects_unusable_or_collision_incompatible_flat_floor(
    tmp_path: Path, fixture: dict[str, object]
) -> None:
    """A flat label is insufficient unless the loaded floor can contact the robot."""
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root, **fixture)

    with pytest.raises(ValueError, match="qualified model capabilities"):
        MicroduckMujocoRuntime(root, bundle, realtime=False)


def test_runtime_rejects_rollers_whose_masks_cannot_collide_with_floor(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(
        root,
        action_code="ROLLER_VELOCITY",
        task_id="Mjlab-Velocity-Flat-MicroDuck-Rollers",
        roller_topology=True,
        wheel_contype=4,
        wheel_conaffinity=4,
    )

    with pytest.raises(ValueError, match="qualified model capabilities"):
        MicroduckMujocoRuntime(root, bundle, realtime=False)


def test_runtime_rejects_arbitrary_ankle_descendants_as_floor_contacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root, exact_foot_topology=False)

    with pytest.raises(ValueError, match="qualified model capabilities"):
        MicroduckMujocoRuntime(root, bundle, realtime=False)


def test_runtime_rejects_extra_passive_wheel_joint(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(
        root,
        action_code="ROLLER_VELOCITY",
        task_id="Mjlab-Velocity-Flat-MicroDuck-Rollers",
        roller_topology=True,
        extra_passive_wheel_joint=True,
    )

    with pytest.raises(ValueError, match="qualified model capabilities"):
        MicroduckMujocoRuntime(root, bundle, realtime=False)


def test_runtime_readiness_rejects_action_preconditions_outside_qualification(
    tmp_path: Path,
) -> None:
    """Catalog availability is false if service preconditions and runtime terrain can never agree."""
    root = tmp_path / "bundle"
    source = _write_verified_bundle(root)
    action = source.actions[0].model_copy(
        update={
            "preconditions": {
                "allowedTerrains": ["slope"],
                "scenarioProfile": "SEEDED_SERVO_RESET_V1",
            }
        }
    )
    unsigned = unsigned_policy_bundle_manifest(source).model_copy(
        update={"actions": [action, *source.actions[1:]]}
    )
    artifacts = {
        source.model.path: source.model.digest,
        source.policies[0].path: source.policies[0].digest,
        **{
            item.path: item.digest for item in source.license.artifacts
        },
    }
    rewritten = publish_policy_bundle(unsigned, artifacts)
    (root / "microduck-policy-bundle.json").write_text(
        rewritten.model_dump_json(by_alias=True, exclude_none=True)
    )

    with pytest.raises(ValueError, match="code-owned.*action envelope"):
        load_verified_bundle(root)


def test_backlash_encoder_observation_and_status_sum_exact_named_companions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root, backlash=True)
    runtime = MicroduckMujocoRuntime(root, bundle, realtime=False)
    runtime._data.qpos[runtime._joint_qpos_indices] = DEFAULT_JOINT_POSE + 0.1
    runtime._data.qpos[runtime._backlash_qpos_indices] = 0.025
    runtime._data.qvel[runtime._joint_qvel_indices] = 0.2
    runtime._data.qvel[runtime._backlash_qvel_indices] = -0.05
    mujoco.mj_forward(runtime._model, runtime._data)

    status = runtime.status()

    np.testing.assert_allclose(status.jointPositionsRad, DEFAULT_JOINT_POSE + 0.125)
    np.testing.assert_allclose(status.jointVelocitiesRadps, 0.15)
    assert not any(
        actuator in runtime._actuator_indices
        for actuator in np.flatnonzero(
            np.isin(runtime._model.actuator_trnid[:, 0], runtime._backlash_joint_ids)
        )
    )


def test_runtime_rejects_non_position_actuator_semantics(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root, actuator_kind="motor")

    with pytest.raises(ValueError, match="position actuator"):
        MicroduckMujocoRuntime(root, bundle, realtime=False)


@pytest.mark.parametrize(
    ("fixture", "message"),
    [
        ({"actuator_gear": 2.0}, "unit positive joint gear"),
        ({"actuator_kind": "general_user_gain"}, "fixed gain"),
        ({"actuator_kind": "general_dynamic"}, "actuator dynamics"),
        ({"actuator_kind": "general_affine_offset"}, "position actuator semantics"),
        ({"actuator_kind": "general_velocity_bias"}, "position actuator semantics"),
        ({"actuator_kind": "position_infinite_gain"}, "position actuator semantics"),
        (
            {"actuator_kind": "general_negative_infinite_bias"},
            "position actuator semantics",
        ),
        ({"extra_passive_actuator": True}, "exactly 14 total actuators"),
    ],
)
def test_runtime_rejects_actuator_semantics_that_are_not_radian_position_targets(
    tmp_path: Path, fixture: dict[str, object], message: str
) -> None:
    """General or passive transmissions can satisfy loose gain/bias checks but change action meaning."""
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root, **fixture)

    with pytest.raises(ValueError, match=message):
        MicroduckMujocoRuntime(root, bundle, realtime=False)


def test_governed_loop_measures_start_cadence_and_faults_repeated_overruns(
    tmp_path: Path,
) -> None:
    class Clock:
        now = 10.0

        def __call__(self) -> float:
            return self.now

    clock = Clock()
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root)
    runtime = MicroduckMujocoRuntime(
        root, bundle, realtime=False, monotonic_clock=clock
    )
    request = _request().model_copy(update={"bundleDigest": bundle.bundleDigest})
    runtime.start(bundle.actions[0], request)
    real_step = runtime._control_step

    def slow_step() -> None:
        real_step()
        clock.now += 0.03

    runtime._control_step = slow_step
    runtime._wait = lambda _: False
    runtime._governed_loop()

    status = runtime.status()
    assert runtime._loop_overruns == 3
    assert status.loopFrequencyHz == pytest.approx(100.0 / 3.0)
    assert status.health["ready"] is False
    assert status.health["reasonCodes"] == ["CONTROL_LOOP_OVERRUN"]


def test_service_tick_observes_concrete_runtime_fault_and_zeros_applied_motion(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root)
    runtime = MicroduckMujocoRuntime(root, bundle, realtime=False)
    service = SimulatorTaskService(
        bundle,
        SqliteTaskStore(tmp_path / "state.sqlite3"),
        runtime,
        monotonic_clock=lambda: 100.0,
    )
    request = _request().model_copy(update={"bundleDigest": bundle.bundleDigest})
    service.create_task(request)
    with runtime._lock:
        runtime._fail_locked("CONTROL_LOOP_OVERRUN")

    service.tick()

    terminal = service.get_task(request.taskId)
    assert terminal.state == "FAILED"
    assert terminal.stopReason == "CONTROL_LOOP_OVERRUN"
    assert runtime.status().appliedMotion["twist"] == [0.0, 0.0, 0.0]


def test_runtime_executes_real_onnx_at_50hz_and_maps_actions_by_joint_name(
    tmp_path: Path,
) -> None:
    """Using raw actuator indices would reverse this fixture's controlled targets."""
    bundle = _write_verified_bundle(tmp_path / "bundle")
    runtime = MicroduckMujocoRuntime(tmp_path / "bundle", bundle, realtime=False)
    action = bundle.actions[0]
    request = _request().model_copy(update={"bundleDigest": bundle.bundleDigest})

    runtime.validate(action, request)
    handle = runtime.start(action, request)
    sample = runtime.sample(handle)
    status = runtime.status()

    assert sample.running is True
    assert status.schema == "BIPED_POSE_V1"
    assert status.simulationTimeS == pytest.approx(0.02)
    assert status.loopFrequencyHz == pytest.approx(50.0)
    assert status.activeTaskId == request.taskId
    assert status.activeActionCode == "WALK_VELOCITY"
    assert status.activePolicyRef == "walk-policy"
    assert status.requestedMotion["twist"] == [0.1, -0.2, 0.3]
    assert status.appliedMotion["twist"] == [0.1, -0.2, 0.3]
    expected_action = np.linspace(-0.13, 0.13, 14, dtype=np.float32)
    np.testing.assert_allclose(
        status.policyTarget["jointPositionsRad"],
        DEFAULT_JOINT_POSE + expected_action,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        runtime._data.ctrl[runtime._actuator_indices],
        DEFAULT_JOINT_POSE + expected_action,
        atol=1e-6,
    )
    passive_id = mujoco.mj_name2id(
        runtime._model, mujoco.mjtObj.mjOBJ_JOINT, "passive_wheel"
    )
    assert passive_id not in runtime._model.actuator_trnid[:, 0]
    assert len(status.jointPositionsRad) == len(status.jointVelocitiesRadps) == 14
    assert status.fallen is False
    assert status.limp is False
    assert status.health == {
        "ready": True,
        "healthy": True,
        "reasonCodes": [],
        "baseLinearVelocityFrame": "WORLD",
        "baseAngularVelocityFrame": "TRUNK_BODY",
    }

    evidence = runtime.safe_stop(handle, "LEASE_EXPIRED")
    assert evidence.stopReason == "LEASE_EXPIRED"
    assert evidence.metrics == {
        "actionCode": "WALK_VELOCITY",
        "baseTravelM": pytest.approx(0.0),
        "bundleDigest": bundle.bundleDigest,
        "checkpoint": "model_100.pt",
        "durationS": pytest.approx(0.02),
        "fallen": False,
        "finalBaseHeightM": pytest.approx(0.12),
        "finalTiltRad": pytest.approx(0.0),
        "maxAbsAction": pytest.approx(0.13),
        "energyProxy": pytest.approx(0.00182),
        "actuatorClampSteps": 0,
        "physicalJointLimitViolations": 0,
        "maxTiltRad": pytest.approx(0.0),
        "minBaseHeightM": pytest.approx(0.12),
        "mjcfDigest": bundle.model.digest,
        "onnxDigest": bundle.policies[0].digest,
        "runIdentity": "entity/project/run-id",
        "rngSeed": 7,
        "terrainIdentity": "flat",
        "scenarioProfile": "SEEDED_SERVO_RESET_V1",
        "resetPerturbationL2Rad": pytest.approx(0.01045295),
        "resetProfile": "DEFAULT_STANDING",
        "sourceCommit": SOURCE_COMMIT,
        "steps": 1,
        "loopOverruns": 0,
        "trackingError": pytest.approx(0.355429, abs=1e-6),
        "trackingErrorSum": pytest.approx(0.355429, abs=1e-6),
        "trackingErrorMax": pytest.approx(0.355429, abs=1e-6),
        "trackingErrorSamples": 1,
    }
    assert runtime.status().limp is False
    assert runtime.status().activeTaskId is None


@pytest.mark.parametrize(
    ("tracking_error_sum", "sample_count", "expected"),
    [
        pytest.param(10.0, 100, 0.1, id="exact-decimal"),
        pytest.param(0.1 + 0.2, 3, 0.1, id="ordinary-floating-sum"),
        pytest.param(1.23456789, 3, 0.411523, id="six-decimal-rounding"),
    ],
)
def test_tracking_mean_has_one_deterministic_runtime_serialization_rule(
    tracking_error_sum: float, sample_count: int, expected: float
) -> None:
    """Changing runtime precision would break genuine qualification round trips."""
    assert canonical_tracking_mean(tracking_error_sum, sample_count) == expected


@pytest.mark.parametrize(
    ("tracking_error_sum", "sample_count"),
    [
        pytest.param(0.0, 0, id="zero-count"),
        pytest.param(1.0, -1, id="negative-count"),
        pytest.param(1.0, True, id="boolean-count"),
        pytest.param(1.0, 1.5, id="fractional-count"),
        pytest.param(math.nan, 1, id="nan-sum"),
        pytest.param(math.inf, 1, id="infinite-sum"),
        pytest.param(-1.0, 1, id="negative-sum"),
    ],
)
def test_tracking_mean_rejects_invalid_canonical_evidence(
    tracking_error_sum: float, sample_count: object
) -> None:
    """A missing denominator or invalid sum cannot define trusted tracking evidence."""
    with pytest.raises(ValueError, match="tracking"):
        canonical_tracking_mean(tracking_error_sum, sample_count)  # type: ignore[arg-type]


def test_runtime_accumulates_tracking_and_distinguishes_clamp_from_joint_limit(
    tmp_path: Path,
) -> None:
    """A final-sample error or combined limit counter would misstate rollout quality."""
    clamped_root = tmp_path / "clamped"
    clamped_bundle = _write_verified_bundle(
        clamped_root, policy_output=np.full(14, 100.0, dtype=np.float32)
    )
    clamped_runtime = MicroduckMujocoRuntime(
        clamped_root, clamped_bundle, realtime=False
    )
    request = _request().model_copy(
        update={"bundleDigest": clamped_bundle.bundleDigest}
    )
    handle = clamped_runtime.start(clamped_bundle.actions[0], request)
    clamped_runtime.sample(handle)
    clamped_runtime.sample(handle)
    clamped = clamped_runtime.safe_stop(handle, "TEST_COMPLETE").metrics

    assert clamped["trackingErrorSamples"] == 2
    assert clamped["trackingError"] == canonical_tracking_mean(
        clamped["trackingErrorSum"], clamped["trackingErrorSamples"]
    )
    assert clamped["trackingError"] <= clamped["trackingErrorMax"]
    assert clamped["actuatorClampSteps"] == 2
    assert clamped["physicalJointLimitViolations"] == 0

    violated_root = tmp_path / "violated"
    violated_bundle = _write_verified_bundle(violated_root)
    violated_runtime = MicroduckMujocoRuntime(
        violated_root, violated_bundle, realtime=False
    )
    violated_request = _request().model_copy(
        update={"bundleDigest": violated_bundle.bundleDigest}
    )
    violated_handle = violated_runtime.start(
        violated_bundle.actions[0], violated_request
    )
    joint_id = violated_runtime._joint_ids[0]
    qpos_index = violated_runtime._joint_qpos_indices[0]
    violated_runtime._data.qpos[qpos_index] = (
        violated_runtime._model.jnt_range[joint_id][1] + 1.0
    )
    violated_runtime.sample(violated_handle)
    violated = violated_runtime.safe_stop(violated_handle, "TEST_COMPLETE").metrics

    assert violated["physicalJointLimitViolations"] == 1


def test_runtime_policy_output_depends_on_exact_command_observation_slot(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    mean = np.zeros(61, dtype=np.float32)
    std = np.ones(61, dtype=np.float32)
    mean[48] = 0.02
    std[48] = 0.04
    bundle = _write_verified_bundle(
        root,
        policy_output=np.zeros(14, dtype=np.float32),
        observation_dependent=True,
        normalizer_mean_values=mean,
        normalizer_std_values=std,
    )
    runtime = MicroduckMujocoRuntime(root, bundle, realtime=False)
    request = _request().model_copy(update={"bundleDigest": bundle.bundleDigest})
    handle = runtime.start(bundle.actions[0], request)

    runtime.sample(handle)

    assert runtime._previous_action[0] == pytest.approx(1.0)


def test_runtime_accepts_normalizer_reached_through_valid_identity_prefix(
    tmp_path: Path,
) -> None:
    """Graph provenance must follow dependencies instead of assuming Sub is node zero."""
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root, identity_before_normalizer=True)

    runtime = MicroduckMujocoRuntime(root, bundle, realtime=False)

    assert runtime.status().health["ready"] is True


@pytest.mark.parametrize(
    "fixture_options",
    [
        {"bypass_normalizer": True},
        {"normalizer_std_values": np.zeros(61, dtype=np.float32)},
        {
            "normalizer_mean_values": np.zeros(61, dtype=np.float64),
            "normalizer_tensor_type": TensorProto.DOUBLE,
        },
        {
            "runtime_requirements": {
                "observationContract": OBSERVATION_CONTRACT,
                "actionContract": ACTION_CONTRACT,
                "normalization": "BAKED_IN_ONNX",
            }
        },
    ],
)
def test_runtime_rejects_bypassed_invalid_or_unbound_normalizer(
    tmp_path: Path, fixture_options: dict[str, object]
) -> None:
    """A dead prefix, unsafe statistics, or missing external fingerprint cannot attest normalization."""
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root, **fixture_options)

    with pytest.raises(ValueError, match="normalization"):
        MicroduckMujocoRuntime(root, bundle, realtime=False)


def test_normalizer_validator_rejects_second_empirical_stats_stage(
    tmp_path: Path,
) -> None:
    """A second 61D Sub/Div changes the deployment observation a second time."""
    policy = tmp_path / "twice-normalized.onnx"
    _write_policy(policy, second_normalizer=True)

    with pytest.raises(ValueError, match="exactly one.*normalization"):
        inspect_normalized_actor(onnx.load(policy, load_external_data=False))


@pytest.mark.parametrize("transform", ["Identity", "Cast", "Reshape", "Neg"])
def test_normalizer_validator_rejects_hidden_second_stats_constant_lineage(
    tmp_path: Path, transform: str
) -> None:
    """Constant-only exporter nodes cannot hide a second 61D Sub/Div stage."""
    policy = tmp_path / f"twice-normalized-{transform.lower()}.onnx"
    _write_policy(
        policy,
        second_normalizer=True,
        second_normalizer_constant_transform=transform,
    )

    with pytest.raises(ValueError, match="exactly one.*normalization"):
        inspect_normalized_actor(onnx.load(policy, load_external_data=False))


@pytest.mark.parametrize("transform", ["Identity", "Cast", "Reshape"])
def test_normalizer_validator_accepts_valid_stats_constant_transforms(
    tmp_path: Path, transform: str
) -> None:
    """Common exporter constant transforms preserve one validated normalizer."""
    policy = tmp_path / f"normalized-{transform.lower()}.onnx"
    _write_policy(policy, normalizer_constant_transform=transform)

    inspected = inspect_normalized_actor(onnx.load(policy, load_external_data=False))

    assert inspected.fingerprint.startswith("sha256:")


def test_normalizer_validator_allows_actor_arithmetic_after_empirical_prefix(
    tmp_path: Path,
) -> None:
    """Actor-local 14D arithmetic is not a second observation normalizer."""
    policy = tmp_path / "actor-arithmetic.onnx"
    _write_policy(policy, actor_post_normalization=True)

    inspected = inspect_normalized_actor(onnx.load(policy, load_external_data=False))

    assert inspected.fingerprint.startswith("sha256:")


def test_runtime_rejects_requested_terrain_not_bound_to_loaded_model(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root)
    runtime = MicroduckMujocoRuntime(root, bundle, realtime=False)
    request = _request().model_copy(
        update={
            "bundleDigest": bundle.bundleDigest,
            "scenario": Scenario(terrain="ramp", seed=7),
        }
    )

    with pytest.raises(ValueError, match="qualified loaded model"):
        runtime.validate(bundle.actions[0], request)


def test_runtime_rejects_unknown_scenario_fields(tmp_path: Path) -> None:
    """Manifest-like free-form scenario data must not silently alter deployment semantics."""
    payload = _request().model_dump(mode="json", by_alias=True)
    payload["scenario"]["friction"] = 0.01

    with pytest.raises(ValidationError):
        TaskCreateRequest.model_validate(payload)


def test_seed_materializes_a_reproducible_physical_servo_reset(tmp_path: Path) -> None:
    """Recording a seed without changing physical state would make replay evidence misleading."""
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root)
    runtime = MicroduckMujocoRuntime(root, bundle, realtime=False)
    first_request = _request().model_copy(update={"bundleDigest": bundle.bundleDigest})
    first_handle = runtime.start(bundle.actions[0], first_request)
    first_reset = runtime._data.qpos[runtime._joint_qpos_indices].copy()
    first_evidence = runtime.safe_stop(first_handle, "CANCELLED")

    same_request = first_request.model_copy(update={"taskId": "2" * 32})
    same_handle = runtime.start(bundle.actions[0], same_request)
    same_reset = runtime._data.qpos[runtime._joint_qpos_indices].copy()
    runtime.safe_stop(same_handle, "CANCELLED")

    other_request = first_request.model_copy(
        update={"taskId": "3" * 32, "scenario": Scenario(terrain="flat", seed=8)}
    )
    other_handle = runtime.start(bundle.actions[0], other_request)
    other_reset = runtime._data.qpos[runtime._joint_qpos_indices].copy()
    runtime.safe_stop(other_handle, "CANCELLED")

    np.testing.assert_array_equal(first_reset, same_reset)
    assert not np.array_equal(first_reset, other_reset)
    assert first_evidence.metrics["rngSeed"] == 7
    assert first_evidence.metrics["scenarioProfile"] == "SEEDED_SERVO_RESET_V1"
    assert first_evidence.metrics["resetPerturbationL2Rad"] > 0.0


@pytest.mark.parametrize(
    ("action_code", "task_id", "roller_topology"),
    [
        ("WALK_VELOCITY", "Mjlab-Velocity-Flat-MicroDuck", False),
        ("VELSTAND_VELOCITY", "Mjlab-VelStand-Flat-MicroDuck", False),
        (
            "ROLLER_VELOCITY",
            "Mjlab-Velocity-Flat-MicroDuck-Rollers",
            True,
        ),
        ("SWIZZLE", "Mjlab-Velocity-Swizzle-MicroDuck", True),
    ],
)
def test_every_supported_action_emits_all_code_owned_compact_metrics(
    tmp_path: Path, action_code: str, task_id: str, roller_topology: bool
) -> None:
    """A declared metric key must be materialized, not merely listed in an action spec."""
    root = tmp_path / action_code.lower()
    bundle = _write_verified_bundle(
        root,
        action_code=action_code,
        task_id=task_id,
        roller_topology=roller_topology,
    )
    runtime = MicroduckMujocoRuntime(root, bundle, realtime=False)
    request = _request().model_copy(
        update={
            "actionCode": action_code,
            "bundleDigest": bundle.bundleDigest,
            "parameters": {"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
        }
    )
    action = next(
        action for action in bundle.actions if action.actionCode == action_code
    )
    handle = runtime.start(action, request)

    metrics = runtime.sample(handle).metrics

    assert set(ACTION_RUNTIME_SPECS[action_code].metric_keys) <= set(metrics)


def test_runtime_rechecks_artifact_hash_immediately_before_loading(
    tmp_path: Path,
) -> None:
    """Trusting only startup verification would permit artifact replacement before load."""
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root)
    (root / bundle.policies[0].path).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="artifact verification"):
        MicroduckMujocoRuntime(root, bundle, realtime=False)


def test_runtime_rejects_symlink_escape_after_bundle_verification(
    tmp_path: Path,
) -> None:
    """Resolving a replaced symlink outside the bundle would load undeclared policy bytes."""
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root)
    outside = tmp_path / "outside.onnx"
    outside.write_bytes((root / bundle.policies[0].path).read_bytes())
    (root / bundle.policies[0].path).unlink()
    (root / bundle.policies[0].path).symlink_to(outside)

    with pytest.raises(ValueError, match="bundle root"):
        MicroduckMujocoRuntime(root, bundle, realtime=False)


def test_runtime_requires_exact_declared_mjcf_dependency_closure(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(
        root, include_dependency=True, declare_dependency=False
    )

    with pytest.raises(ValueError, match="model dependency closure"):
        MicroduckMujocoRuntime(root, bundle, realtime=False)


def test_runtime_loads_declared_mjcf_dependency_from_verified_snapshot(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root, include_dependency=True)

    runtime = MicroduckMujocoRuntime(root, bundle, realtime=False)

    assert runtime.status().health["ready"] is True


@pytest.mark.parametrize(
    ("fixture_options", "message"),
    [
        ({"input_dimension": 60}, "input"),
        (
            {"metadata_overrides": {"microduck.action_contract": "WRONG"}},
            "metadata",
        ),
        ({"policy_output": np.full(14, np.nan, dtype=np.float32)}, "finite"),
        ({"tensor_type": TensorProto.DOUBLE}, r"tensor\(float\)"),
        (
            {"metadata_overrides": {"microduck.normalization": ""}},
            "normalization",
        ),
    ],
)
def test_runtime_rejects_incompatible_or_non_finite_onnx(
    tmp_path: Path, fixture_options: dict[str, object], message: str
) -> None:
    """Loading a shape-, contract-, or numeric-incompatible actor would corrupt control."""
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root, **fixture_options)

    with pytest.raises(ValueError, match=message):
        MicroduckMujocoRuntime(root, bundle, realtime=False)


def test_runtime_fail_safe_stops_when_state_becomes_non_finite(tmp_path: Path) -> None:
    """Continuing after a non-finite MuJoCo state would send undefined servo targets."""
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root)
    runtime = MicroduckMujocoRuntime(root, bundle, realtime=False)
    request = _request().model_copy(update={"bundleDigest": bundle.bundleDigest})
    handle = runtime.start(bundle.actions[0], request)
    runtime._data.qpos[runtime._joint_qpos_indices[0]] = np.nan

    sample = runtime.sample(handle)

    assert sample.running is False
    assert sample.terminalState == "FAILED"
    assert sample.stopReason == "NON_FINITE_STATE"
    assert runtime.status().limp is True
    assert runtime.status().health["ready"] is False
    assert np.all(runtime._model.actuator_gainprm[runtime._actuator_indices, 0] == 0.0)


def test_fatal_limp_clears_every_affine_force_term(tmp_path: Path) -> None:
    """Fatal limp must remain force-free even if an affine constant was introduced."""
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root)
    runtime = MicroduckMujocoRuntime(root, bundle, realtime=False)
    runtime._model.actuator_biasprm[runtime._actuator_indices, 0] = 0.25
    runtime._data.ctrl[runtime._actuator_indices] = 1.0
    mujoco.mj_forward(runtime._model, runtime._data)
    assert np.any(np.abs(runtime._data.actuator_force) > 0.0)

    with runtime._lock:
        runtime._fail_locked("TEST_FATAL")
    mujoco.mj_forward(runtime._model, runtime._data)

    assert np.all(runtime._data.ctrl[runtime._actuator_indices] == 0.0)
    assert np.all(runtime._model.actuator_gainprm[runtime._actuator_indices] == 0.0)
    assert np.all(runtime._model.actuator_biasprm[runtime._actuator_indices] == 0.0)
    assert np.all(runtime._data.actuator_force[runtime._actuator_indices] == 0.0)
    assert np.all(runtime._data.qfrc_actuator == 0.0)


def test_invalid_handle_has_no_sampling_or_stop_side_effect_and_valid_stop_is_idempotent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root)
    runtime = MicroduckMujocoRuntime(root, bundle, realtime=False)
    request = _request().model_copy(update={"bundleDigest": bundle.bundleDigest})
    handle = runtime.start(bundle.actions[0], request)
    invalid = type(handle)(taskId="f" * 32)
    initial_time = runtime._data.time

    with pytest.raises(RuntimeError, match="does not own"):
        runtime.sample(invalid)
    assert runtime._data.time == initial_time
    assert runtime._stop_event.is_set() is False
    with pytest.raises(RuntimeError, match="does not own"):
        runtime.safe_stop(invalid, "INVALID")
    assert runtime._stop_event.is_set() is False
    assert runtime.status().activeTaskId == request.taskId
    with pytest.raises(RuntimeError, match="active task"):
        runtime.safe_stop(None, "UNOWNED_STOP")
    assert runtime._stop_event.is_set() is False
    assert runtime.status().activeTaskId == request.taskId

    first = runtime.safe_stop(handle, "CANCELLED")
    second = runtime.safe_stop(handle, "CANCELLED")
    assert second == first


@pytest.mark.parametrize(
    ("fixture", "message"),
    [
        ({"freejoint_on_child": True}, "owned by trunk_base"),
        ({"gyro_kind": "accelerometer"}, "gyro sensor"),
        ({"imu_site_quat": "0.9238795 0 0 0.3826834"}, "identity-aligned"),
    ],
)
def test_runtime_rejects_wrong_root_ownership_or_imu_frame(
    tmp_path: Path, fixture: dict[str, object], message: str
) -> None:
    """BIPED_POSE frames are invalid if the named root or gyro belongs to another frame."""
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root, **fixture)

    with pytest.raises(ValueError, match=message):
        MicroduckMujocoRuntime(root, bundle, realtime=False)


def test_runtime_rejects_commands_outside_code_owned_bounds(
    tmp_path: Path,
) -> None:
    """A runtime clamp cannot substitute for rejecting intent outside the safe API."""
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root)
    runtime = MicroduckMujocoRuntime(root, bundle, realtime=False)
    request = _request().model_copy(update={"bundleDigest": bundle.bundleDigest})
    handle = runtime.start(bundle.actions[0], request)

    with pytest.raises(ValueError, match="code-owned action bounds"):
        runtime.command(handle, {"vxMps": 1.0, "vyMps": -0.5, "yawRateRadps": 2.0})


@pytest.mark.parametrize(
    ("parameters", "lease_ms"),
    [
        ({"vxMps": 0.400001, "vyMps": 0.0, "yawRateRadps": 0.0}, 500),
        ({"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0}, 5_001),
    ],
)
def test_runtime_enforces_code_owned_command_and_lease_bounds_after_manifest_widening(
    tmp_path: Path, parameters: dict[str, float], lease_ms: int
) -> None:
    """Runtime validation must remain safe even if an in-memory manifest is widened."""
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root)
    runtime = MicroduckMujocoRuntime(root, bundle, realtime=False)
    walk = bundle.actions[0]
    assert walk.lease is not None
    widened = walk.model_copy(
        update={
            "parameterSchema": {
                **walk.parameterSchema,
                "properties": {
                    **walk.parameterSchema["properties"],
                    "vxMps": {"type": "number", "minimum": -1_000, "maximum": 1_000},
                },
            },
            "lease": walk.lease.model_copy(update={"maxLeaseMs": 1_000_000}),
        }
    )
    runtime._bundle = bundle.model_copy(
        update={"actions": [widened, *bundle.actions[1:]]}
    )
    request = TaskCreateRequest(
        schema="MICRODUCK_SIM_TASK_V1",
        taskId="9" * 32,
        actionCode="WALK_VELOCITY",
        bundleVersion=bundle.bundleVersion,
        bundleDigest=bundle.bundleDigest,
        parameters=parameters,
        scenario={"terrain": "flat", "seed": 7},
        leaseMs=lease_ms,
        requestedBy="runtime-boundary-test",
    )

    with pytest.raises(ValueError, match="code-owned"):
        runtime.validate(widened, request)


def test_stand_uses_trained_sitting_reset_fixed_goal_and_settled_completion(
    tmp_path: Path,
) -> None:
    """STAND must run the SitStand family from its real sitting reset to settled success."""
    root = tmp_path / "bundle"
    source = _write_verified_bundle(
        root,
        policy_output=np.zeros(14, dtype=np.float32),
        action_code="STAND",
        task_id="Mjlab-SitStand-Flat-MicroDuck",
    )
    bundle = _rewrite_as_stand_bundle(root, source)
    runtime = MicroduckMujocoRuntime(root, bundle, realtime=False)
    request = TaskCreateRequest(
        schema="MICRODUCK_SIM_TASK_V1",
        taskId="3" * 32,
        actionCode="STAND",
        bundleVersion=bundle.bundleVersion,
        bundleDigest=bundle.bundleDigest,
        parameters={},
        scenario={"terrain": "flat", "seed": 7},
        requestedBy="stand-runtime-test",
    )

    stand_action = next(
        action for action in bundle.actions if action.actionCode == "STAND"
    )
    runtime.validate(stand_action, request)
    handle = runtime.start(stand_action, request)
    assert runtime._data.qpos[runtime._joint_qpos_indices[3]] == pytest.approx(
        1.35, abs=0.01
    )
    sample = runtime.sample(handle)
    for _ in range(299):
        if not sample.running:
            break
        sample = runtime.sample(handle)

    evidence = runtime.safe_stop(handle, "TASK_COMPLETE")
    assert sample.terminalState == "SUCCEEDED"
    assert sample.stopReason == "STAND_POSE_SETTLED"
    assert evidence.metrics["resetProfile"] == "TRAINED_SITTING"
    assert evidence.metrics["standPoseError"] <= 0.08
    assert evidence.metrics["trackingErrorSamples"] > 0
    assert evidence.metrics["standSettledSteps"] == 10
    assert evidence.metrics["settledPoseErrorMax"] <= 0.08
    assert 0.09 <= evidence.metrics["settledHeightMinM"]
    assert evidence.metrics["settledHeightMaxM"] <= 0.14
    assert evidence.metrics["settledTiltMaxRad"] <= 0.262
    assert evidence.metrics["settledJointSpeedMaxRadps"] <= 0.5


def test_configured_app_rejects_candidate_and_composes_promoted_runtime(
    tmp_path: Path,
) -> None:
    """Hash-valid candidate bytes must never become executable before qualification."""
    candidate_root = tmp_path / "candidate"
    candidate = _write_verified_bundle(candidate_root)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    candidate_app = create_configured_app(
        {
            "MICRODUCK_ROM_BUNDLE_DIR": str(candidate_root),
            "MICRODUCK_ROM_STATE_DB": str(state_dir / "tasks.sqlite"),
            "MICRODUCK_ROM_BEARER_TOKEN": "secret-token",
            "MICRODUCK_ROM_HOST": "127.0.0.1",
            "MICRODUCK_ROM_PORT": "8000",
        }
    )
    assert candidate_app.state.readiness_reason_codes == ["QUALIFICATION_UNAVAILABLE"]

    configuration = ReleaseConfiguration(
        release="1.0.1",
        createdAt=datetime(2026, 8, 29, 12, 0, tzinfo=UTC),
        actions=(
            ActionQualificationConfig(
                actionCode="WALK_VELOCITY",
                mandatory=True,
                terrain="flat",
                resetProfile="DEFAULT_STANDING",
                seeds=(7, 11, 29),
                maxSteps=100,
                parameters={"vxMps": 0.1, "vyMps": 0.0, "yawRateRadps": 0.0},
                thresholds=QualificationThresholds(
                    minSuccessRate=1.0,
                    maxFallRate=0.0,
                    maxMeanTrackingError=10.0,
                    minMeanDistanceM=0.0,
                    maxMeanEnergyProxy=10_000.0,
                    maxActuatorClampSteps=100,
                    maxPhysicalJointLimitViolations=0,
                    actionMetric="trackingError",
                    actionMetricOperator="lte",
                    actionMetricThreshold=10.0,
                ),
            ),
        ),
    )
    promoted_zip = tmp_path / "qualified.zip"
    promoted = qualify_and_promote(
        candidate_root,
        promoted_zip,
        configuration,
        timestamp=lambda: datetime(2026, 8, 29, 12, 0, tzinfo=UTC),
    )
    qualified_root = tmp_path / "qualified"
    with zipfile.ZipFile(promoted_zip) as archive:
        archive.extractall(qualified_root)

    state_db = state_dir / "tasks.sqlite"
    interrupted = _request().model_copy(
        update={
            "bundleVersion": promoted.manifest.bundleVersion,
            "bundleDigest": promoted.manifest.bundleDigest,
        }
    )
    store = SqliteTaskStore(state_db)
    store.create(interrupted, sha256_prefixed(interrupted))
    store.transition(interrupted.taskId, "VALIDATING", event_type="TASK_VALIDATING")
    store.transition(interrupted.taskId, "RUNNING", event_type="TASK_STARTED")
    bearer_file = tmp_path / "rom-bearer"
    bearer_file.write_bytes(b"secret-token\n")
    bearer_file.chmod(0o400)
    ready_app = create_configured_app(
        {
            "MICRODUCK_ROM_BUNDLE_DIR": str(qualified_root),
            "MICRODUCK_ROM_STATE_DB": str(state_db),
            "MICRODUCK_ROM_BEARER_TOKEN_FILE": str(bearer_file),
            "MICRODUCK_ROM_HOST": "127.0.0.1",
            "MICRODUCK_ROM_PORT": "8000",
        }
    )

    response = TestClient(ready_app).get(
        "/v1/ready", headers={"Authorization": "Bearer secret-token"}
    )
    assert response.status_code == 200
    assert response.json() == {
        "ready": True,
        "reasonCodes": [],
        "robotModel": "MICRODUCK",
        "bundleId": candidate.bundleId,
        "bundleVersion": promoted.manifest.bundleVersion,
        "bundleDigest": promoted.manifest.bundleDigest,
    }
    assert SqliteTaskStore(state_db).get(interrupted.taskId).state == "UNKNOWN"


def test_qualified_stand_api_runs_from_accepted_to_succeeded(
    tmp_path: Path,
) -> None:
    """The concrete API must expose genuine governed discrete completion, not a fake runtime."""
    candidate_root = tmp_path / "candidate"
    candidate = _rewrite_as_stand_bundle(
        candidate_root,
        _write_verified_bundle(
            candidate_root,
            policy_output=np.zeros(14, dtype=np.float32),
            action_code="STAND",
            task_id="Mjlab-SitStand-Flat-MicroDuck",
        ),
    )
    configuration = ReleaseConfiguration(
        release="1.0.1",
        createdAt=datetime(2026, 8, 29, 12, 0, tzinfo=UTC),
        actions=(
            ActionQualificationConfig(
                actionCode="STAND",
                mandatory=True,
                terrain="flat",
                resetProfile="TRAINED_SITTING",
                seeds=(7, 11, 29),
                maxSteps=100,
                parameters={},
                thresholds=QualificationThresholds(
                    minSuccessRate=1.0,
                    maxFallRate=0.0,
                    maxMeanTrackingError=10.0,
                    minMeanDistanceM=0.0,
                    maxMeanEnergyProxy=10_000.0,
                    maxActuatorClampSteps=100,
                    maxPhysicalJointLimitViolations=0,
                    actionMetric="standPoseError",
                    actionMetricOperator="lte",
                    actionMetricThreshold=0.08,
                ),
            ),
        ),
    )
    promoted_zip = tmp_path / "stand-qualified.zip"
    promoted = qualify_and_promote(
        candidate_root,
        promoted_zip,
        configuration,
        timestamp=lambda: datetime(2026, 8, 29, 12, 0, tzinfo=UTC),
    )
    installed = tmp_path / "installed"
    with zipfile.ZipFile(promoted_zip) as archive:
        archive.extractall(installed)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    app = create_configured_app(
        {
            "MICRODUCK_ROM_BUNDLE_DIR": str(installed),
            "MICRODUCK_ROM_STATE_DB": str(state_dir / "tasks.sqlite"),
            "MICRODUCK_ROM_BEARER_TOKEN": "secret-token",
        }
    )
    headers = {"Authorization": "Bearer secret-token"}
    with TestClient(app) as client:
        created = client.post(
            "/v1/tasks",
            headers=headers,
            json={
                "schema": "MICRODUCK_SIM_TASK_V1",
                "taskId": "4" * 32,
                "actionCode": "STAND",
                "bundleVersion": promoted.manifest.bundleVersion,
                "bundleDigest": promoted.manifest.bundleDigest,
                "parameters": {},
                "scenario": {"terrain": "flat", "seed": 7},
                "requestedBy": "stand-api-test",
            },
        )
        assert created.status_code == 202
        assert created.json()["state"] == "ACCEPTED"
        deadline = time.monotonic() + 5.0
        observed = []
        while time.monotonic() < deadline:
            snapshot = client.get("/v1/tasks/" + "4" * 32, headers=headers).json()
            observed.append(snapshot["state"])
            if snapshot["state"] == "SUCCEEDED":
                break
            time.sleep(0.02)

    assert "RUNNING" in observed
    assert snapshot["state"] == "SUCCEEDED"
    assert snapshot["stopReason"] == "STAND_POSE_SETTLED"
    assert snapshot["evidence"]["metrics"]["standSettledSteps"] >= 10
    assert candidate.bundleId == promoted.manifest.bundleId
