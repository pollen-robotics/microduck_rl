from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import mujoco
import numpy as np
import onnx
import pytest
from fastapi.testclient import TestClient
from onnx import TensorProto, helper

from mjlab_microduck.rom.action_specs import ACTION_RUNTIME_SPECS
from mjlab_microduck.rom.contracts import (
    ACTION_CONTRACT,
    CONTROLLED_SERVO_JOINTS,
    OBSERVATION_CONTRACT,
    ActionContract,
    ActionDefinition,
    CompletionContract,
    LeaseContract,
    ModelArtifact,
    ObservationContract,
    PolicyArtifact,
    PolicyBundle,
    TaskCreateRequest,
    sha256_prefixed,
)
from mjlab_microduck.rom.main import create_configured_app, load_verified_bundle
from mjlab_microduck.rom.mujoco_runtime import MicroduckMujocoRuntime
from mjlab_microduck.rom.observation import DEFAULT_JOINT_POSE
from mjlab_microduck.rom.onnx_policy import inspect_normalized_actor
from mjlab_microduck.rom.service import SimulatorTaskService
from mjlab_microduck.rom.store import SqliteTaskStore

SOURCE_COMMIT = "a" * 40
TASK_ID = "Mjlab-Velocity-Flat-MicroDuck"


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
    normalizer_input = "prefixed" if identity_before_normalizer else "observations"
    nodes = [
        *(
            [helper.make_node("Identity", ["observations"], ["prefixed"])]
            if identity_before_normalizer
            else []
        ),
        helper.make_node("Sub", [normalizer_input, "normalizer_mean"], ["centered"]),
        helper.make_node("Div", ["centered", "normalizer_std"], ["normalized"]),
        helper.make_node(
            "MatMul",
            ["observations" if bypass_normalizer else "normalized", "weights"],
            ["linear"],
        ),
        helper.make_node("Add", ["linear", "bias"], ["actions"]),
    ]
    graph = helper.make_graph(
        nodes,
        "fixture-policy",
        [observations],
        [actions],
        [normalizer_mean, normalizer_std, weights, bias],
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
) -> None:
    bodies: list[str] = []
    closing: list[str] = []
    for index, joint in enumerate(CONTROLLED_SERVO_JOINTS):
        bodies.extend(
            [
                f'<body name="link_{index}" pos="0 0 0.005">',
                f'<joint name="{joint}" type="hinge" axis="0 0 1" range="-2 2" armature="0.01" damping="0.1"/>',
                *(
                    [
                        f'<joint name="passive_{joint}_backlash" type="hinge" axis="0 0 1" range="-0.1 0.1"/>'
                    ]
                    if backlash
                    else []
                ),
                '<geom type="sphere" size="0.002" mass="0.01"/>',
                *(
                    [
                        f'<body name="wheel_{wheel}"><joint name="passive_{wheel}_wheel" type="hinge" axis="0 1 0"/>'
                        '<geom type="cylinder" size="0.003 0.002" mass="0.001"/></body>'
                        for wheel in (
                            ("LF", "LR") if joint == "left_ankle" else ("RF", "RR")
                        )
                    ]
                    if roller_topology and joint in {"left_ankle", "right_ankle"}
                    else []
                ),
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
  <option timestep="0.005" gravity="0 0 0"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05"/>
    <body name="trunk_base" pos="0 0 0.12">
      {"" if freejoint_on_child else '<freejoint name="trunk_base_freejoint"/>'}
      <geom type="sphere" size="0.01" mass="0.1"/>
      <site name="imu" quat="{imu_site_quat}"/>
      <body name="roller"><joint name="passive_wheel" type="hinge"/><geom type="sphere" size="0.002" mass="0.001"/></body>
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
) -> PolicyBundle:
    model_path = root / "models" / "robot.xml"
    policy_path = root / "policies" / "walk.onnx"
    model_path.parent.mkdir(parents=True)
    policy_path.parent.mkdir(parents=True)
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
    action = ActionDefinition(
        actionCode=action_code,
        executionMode="CONTINUOUS_LEASE",
        availability="AVAILABLE",
        policyRef=policy.policyRef,
        parameterSchema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "vxMps": {"type": "number", "minimum": -0.4, "maximum": 0.4},
                "vyMps": {"type": "number", "minimum": -0.3, "maximum": 0.3},
                "yawRateRadps": {"type": "number", "minimum": -1.0, "maximum": 1.0},
            },
            "required": ["vxMps", "vyMps", "yawRateRadps"],
        },
        lease=LeaseContract(
            minLeaseMs=100,
            defaultLeaseMs=500,
            maxLeaseMs=5_000,
            commandCadenceMs=50,
            safeStopBehavior="ZERO_TWIST",
        ),
        preconditions={
            "allowedTerrains": ["flat"],
            "scenarioProfile": "SEEDED_SERVO_RESET_V1",
        },
    )
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
    unsigned = PolicyBundle(
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
        actions=[action],
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
        license={"artifacts": []},
    )
    artifact_digests = {
        unsigned.model.path: unsigned.model.digest,
        policy.path: policy.digest,
    }
    if include_dependency and declare_dependency:
        artifact_digests["models/extra.xml"] = _digest(dependency_path)
    bundle = unsigned.model_copy(
        update={
            "bundleDigest": sha256_prefixed(
                {
                    "manifest": unsigned.model_dump(
                        mode="json", by_alias=True, exclude={"bundleDigest"}
                    ),
                    "artifacts": artifact_digests,
                }
            )
        }
    )
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
    action = ActionDefinition(
        actionCode="STAND",
        executionMode="DISCRETE",
        availability="AVAILABLE",
        policyRef=policy.policyRef,
        parameterSchema={
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        },
        completion=CompletionContract(
            terminalConditions=["TASK_COMPLETE", "FALLEN", "TIMEOUT"],
            maxDurationMs=15_000,
        ),
        preconditions={"allowedTerrains": ["flat"]},
    )
    unsigned = source.model_copy(
        update={"bundleDigest": None, "policies": [policy], "actions": [action]}
    )
    digests = {
        source.model.path: source.model.digest,
        policy.path: policy.digest,
    }
    rewritten = unsigned.model_copy(
        update={
            "bundleDigest": sha256_prefixed(
                {
                    "manifest": unsigned.model_dump(
                        mode="json", by_alias=True, exclude={"bundleDigest"}
                    ),
                    "artifacts": digests,
                }
            )
        }
    )
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


def test_runtime_readiness_rejects_available_action_with_wrong_policy_task_identity(
    tmp_path: Path,
) -> None:
    """Deferring task-family validation until POST would create false catalog availability."""
    root = tmp_path / "bundle"
    source = _write_verified_bundle(root)
    action = source.actions[0].model_copy(update={"actionCode": "VELSTAND_VELOCITY"})
    unsigned = source.model_copy(update={"bundleDigest": None, "actions": [action]})
    artifact_digests = {
        source.model.path: source.model.digest,
        source.policies[0].path: source.policies[0].digest,
    }
    rewritten = unsigned.model_copy(
        update={
            "bundleDigest": sha256_prefixed(
                {
                    "manifest": unsigned.model_dump(
                        mode="json", by_alias=True, exclude={"bundleDigest"}
                    ),
                    "artifacts": artifact_digests,
                }
            )
        }
    )
    (root / "microduck-policy-bundle.json").write_text(
        rewritten.model_dump_json(by_alias=True, exclude_none=True)
    )
    verified = load_verified_bundle(root)

    with pytest.raises(ValueError, match="policy task identity"):
        MicroduckMujocoRuntime(root, verified, realtime=False)


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
    unsigned = source.model_copy(update={"bundleDigest": None, "actions": [action]})
    artifacts = {
        source.model.path: source.model.digest,
        source.policies[0].path: source.policies[0].digest,
    }
    rewritten = unsigned.model_copy(
        update={
            "bundleDigest": sha256_prefixed(
                {
                    "manifest": unsigned.model_dump(
                        mode="json", by_alias=True, exclude={"bundleDigest"}
                    ),
                    "artifacts": artifacts,
                }
            )
        }
    )
    (root / "microduck-policy-bundle.json").write_text(
        rewritten.model_dump_json(by_alias=True, exclude_none=True)
    )

    with pytest.raises(ValueError, match="preconditions do not match qualification"):
        MicroduckMujocoRuntime(root, load_verified_bundle(root), realtime=False)


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
        "modelIdentity": bundle.model.digest,
        "sourceCommit": SOURCE_COMMIT,
        "steps": 1,
        "loopOverruns": 0,
        "trackingError": pytest.approx(0.355429, abs=1e-6),
    }
    assert runtime.status().limp is False
    assert runtime.status().activeTaskId is None


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


def test_runtime_rejects_requested_terrain_not_bound_to_loaded_model(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root)
    runtime = MicroduckMujocoRuntime(root, bundle, realtime=False)
    request = _request().model_copy(
        update={
            "bundleDigest": bundle.bundleDigest,
            "scenario": {"terrain": "slope", "seed": 7},
        }
    )

    with pytest.raises(ValueError, match="qualified loaded model"):
        runtime.validate(bundle.actions[0], request)


def test_runtime_rejects_unknown_scenario_fields(tmp_path: Path) -> None:
    """Manifest-like free-form scenario data must not silently alter deployment semantics."""
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root)
    runtime = MicroduckMujocoRuntime(root, bundle, realtime=False)
    request = _request().model_copy(
        update={
            "bundleDigest": bundle.bundleDigest,
            "scenario": {"terrain": "flat", "seed": 7, "friction": 0.01},
        }
    )

    with pytest.raises(ValueError, match="exact terrain and seed"):
        runtime.validate(bundle.actions[0], request)


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
        update={"taskId": "3" * 32, "scenario": {"terrain": "flat", "seed": 8}}
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
    handle = runtime.start(bundle.actions[0], request)

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


def test_runtime_reports_requested_and_safety_limited_command_separately(
    tmp_path: Path,
) -> None:
    """Overwriting requested intent with a clamp would hide why motion was limited."""
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root)
    runtime = MicroduckMujocoRuntime(root, bundle, realtime=False)
    request = _request().model_copy(update={"bundleDigest": bundle.bundleDigest})
    handle = runtime.start(bundle.actions[0], request)

    runtime.command(handle, {"vxMps": 1.0, "vyMps": -0.5, "yawRateRadps": 2.0})
    status = runtime.status()

    assert status.requestedMotion["twist"] == [1.0, -0.5, 2.0]
    assert status.appliedMotion["twist"] == [0.4, -0.3, 1.0]
    assert status.limitingReason == "COMMAND_LIMIT"


def test_discrete_action_without_exact_runtime_semantics_is_refused(
    tmp_path: Path,
) -> None:
    """A manifest Python name must never turn an unsupported maneuver into success."""
    root = tmp_path / "bundle"
    bundle = _rewrite_as_stand_bundle(root, _write_verified_bundle(root))
    with pytest.raises(ValueError, match="no runtime semantics"):
        MicroduckMujocoRuntime(root, bundle, realtime=False)


def test_configured_app_composes_verified_ready_concrete_runtime(
    tmp_path: Path,
) -> None:
    """Leaving the Task-6 placeholder selected would keep valid installations unready."""
    root = tmp_path / "bundle"
    bundle = _write_verified_bundle(root)
    app = create_configured_app(
        {
            "MICRODUCK_ROM_BUNDLE_DIR": str(root),
            "MICRODUCK_ROM_STATE_DB": str(tmp_path / "state" / "tasks.sqlite"),
            "MICRODUCK_ROM_BEARER_TOKEN": "secret-token",
            "MICRODUCK_ROM_HOST": "127.0.0.1",
            "MICRODUCK_ROM_PORT": "8000",
        }
    )
    (tmp_path / "state").mkdir(exist_ok=True)

    # Recompose after the state directory exists; the first pass must remain fail-closed.
    assert app.state.readiness_reason_codes == ["STATE_DB_UNAVAILABLE"]
    state_db = tmp_path / "state" / "tasks.sqlite"
    interrupted = _request().model_copy(update={"bundleDigest": bundle.bundleDigest})
    store = SqliteTaskStore(state_db)
    store.create(interrupted, sha256_prefixed(interrupted))
    store.transition(interrupted.taskId, "VALIDATING", event_type="TASK_VALIDATING")
    store.transition(interrupted.taskId, "RUNNING", event_type="TASK_STARTED")
    ready_app = create_configured_app(
        {
            "MICRODUCK_ROM_BUNDLE_DIR": str(root),
            "MICRODUCK_ROM_STATE_DB": str(state_db),
            "MICRODUCK_ROM_BEARER_TOKEN": "secret-token",
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
        "bundleId": bundle.bundleId,
        "bundleVersion": bundle.bundleVersion,
        "bundleDigest": bundle.bundleDigest,
    }
    assert SqliteTaskStore(state_db).get(interrupted.taskId).state == "UNKNOWN"
