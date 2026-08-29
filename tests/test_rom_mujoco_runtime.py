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
    metadata_overrides: dict[str, str] | None = None,
) -> None:
    output_values = (
        np.linspace(-0.13, 0.13, 14, dtype=np.float32)
        if output is None
        else np.asarray(output, dtype=np.float32)
    )
    observations = helper.make_tensor_value_info(
        "observations", TensorProto.FLOAT, [1, input_dimension]
    )
    actions = helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 14])
    weights = helper.make_tensor(
        "weights",
        TensorProto.FLOAT,
        [input_dimension, 14],
        np.zeros((input_dimension, 14), dtype=np.float32).ravel(),
    )
    bias = helper.make_tensor("bias", TensorProto.FLOAT, [14], output_values.ravel())
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["observations", "weights"], ["linear"]),
            helper.make_node("Add", ["linear", "bias"], ["actions"]),
        ],
        "fixture-policy",
        [observations],
        [actions],
        [weights, bias],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
    )
    metadata = {
        "microduck.task_id": TASK_ID,
        "microduck.source_commit": SOURCE_COMMIT,
        "microduck.observation_contract": OBSERVATION_CONTRACT,
        "microduck.action_contract": ACTION_CONTRACT,
        "microduck.checkpoint": "model_100.pt",
        "microduck.run_identity": "entity/project/run-id",
    }
    metadata.update(metadata_overrides or {})
    for key, value in sorted(metadata.items()):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, path)


def _write_model(path: Path) -> None:
    bodies: list[str] = []
    closing: list[str] = []
    for index, joint in enumerate(CONTROLLED_SERVO_JOINTS):
        bodies.extend(
            [
                f'<body name="link_{index}" pos="0 0 0.005">',
                f'<joint name="{joint}" type="hinge" axis="0 0 1" range="-2 2" armature="0.01" damping="0.1"/>',
                '<geom type="sphere" size="0.002" mass="0.01"/>',
            ]
        )
        closing.append("</body>")
    actuators = "\n".join(
        f'<position name="servo_{joint}" joint="{joint}" kp="1" ctrlrange="-2 2"/>'
        for joint in reversed(CONTROLLED_SERVO_JOINTS)
    )
    path.write_text(
        f"""
<mujoco model="microduck-runtime-fixture">
  <compiler angle="radian"/>
  <option timestep="0.005" gravity="0 0 0"/>
  <worldbody>
    <body name="trunk_base" pos="0 0 0.12">
      <freejoint name="trunk_base_freejoint"/>
      <geom type="sphere" size="0.01" mass="0.1"/>
      <site name="imu"/>
      <body name="roller"><joint name="passive_wheel" type="hinge"/><geom type="sphere" size="0.002" mass="0.001"/></body>
      {"".join(bodies)}
      {"".join(reversed(closing))}
    </body>
  </worldbody>
  <actuator>{actuators}</actuator>
  <sensor><gyro name="imu_ang_vel" site="imu"/></sensor>
</mujoco>
""".strip()
    )


def _write_verified_bundle(
    root: Path,
    *,
    policy_output: np.ndarray | None = None,
    input_dimension: int = 61,
    metadata_overrides: dict[str, str] | None = None,
    runtime_requirements: dict[str, str] | None = None,
) -> PolicyBundle:
    model_path = root / "models" / "robot.xml"
    policy_path = root / "policies" / "walk.onnx"
    model_path.parent.mkdir(parents=True)
    policy_path.parent.mkdir(parents=True)
    _write_model(model_path)
    _write_policy(
        policy_path,
        output=policy_output,
        input_dimension=input_dimension,
        metadata_overrides=metadata_overrides,
    )
    policy = PolicyArtifact(
        policyRef="walk-policy",
        path="policies/walk.onnx",
        digest=_digest(policy_path),
        taskId=TASK_ID,
        checkpoint="model_100.pt",
        experimentRef="entity/project/run-id",
        runtimeRequirements=runtime_requirements
        or {
            "observationContract": OBSERVATION_CONTRACT,
            "actionContract": ACTION_CONTRACT,
            "normalization": "BAKED_IN_ONNX",
        },
    )
    action = ActionDefinition(
        actionCode="WALK_VELOCITY",
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
        preconditions={"allowedTerrains": ["flat"]},
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
        qualification={"artifacts": [], "modelClosure": []},
        license={"artifacts": []},
    )
    artifact_digests = {
        unsigned.model.path: unsigned.model.digest,
        policy.path: policy.digest,
    }
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
    assert status.health == {"ready": True, "healthy": True, "reasonCodes": []}

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
        "seed": 7,
        "sourceCommit": SOURCE_COMMIT,
        "steps": 1,
        "terrain": "flat",
    }
    assert runtime.status().limp is False
    assert runtime.status().activeTaskId is None


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


@pytest.mark.parametrize(
    ("fixture_options", "message"),
    [
        ({"input_dimension": 60}, "input"),
        (
            {"metadata_overrides": {"microduck.action_contract": "WRONG"}},
            "metadata",
        ),
        ({"policy_output": np.full(14, np.nan, dtype=np.float32)}, "finite"),
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


def test_discrete_completion_uses_internal_action_mapping_and_pose_gate(
    tmp_path: Path,
) -> None:
    """A manifest Python name or a 100 ms timer must not declare a maneuver complete."""
    root = tmp_path / "bundle"
    bundle = _rewrite_as_stand_bundle(root, _write_verified_bundle(root))
    runtime = MicroduckMujocoRuntime(root, bundle, realtime=False)
    request = TaskCreateRequest(
        schema="MICRODUCK_SIM_TASK_V1",
        taskId="2" * 32,
        actionCode="STAND",
        bundleVersion=bundle.bundleVersion,
        bundleDigest=bundle.bundleDigest,
        parameters={},
        scenario={"terrain": "flat", "seed": 11},
        requestedBy="test",
    )
    handle = runtime.start(bundle.actions[0], request)

    for _ in range(5):
        early = runtime.sample(handle)
    assert early.running is True
    for _ in range(95):
        terminal = runtime.sample(handle)

    assert terminal.running is False
    assert terminal.terminalState == "SUCCEEDED"
    assert terminal.stopReason == "TASK_COMPLETE"


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
