import hashlib
import importlib.util
import json
from itertools import count
from pathlib import Path

import mujoco
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from mjlab_microduck import policy_rollout as rollout_module
from mjlab_microduck.blender_motion import validate_motion
from mjlab_microduck.policy_rollout import (
    PolicyRolloutConfig,
    PolicyRolloutError,
    export_policy_rollout,
)

EXPECTED_JOINT_NAMES = (
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
INFER_POLICY_PATH = Path(__file__).parents[1] / "scripts" / "infer_policy.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def make_policy(tmp_path):
    policy_count = count()

    def make_policy(
        *,
        input_width: int = 61,
        output_width: int = 14,
        joint_names: tuple[str, ...] = EXPECTED_JOINT_NAMES,
        state_sensitive: bool = False,
    ) -> Path:
        kind = "state-sensitive" if state_sensitive else "constant"
        policy_path = tmp_path / f"{kind}-policy-{next(policy_count)}.onnx"
        input_info = helper.make_tensor_value_info(
            "obs", TensorProto.FLOAT, [1, input_width]
        )
        output_info = helper.make_tensor_value_info(
            "actions", TensorProto.FLOAT, [1, output_width]
        )
        if state_sensitive:
            if input_width != 61 or output_width != 14:
                raise ValueError("state-sensitive fixture requires the canonical contract")
            weights = np.zeros((input_width, output_width), dtype=np.float32)
            bias = np.empty(output_width, dtype=np.float32)
            for joint in range(output_width):
                weights[joint % 3, joint] = 0.003 * (joint + 1)
                weights[3 + (joint + 1) % 3, joint] = 0.004 * (-1) ** joint
                weights[6 + joint, joint] = 0.12
                weights[20 + joint, joint] = 0.003
                weights[34 + joint, joint] = 0.35
                weights[48 + joint % 13, joint] = 0.02
                bias[joint] = 0.005 * (joint % 5 - 2)
            weight_value = helper.make_tensor(
                "weights",
                TensorProto.FLOAT,
                weights.shape,
                weights.reshape(-1),
            )
            bias_value = helper.make_tensor(
                "bias", TensorProto.FLOAT, bias.shape, bias
            )
            nodes = [
                helper.make_node("MatMul", ["obs", "weights"], ["weighted"]),
                helper.make_node("Add", ["weighted", "bias"], ["actions"]),
            ]
            graph = helper.make_graph(
                nodes,
                "state-sensitive-policy",
                [input_info],
                [output_info],
                initializer=[weight_value, bias_value],
            )
        else:
            output_value = helper.make_tensor(
                "constant_actions",
                TensorProto.FLOAT,
                [1, output_width],
                [0.0] * output_width,
            )
            graph = helper.make_graph(
                [helper.make_node("Constant", [], ["actions"], value=output_value)],
                "constant-policy",
                [input_info],
                [output_info],
            )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 18)],
        )
        metadata = model.metadata_props.add()
        metadata.key = "joint_names"
        metadata.value = ",".join(joint_names)
        onnx.save(model, policy_path)
        return policy_path

    return make_policy


@pytest.fixture
def policy_path(make_policy):
    return make_policy()


def _infer_policy_module():
    spec = importlib.util.spec_from_file_location("infer_policy_reference", INFER_POLICY_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _initialize_reference_state(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    root_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
    )
    root_qpos_address = int(model.jnt_qposadr[root_joint_id])
    data.qpos[root_qpos_address : root_qpos_address + 7] = [
        0.0,
        0.0,
        0.125,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    data.qpos[model.jnt_qposadr[1:]] = rollout_module.DEFAULT_POSE
    data.ctrl[:] = rollout_module.DEFAULT_POSE


def _reference_rollout(
    policy_path: Path, frames: int, command: tuple[float, float, float]
):
    reference = _infer_policy_module()
    model = mujoco.MjModel.from_xml_path(str(rollout_module.MICRODUCK_SCENE_XML))
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    _initialize_reference_state(model, data)
    policy = reference.PolicyInference(
        model,
        data,
        walking_onnx_path=str(policy_path),
        action_scale=1.0,
        use_projected_gravity=True,
        new_cmd_obs=True,
    )
    policy.set_vel_cmd(*command)
    joint_qpos_addresses = model.jnt_qposadr[1:]
    joint_qvel_addresses = model.jnt_dofadr[1:]
    result = {
        "joint_pos": np.empty((frames, 14), dtype=np.float32),
        "joint_vel": np.empty((frames, 14), dtype=np.float32),
        "body_pos_w": np.empty((frames, 15, 3), dtype=np.float32),
        "body_quat_w": np.empty((frames, 15, 4), dtype=np.float32),
    }
    for frame in range(frames):
        mujoco.mj_forward(model, data)
        result["joint_pos"][frame] = data.qpos[joint_qpos_addresses]
        result["joint_vel"][frame] = data.qvel[joint_qvel_addresses]
        result["body_pos_w"][frame] = data.xpos[1:]
        result["body_quat_w"][frame] = data.xquat[1:]
        action = policy.infer()
        policy.apply_action(action)
        for _ in range(4):
            mujoco.mj_step(model, data)
    return result


def test_rejects_duration_without_integral_50hz_frames(tmp_path, policy_path):
    cfg = PolicyRolloutConfig(policy_path, tmp_path / "out.npz", duration_s=0.011)
    with pytest.raises(PolicyRolloutError, match="integral number of 50 Hz frames"):
        export_policy_rollout(cfg)


@pytest.mark.parametrize("input_width,output_width", [(60, 14), (61, 13)])
def test_rejects_incompatible_onnx_contract(
    tmp_path, make_policy, input_width, output_width
):
    policy = make_policy(input_width=input_width, output_width=output_width)
    cfg = PolicyRolloutConfig(policy, tmp_path / "out.npz", duration_s=0.02)
    with pytest.raises(PolicyRolloutError, match=r"\[1,61\].*\[1,14\]"):
        export_policy_rollout(cfg)


def test_rejects_joint_metadata_order_drift(tmp_path, make_policy):
    policy = make_policy(joint_names=tuple(reversed(EXPECTED_JOINT_NAMES)))
    with pytest.raises(PolicyRolloutError, match="joint_names.*index 0"):
        export_policy_rollout(PolicyRolloutConfig(policy, tmp_path / "out.npz", 0.02))


def test_public_config_uses_approved_forward_command_default(tmp_path):
    config = PolicyRolloutConfig(tmp_path / "policy.onnx", tmp_path / "out.npz")

    assert config.command == (0.30, 0.0, 0.0)


@pytest.mark.parametrize("invalid", [("fast", 0.0, 0.0), (object(), 0.0, 0.0)])
def test_nonnumeric_command_is_normalized_to_policy_rollout_error(
    tmp_path, invalid
):
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"not reached")

    with pytest.raises(
        PolicyRolloutError, match="command must contain exactly three finite values"
    ):
        export_policy_rollout(
            PolicyRolloutConfig(policy, tmp_path / "out.npz", command=invalid)
        )


@pytest.mark.parametrize("alias_kind", ["same", "resolved", "symlink"])
def test_rejects_policy_output_collision_before_loading_and_preserves_policy(
    tmp_path, monkeypatch, alias_kind
):
    policy = tmp_path / "policy.npz"
    original = b"original policy bytes"
    policy.write_bytes(original)
    if alias_kind == "same":
        output = policy
    elif alias_kind == "resolved":
        output = policy.resolve()
    else:
        output = tmp_path / "policy-alias.npz"
        output.symlink_to(policy)

    def fail_if_loaded(*_args, **_kwargs):
        pytest.fail("ONNX session creation must not occur for an output collision")

    monkeypatch.setattr(rollout_module.ort, "InferenceSession", fail_if_loaded)
    with pytest.raises(PolicyRolloutError, match="policy_path.*output_path.*different"):
        export_policy_rollout(PolicyRolloutConfig(policy, output))

    assert policy.read_bytes() == original


def test_rejects_non_npz_output_before_loading_policy(tmp_path, monkeypatch):
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"not reached")

    def fail_if_loaded(*_args, **_kwargs):
        pytest.fail("ONNX session creation must not occur for an invalid output suffix")

    monkeypatch.setattr(rollout_module.ort, "InferenceSession", fail_if_loaded)
    with pytest.raises(PolicyRolloutError, match=r"output_path.*\.npz"):
        export_policy_rollout(PolicyRolloutConfig(policy, tmp_path / "motion.onnx"))


def test_exports_valid_three_frame_rollout(tmp_path, policy_path):
    result = export_policy_rollout(
        PolicyRolloutConfig(policy_path, tmp_path / "rollout.npz", duration_s=0.06)
    )

    archive = np.load(result, allow_pickle=False)
    assert archive["joint_pos"].shape == (3, 14)
    assert archive["body_pos_w"].shape == (3, 15, 3)
    assert archive["fps"].tolist() == [50]
    assert tuple(archive["joint_names"]) == EXPECTED_JOINT_NAMES
    assert json.loads(str(archive["source_hashes_json"][0]))["policy_sha256"] == sha256(
        policy_path
    )
    assert validate_motion(result).frames == 3


def test_hashes_canonical_rollout_configuration(tmp_path, policy_path):
    config = PolicyRolloutConfig(
        policy_path,
        tmp_path / "rollout.npz",
        duration_s=0.02,
        command=(0.30, -0.05, 0.20),
        seed=7,
    )
    result = export_policy_rollout(config)
    canonical_json = (
        '{"command":[0.3,-0.05,0.2],"control_decimation":4,'
        '"control_hz":50,"duration_s":0.02,"seed":7,"timestep_s":0.005}'
    )
    expected = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    with np.load(result, allow_pickle=False) as archive:
        source_hashes = json.loads(str(archive["source_hashes_json"][0]))

    assert source_hashes["rollout_config_sha256"] == expected


def test_state_sensitive_rollout_matches_50hz_inference_reference(
    tmp_path, make_policy
):
    policy = make_policy(state_sensitive=True)
    command = (0.30, -0.05, 0.20)
    frames = 4
    result = export_policy_rollout(
        PolicyRolloutConfig(
            policy,
            tmp_path / "state-sensitive.npz",
            duration_s=frames / 50,
            command=command,
        )
    )
    expected = _reference_rollout(policy, frames, command)

    with np.load(result, allow_pickle=False) as archive:
        for field, values in expected.items():
            np.testing.assert_allclose(archive[field], values, rtol=0.0, atol=2e-6)


def test_body_linear_velocity_is_world_velocity_at_each_body_origin(
    tmp_path, policy_path, monkeypatch
):
    real_mj_data = mujoco.MjData
    model = mujoco.MjModel.from_xml_path(str(rollout_module.MICRODUCK_SCENE_XML))
    data = real_mj_data(model)
    _initialize_reference_state(model, data)
    initial_qvel = np.linspace(-0.3, 0.4, model.nv)
    data.qvel[:] = initial_qvel
    mujoco.mj_forward(model, data)
    expected = np.empty((model.nbody - 1, 3), dtype=np.float64)
    for body_id in range(1, model.nbody):
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            velocity,
            0,
        )
        expected[body_id - 1] = velocity[3:]
    assert np.max(np.abs(expected - data.cvel[1:, 3:])) > 1e-3

    def data_with_initial_velocity(export_model):
        export_data = real_mj_data(export_model)
        export_data.qvel[:] = initial_qvel
        return export_data

    monkeypatch.setattr(rollout_module.mujoco, "MjData", data_with_initial_velocity)
    result = export_policy_rollout(
        PolicyRolloutConfig(policy_path, tmp_path / "velocity.npz", duration_s=0.02)
    )

    with np.load(result, allow_pickle=False) as archive:
        np.testing.assert_allclose(
            archive["body_lin_vel_w"][0], expected, rtol=0.0, atol=2e-6
        )


def test_repeated_state_sensitive_exports_are_byte_identical(tmp_path, make_policy):
    policy = make_policy(state_sensitive=True)
    first = export_policy_rollout(
        PolicyRolloutConfig(policy, tmp_path / "first.npz", duration_s=0.06, seed=13)
    )
    second = export_policy_rollout(
        PolicyRolloutConfig(policy, tmp_path / "second.npz", duration_s=0.06, seed=13)
    )

    assert first.read_bytes() == second.read_bytes()
