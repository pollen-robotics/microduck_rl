import hashlib
import json
from itertools import count
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
import pytest

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
    ) -> Path:
        policy_path = tmp_path / f"constant-policy-{next(policy_count)}.onnx"
        input_info = helper.make_tensor_value_info(
            "obs", TensorProto.FLOAT, [1, input_width]
        )
        output_info = helper.make_tensor_value_info(
            "actions", TensorProto.FLOAT, [1, output_width]
        )
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
