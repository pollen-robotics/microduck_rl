from pathlib import Path

import mujoco
import numpy as np
import pytest

from mjlab_microduck.blender_motion import MotionValidationError, validate_motion
from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_XML


def _valid_archive(path: Path) -> Path:
    model = mujoco.MjModel.from_xml_path(str(MICRODUCK_WALK_XML))
    data = mujoco.MjData(model)
    data.qpos[2] = 0.3
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(1, model.njnt)
    ]
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
        for index in range(1, model.nbody)
    ]
    zeros_joint = np.zeros((1, len(joint_names)), dtype=np.float32)
    zeros_body = np.zeros((1, len(body_names), 3), dtype=np.float32)
    np.savez_compressed(
        path,
        joint_pos=zeros_joint,
        joint_vel=zeros_joint,
        body_pos_w=data.xpos[1:][None].astype(np.float32),
        body_quat_w=data.xquat[1:][None].astype(np.float32),
        body_lin_vel_w=zeros_body,
        body_ang_vel_w=zeros_body,
        fps=np.array([50], dtype=np.int32),
        schema_version=np.array([1], dtype=np.int32),
        joint_names=np.asarray(joint_names),
        body_names=np.asarray(body_names),
        source_hashes_json=np.asarray(["{}"]),
    )
    return path


def test_accepts_native_archive_and_replays_kinematics(tmp_path):
    result = validate_motion(_valid_archive(tmp_path / "valid.npz"))
    assert result.frames == 1
    assert result.max_position_error_m < 1e-6
    assert result.max_orientation_error_rad < 1e-6


def test_rejects_body_pose_that_disagrees_with_joint_replay(tmp_path):
    source = _valid_archive(tmp_path / "valid.npz")
    with np.load(source) as loaded:
        archive = {key: loaded[key] for key in loaded.files}
    archive["body_pos_w"] = archive["body_pos_w"].copy()
    archive["body_pos_w"][0, -1, 0] += 0.01
    invalid = tmp_path / "invalid.npz"
    np.savez_compressed(invalid, **archive)
    with pytest.raises(MotionValidationError, match="kinematic replay"):
        validate_motion(invalid)


def test_rejects_wrong_joint_order(tmp_path):
    source = _valid_archive(tmp_path / "valid.npz")
    with np.load(source) as loaded:
        archive = {key: loaded[key] for key in loaded.files}
    archive["joint_names"] = archive["joint_names"][::-1]
    invalid = tmp_path / "invalid-order.npz"
    np.savez_compressed(invalid, **archive)
    with pytest.raises(MotionValidationError, match="joint_names"):
        validate_motion(invalid)


def test_rejects_nonroot_zero_quaternion(tmp_path):
    source = _valid_archive(tmp_path / "valid.npz")
    with np.load(source) as loaded:
        archive = {key: loaded[key] for key in loaded.files}
    archive["body_quat_w"] = archive["body_quat_w"].copy()
    archive["body_quat_w"][0, -1] = 0.0
    invalid = tmp_path / "zero-quat.npz"
    np.savez_compressed(invalid, **archive)
    with pytest.raises(MotionValidationError, match="zero quaternion"):
        validate_motion(invalid)


def test_rejects_extra_keys_and_invalid_source_hash_metadata(tmp_path):
    source = _valid_archive(tmp_path / "valid.npz")
    with np.load(source) as loaded:
        archive = {key: loaded[key] for key in loaded.files}
    archive["surprise"] = np.array([1])
    extra = tmp_path / "extra.npz"
    np.savez_compressed(extra, **archive)
    with pytest.raises(MotionValidationError, match="unexpected keys"):
        validate_motion(extra)
    archive.pop("surprise")
    archive["source_hashes_json"] = np.asarray(["not-json"])
    malformed = tmp_path / "bad-hashes.npz"
    np.savez_compressed(malformed, **archive)
    with pytest.raises(MotionValidationError, match="source_hashes_json"):
        validate_motion(malformed)
