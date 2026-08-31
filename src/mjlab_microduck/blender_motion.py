"""Validate Blender motion archives against the canonical Microduck MJCF."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import mujoco
import numpy as np


MICRODUCK_WALK_XML = Path(__file__).parent / "robot/microduck/robot_walk.xml"


REQUIRED_KEYS = {
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "fps",
    "schema_version",
    "joint_names",
    "body_names",
    "source_hashes_json",
}


class MotionValidationError(ValueError):
    """An archive cannot be consumed safely by the Microduck motion pipeline."""


@dataclass(frozen=True)
class MotionValidationResult:
    frames: int
    max_position_error_m: float
    max_orientation_error_rad: float


def _names(model: mujoco.MjModel, object_type, start: int, count: int) -> tuple[str, ...]:
    return tuple(
        mujoco.mj_id2name(model, object_type, index)
        for index in range(start, count)
    )


def _scalar(array: np.ndarray, name: str) -> int:
    flattened = np.asarray(array).reshape(-1)
    if flattened.size != 1:
        raise MotionValidationError(f"{name} must contain exactly one value")
    return int(flattened[0])


def validate_motion(
    path: str | Path,
    *,
    mjcf_path: str | Path = MICRODUCK_WALK_XML,
    position_tolerance_m: float = 1e-4,
    orientation_tolerance_rad: float = 1e-4,
) -> MotionValidationResult:
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    expected_joints = _names(model, mujoco.mjtObj.mjOBJ_JOINT, 1, model.njnt)
    expected_bodies = _names(model, mujoco.mjtObj.mjOBJ_BODY, 1, model.nbody)
    with np.load(Path(path), allow_pickle=False) as archive:
        missing = REQUIRED_KEYS - set(archive.files)
        if missing:
            raise MotionValidationError(f"archive is missing keys: {sorted(missing)}")
        extra = set(archive.files) - REQUIRED_KEYS
        if extra:
            raise MotionValidationError(f"archive has unexpected keys: {sorted(extra)}")
        if _scalar(archive["fps"], "fps") != 50:
            raise MotionValidationError("fps must be 50")
        if _scalar(archive["schema_version"], "schema_version") != 1:
            raise MotionValidationError("schema_version must be 1")
        joint_names = tuple(str(name) for name in archive["joint_names"])
        body_names = tuple(str(name) for name in archive["body_names"])
        if joint_names != expected_joints:
            raise MotionValidationError(
                f"joint_names do not match canonical order: {joint_names!r}"
            )
        if body_names != expected_bodies:
            raise MotionValidationError(
                f"body_names do not match canonical order: {body_names!r}"
            )
        hashes = np.asarray(archive["source_hashes_json"])
        if hashes.shape != (1,) or hashes.dtype.kind not in "US":
            raise MotionValidationError("source_hashes_json must contain one JSON string")
        try:
            hash_payload = json.loads(str(hashes[0]))
        except json.JSONDecodeError as exc:
            raise MotionValidationError("source_hashes_json is not valid JSON") from exc
        if not isinstance(hash_payload, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in hash_payload.items()
        ):
            raise MotionValidationError("source_hashes_json must encode a string map")
        joint_pos = np.asarray(archive["joint_pos"], dtype=np.float64)
        body_pos = np.asarray(archive["body_pos_w"], dtype=np.float64)
        body_quat = np.asarray(archive["body_quat_w"], dtype=np.float64)
        frames = joint_pos.shape[0] if joint_pos.ndim == 2 else 0
        expected_shapes = {
            "joint_pos": (frames, len(expected_joints)),
            "joint_vel": (frames, len(expected_joints)),
            "body_pos_w": (frames, len(expected_bodies), 3),
            "body_quat_w": (frames, len(expected_bodies), 4),
            "body_lin_vel_w": (frames, len(expected_bodies), 3),
            "body_ang_vel_w": (frames, len(expected_bodies), 3),
        }
        for name, shape in expected_shapes.items():
            values = np.asarray(archive[name])
            if frames == 0 or values.shape != shape:
                raise MotionValidationError(f"{name} must have shape {shape}, got {values.shape}")
            if not np.isfinite(values).all():
                raise MotionValidationError(f"{name} contains non-finite values")
        quaternion_norms = np.linalg.norm(body_quat, axis=2)
        if np.any(quaternion_norms < 1e-12):
            frame, body = (int(value) for value in np.argwhere(quaternion_norms < 1e-12)[0])
            raise MotionValidationError(
                f"body_quat_w has zero quaternion at frame {frame}, body {body}"
            )

    data = mujoco.MjData(model)
    max_position_error = 0.0
    max_orientation_error = 0.0
    joint_qpos_addresses = model.jnt_qposadr[1:]
    for frame in range(frames):
        data.qpos[:] = 0.0
        data.qpos[:3] = body_pos[frame, 0]
        root_quat = body_quat[frame, 0]
        data.qpos[3:7] = root_quat / np.linalg.norm(root_quat)
        data.qpos[joint_qpos_addresses] = joint_pos[frame]
        mujoco.mj_forward(model, data)
        position_error = np.linalg.norm(data.xpos[1:] - body_pos[frame], axis=1)
        expected_quat = body_quat[frame]
        expected_quat /= np.linalg.norm(expected_quat, axis=1, keepdims=True)
        dots = np.clip(np.abs(np.sum(data.xquat[1:] * expected_quat, axis=1)), 0.0, 1.0)
        orientation_error = 2.0 * np.arccos(dots)
        max_position_error = max(max_position_error, float(position_error.max()))
        max_orientation_error = max(max_orientation_error, float(orientation_error.max()))
    if (
        max_position_error > position_tolerance_m
        or max_orientation_error > orientation_tolerance_rad
    ):
        raise MotionValidationError(
            "kinematic replay differs from exported body transforms: "
            f"position={max_position_error:.6g} m, "
            f"orientation={max_orientation_error:.6g} rad"
        )
    return MotionValidationResult(frames, max_position_error, max_orientation_error)
