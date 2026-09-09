"""Scene/policy plumbing for the move-away demo.

Holds the pieces that must agree with the rest of the repo: where the scene
lives, the default pose, and the observation layout the exported ONNX policies
expect.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

# Repo root is four levels up: scripts/behavior_demos/move_away/<this file>.
REPO_ROOT = Path(__file__).resolve().parents[3]
SCENE_XML = (
    REPO_ROOT
    / "src"
    / "mjlab_microduck"
    / "robot"
    / "microduck"
    / "scene_move_away.xml"
)

# Control rate the policies are trained and deployed at.
CTRL_HZ = 50.0

# The gyro observation MUST come from this sensor. `mj_name2id` returns -1 for
# an unknown name, and `model.sensor_adr[-1]` is a VALID index (the last
# sensor), so a typo here does not raise — it silently feeds a different
# quantity into the policy. That failure mode produced a plausible-looking but
# wrong demo, so the name is asserted at load time.
GYRO_SENSOR = "imu_ang_vel"

# STAND2 pose, identical to HOME_FRAME in microduck_constants.py and to the
# STAND keyframe in the scenes. Actions are offsets from this pose and joint
# observations are relative to it.
DEFAULT_POSE = np.array(
    [
        0.0,  # left_hip_yaw
        -0.0873,  # left_hip_roll
        -0.4579,  # left_hip_pitch
        -0.0049,  # left_knee
        0.4530,  # left_ankle
        0.3491,  # neck_pitch
        0.3491,  # head_pitch
        0.0,  # head_yaw
        0.0,  # head_roll
        0.0,  # right_hip_yaw
        0.0873,  # right_hip_roll
        0.4579,  # right_hip_pitch
        0.0049,  # right_knee
        -0.4530,  # right_ankle
    ],
    dtype=np.float32,
)

HEAD_PITCH_ACT = 6
HEAD_YAW_ACT = 7


def quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate ``vec`` by the inverse of quaternion ``[w, x, y, z]``."""
    w = quat[0]
    xyz = quat[1:4]
    t = np.cross(xyz, vec) * 2.0
    return vec - w * t + np.cross(xyz, t)


def sensor_address(model: mujoco.MjModel, name: str) -> int:
    """Resolve a sensor's address, refusing to silently accept a bad name."""
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise ValueError(f"sensor {name!r} not found in the model")
    return int(model.sensor_adr[sensor_id])


def actuator_indices(model: mujoco.MjModel) -> tuple[list[int], list[int]]:
    """qpos and qvel indices of every actuated joint, in actuator order."""
    qpos_idx = [int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]
    qvel_idx = [int(model.jnt_dofadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]
    return qpos_idx, qvel_idx


def build_observation(
    gyro: np.ndarray,
    projected_gravity: np.ndarray,
    joint_pos_rel: np.ndarray,
    joint_vel: np.ndarray,
    last_action: np.ndarray,
    twist: np.ndarray,
    command_dim: int,
) -> np.ndarray:
    """Assemble the actor observation vector.

    Layout (must match the exported policy):
    ``[base_ang_vel(3), projected_gravity(3), joint_pos(14), joint_vel(14),
    last_action(14), command(command_dim)]``. The command block is
    ``[twist(3), head_pose(4), body_pose(6)]`` on 61D policies; this demo only
    writes the twist slot and ZERO-PADS the rest, which is the documented
    convention for a task that does not use a slot.
    """
    command = np.zeros(command_dim, dtype=np.float32)
    command[0:3] = twist
    return np.concatenate(
        [gyro, projected_gravity, joint_pos_rel, joint_vel, last_action, command]
    ).astype(np.float32)
