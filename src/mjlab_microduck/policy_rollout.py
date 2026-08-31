"""Headless, deterministic ONNX policy rollouts for the canonical Microduck scene."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from mjlab_microduck.blender_motion import validate_motion

MICRODUCK_SCENE_XML = Path(__file__).parent / "robot/microduck/scene.xml"
SIMULATION_TIMESTEP_S = 0.005
CONTROL_DECIMATION = 4
CONTROL_HZ = 50
DEFAULT_POSE = np.asarray(
    [
        0.0,
        -0.0873,
        -0.4579,
        -0.0049,
        0.4530,
        0.3491,
        0.3491,
        0.0,
        0.0,
        0.0,
        0.0873,
        0.4579,
        0.0049,
        -0.4530,
    ],
    dtype=np.float32,
)


class PolicyRolloutError(ValueError):
    """A policy rollout cannot be exported safely."""


@dataclass(frozen=True)
class PolicyRolloutConfig:
    policy_path: Path
    output_path: Path
    duration_s: float = 4.0
    command: tuple[float, float, float] = (0.30, 0.0, 0.0)
    seed: int = 0


def _joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    return tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(1, model.njnt)
    )


def _body_names(model: mujoco.MjModel) -> tuple[str, ...]:
    return tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        for body_id in range(1, model.nbody)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_shape(shape: list[object]) -> str:
    return "[" + ",".join(str(value) for value in shape) + "]"


def _validate_joint_metadata(
    session: ort.InferenceSession, expected_joint_names: tuple[str, ...]
) -> None:
    metadata = session.get_modelmeta().custom_metadata_map
    metadata_names = metadata.get("joint_names")
    if metadata_names is None:
        raise PolicyRolloutError("ONNX metadata must include comma-separated joint_names")
    actual_joint_names = tuple(metadata_names.split(","))
    for index, (expected, actual) in enumerate(
        zip(expected_joint_names, actual_joint_names, strict=False)
    ):
        if expected != actual:
            raise PolicyRolloutError(
                "joint_names mismatch at index "
                f"{index}: expected {expected!r}, got {actual!r}"
            )
    if len(actual_joint_names) != len(expected_joint_names):
        index = min(len(actual_joint_names), len(expected_joint_names))
        expected = expected_joint_names[index] if index < len(expected_joint_names) else None
        actual = actual_joint_names[index] if index < len(actual_joint_names) else None
        raise PolicyRolloutError(
            "joint_names mismatch at index "
            f"{index}: expected {expected!r}, got {actual!r}"
        )


def _validate_config(
    config: PolicyRolloutConfig,
) -> tuple[
    Path,
    Path,
    float,
    np.ndarray,
    int,
    ort.InferenceSession,
    mujoco.MjModel,
]:
    policy_path = Path(config.policy_path)
    output_path = Path(config.output_path)
    if not policy_path.is_file():
        raise PolicyRolloutError(f"policy file does not exist: {policy_path}")
    if output_path.suffix.lower() != ".npz":
        raise PolicyRolloutError("output_path must use the .npz extension")
    if policy_path.resolve() == output_path.resolve():
        raise PolicyRolloutError("policy_path and output_path must resolve to different files")
    try:
        duration_s = float(config.duration_s)
    except (TypeError, ValueError) as exc:
        raise PolicyRolloutError("duration_s must be a positive finite number") from exc
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise PolicyRolloutError("duration_s must be a positive finite number")
    frames_float = duration_s * CONTROL_HZ
    frames = round(frames_float)
    if not math.isclose(frames_float, frames, rel_tol=0.0, abs_tol=1e-9):
        raise PolicyRolloutError(
            f"duration_s must produce an integral number of {CONTROL_HZ} Hz frames"
        )
    try:
        command = np.asarray(config.command, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PolicyRolloutError(
            "command must contain exactly three finite values"
        ) from exc
    if command.shape != (3,) or not np.isfinite(command).all():
        raise PolicyRolloutError("command must contain exactly three finite values")
    if not isinstance(config.seed, (int, np.integer)):
        raise PolicyRolloutError("seed must be an integer")
    try:
        session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise PolicyRolloutError(f"could not load ONNX policy: {policy_path}") from exc
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    input_shape = list(inputs[0].shape) if len(inputs) == 1 else []
    output_shape = list(outputs[0].shape) if len(outputs) == 1 else []
    if (
        len(inputs) != 1
        or len(outputs) != 1
        or inputs[0].name != "obs"
        or outputs[0].name != "actions"
        or input_shape != [1, 61]
        or output_shape != [1, 14]
    ):
        input_description = (
            f"{inputs[0].name} {_format_shape(input_shape)}"
            if len(inputs) == 1
            else f"{len(inputs)} inputs"
        )
        output_description = (
            f"{outputs[0].name} {_format_shape(output_shape)}"
            if len(outputs) == 1
            else f"{len(outputs)} outputs"
        )
        raise PolicyRolloutError(
            "ONNX contract must be obs [1,61] -> actions [1,14]; got "
            f"{input_description} -> {output_description}"
        )
    model = mujoco.MjModel.from_xml_path(str(MICRODUCK_SCENE_XML))
    model.opt.timestep = SIMULATION_TIMESTEP_S
    if not math.isclose(
        CONTROL_DECIMATION * float(model.opt.timestep),
        1.0 / CONTROL_HZ,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise PolicyRolloutError(
            "canonical simulation clock must use four 0.005 s substeps per 50 Hz frame"
        )
    _validate_joint_metadata(session, _joint_names(model))
    return policy_path, output_path, duration_s, command, frames, session, model


def _observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    trunk_body_id: int,
    angular_velocity_sensor_id: int,
    joint_qpos_addresses: np.ndarray,
    joint_qvel_addresses: np.ndarray,
    previous_action: np.ndarray,
    command: np.ndarray,
) -> np.ndarray:
    sensor_address = int(model.sensor_adr[angular_velocity_sensor_id])
    angular_velocity = data.sensordata[sensor_address : sensor_address + 3]
    trunk_rotation = data.xmat[trunk_body_id].reshape(3, 3)
    projected_gravity = trunk_rotation.T @ np.asarray([0.0, 0.0, -1.0])
    command_block = np.concatenate((command, np.zeros(10, dtype=np.float32)))
    observation = np.concatenate(
        (
            angular_velocity,
            projected_gravity,
            data.qpos[joint_qpos_addresses] - DEFAULT_POSE,
            data.qvel[joint_qvel_addresses],
            previous_action,
            command_block,
        )
    )
    if observation.shape != (61,):
        raise PolicyRolloutError(
            f"internal observation has shape {observation.shape}, expected (61,)"
        )
    return observation.astype(np.float32, copy=False)


def _rollout_config_sha256(
    *, duration_s: float, command: np.ndarray, seed: int
) -> str:
    payload = {
        "command": [float(value) for value in command],
        "control_decimation": CONTROL_DECIMATION,
        "control_hz": CONTROL_HZ,
        "duration_s": float(duration_s),
        "seed": int(seed),
        "timestep_s": SIMULATION_TIMESTEP_S,
    }
    canonical_json = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _body_world_velocities(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[np.ndarray, np.ndarray]:
    angular = np.empty((model.nbody - 1, 3), dtype=np.float64)
    linear = np.empty_like(angular)
    velocity = np.empty(6, dtype=np.float64)
    for body_id in range(1, model.nbody):
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            velocity,
            0,
        )
        angular[body_id - 1] = velocity[:3]
        linear[body_id - 1] = velocity[3:]
    return angular, linear


def build_motion_archive(
    *,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    body_lin_vel_w: np.ndarray,
    body_ang_vel_w: np.ndarray,
    joint_names: tuple[str, ...],
    body_names: tuple[str, ...],
    source_hashes: dict[str, str],
) -> dict[str, np.ndarray]:
    """Build the native, self-describing motion archive consumed by Blender."""
    return {
        "joint_pos": np.asarray(joint_pos, dtype=np.float32),
        "joint_vel": np.asarray(joint_vel, dtype=np.float32),
        "body_pos_w": np.asarray(body_pos_w, dtype=np.float32),
        "body_quat_w": np.asarray(body_quat_w, dtype=np.float32),
        "body_lin_vel_w": np.asarray(body_lin_vel_w, dtype=np.float32),
        "body_ang_vel_w": np.asarray(body_ang_vel_w, dtype=np.float32),
        "fps": np.asarray([50], dtype=np.int32),
        "schema_version": np.asarray([1], dtype=np.int32),
        "joint_names": np.asarray(joint_names),
        "body_names": np.asarray(body_names),
        "source_hashes_json": np.asarray([json.dumps(source_hashes, sort_keys=True)]),
    }


def export_policy_rollout(config: PolicyRolloutConfig) -> Path:
    """Run a canonical 50 Hz MuJoCo rollout and atomically export its archive."""
    policy_path, output_path, duration_s, command_values, frames, session, model = (
        _validate_config(config)
    )
    np.random.default_rng(config.seed)
    data = mujoco.MjData(model)
    joint_names = _joint_names(model)
    body_names = _body_names(model)
    joint_qpos_addresses = model.jnt_qposadr[1:]
    joint_qvel_addresses = model.jnt_dofadr[1:]
    root_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
    )
    trunk_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    angular_velocity_sensor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel"
    )
    if min(root_joint_id, trunk_body_id, angular_velocity_sensor_id) < 0:
        raise PolicyRolloutError("canonical scene is missing a required root, trunk, or IMU")
    if len(joint_names) != len(DEFAULT_POSE) or model.nu != len(DEFAULT_POSE):
        raise PolicyRolloutError("canonical scene must expose exactly 14 actuated joints")

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
    data.qpos[joint_qpos_addresses] = DEFAULT_POSE
    data.ctrl[:] = DEFAULT_POSE

    joint_pos = np.empty((frames, len(joint_names)), dtype=np.float32)
    joint_vel = np.empty_like(joint_pos)
    body_pos_w = np.empty((frames, len(body_names), 3), dtype=np.float32)
    body_quat_w = np.empty((frames, len(body_names), 4), dtype=np.float32)
    body_lin_vel_w = np.empty((frames, len(body_names), 3), dtype=np.float32)
    body_ang_vel_w = np.empty((frames, len(body_names), 3), dtype=np.float32)
    previous_action = np.zeros(len(joint_names), dtype=np.float32)
    command = command_values.astype(np.float32)

    for frame in range(frames):
        mujoco.mj_forward(model, data)
        joint_pos[frame] = data.qpos[joint_qpos_addresses]
        joint_vel[frame] = data.qvel[joint_qvel_addresses]
        body_pos_w[frame] = data.xpos[1:]
        body_quat_w[frame] = data.xquat[1:]
        angular_velocity, linear_velocity = _body_world_velocities(model, data)
        body_ang_vel_w[frame] = angular_velocity
        body_lin_vel_w[frame] = linear_velocity

        observation = _observation(
            model,
            data,
            trunk_body_id=trunk_body_id,
            angular_velocity_sensor_id=angular_velocity_sensor_id,
            joint_qpos_addresses=joint_qpos_addresses,
            joint_qvel_addresses=joint_qvel_addresses,
            previous_action=previous_action,
            command=command,
        )
        action_batch = np.asarray(
            session.run(["actions"], {"obs": observation[None, :]})[0], dtype=np.float32
        )
        if action_batch.shape != (1, len(joint_names)) or not np.isfinite(action_batch).all():
            raise PolicyRolloutError("ONNX action output must be finite with shape [1,14]")
        previous_action = action_batch[0].copy()
        data.ctrl[:] = DEFAULT_POSE + previous_action
        for _ in range(CONTROL_DECIMATION):
            mujoco.mj_step(model, data)

    archive = build_motion_archive(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        joint_names=joint_names,
        body_names=body_names,
        source_hashes={
            "policy_sha256": _sha256(policy_path),
            "rollout_config_sha256": _rollout_config_sha256(
                duration_s=duration_s,
                command=command_values,
                seed=int(config.seed),
            ),
            "scene_sha256": _sha256(MICRODUCK_SCENE_XML),
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent, prefix=f".{output_path.stem}.", suffix=".npz"
    )
    temporary_path = Path(temporary_name)
    try:
        with open(file_descriptor, "wb", closefd=True) as temporary_file:
            np.savez_compressed(temporary_file, **archive)
        validate_motion(temporary_path)
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path
