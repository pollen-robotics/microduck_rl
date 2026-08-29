"""Shared, unnormalized MicroDuck deployment observation construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

OBSERVATION_NORMALIZATION = "BAKED_IN_ONNX"

# HOME/STAND2 from robot.microduck_constants and the existing deployment rehearsal.
DEFAULT_JOINT_POSE: NDArray[np.float32] = np.array(
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


@dataclass(frozen=True)
class DeploymentState:
    """One policy-rate state using MuJoCo's ``[w, x, y, z]`` quaternion order."""

    base_angular_velocity_radps: ArrayLike
    base_orientation_wxyz: ArrayLike
    joint_positions_rad: ArrayLike
    joint_velocities_radps: ArrayLike
    previous_action: ArrayLike
    gravity_vector_body: ArrayLike | None = None


@dataclass(frozen=True)
class DeploymentCommand:
    """The fixed command block: twist(3), head pose(4), then body pose(6)."""

    twist: ArrayLike
    head_pose: ArrayLike
    body_pose: ArrayLike

    @classmethod
    def zero(cls) -> DeploymentCommand:
        return cls(
            twist=np.zeros(3, dtype=np.float32),
            head_pose=np.zeros(4, dtype=np.float32),
            body_pose=np.zeros(6, dtype=np.float32),
        )


def _vector(value: ArrayLike, length: int, name: str) -> NDArray[np.float32]:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


def project_gravity_wxyz(quaternion_wxyz: ArrayLike) -> NDArray[np.float32]:
    """Rotate world down into the body frame with MuJoCo ``wxyz`` convention."""
    quaternion = _vector(quaternion_wxyz, 4, "base_orientation_wxyz")
    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(np.float32).eps:
        raise ValueError("base_orientation_wxyz must be a non-zero quaternion")
    quaternion = quaternion / norm
    world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    scalar = quaternion[0]
    vector = quaternion[1:4]
    cross = np.cross(vector, world_gravity) * 2.0
    return (world_gravity - scalar * cross + np.cross(vector, cross)).astype(np.float32)


def build_actor_observation(
    state: DeploymentState, command: DeploymentCommand
) -> NDArray[np.float32]:
    """Build the exact raw 61D actor input; ONNX owns all normalization."""
    observation = np.concatenate(
        (
            _vector(
                state.base_angular_velocity_radps, 3, "base_angular_velocity_radps"
            ),
            (
                project_gravity_wxyz(state.base_orientation_wxyz)
                if state.gravity_vector_body is None
                else _vector(state.gravity_vector_body, 3, "gravity_vector_body")
            ),
            _vector(state.joint_positions_rad, 14, "joint_positions_rad")
            - DEFAULT_JOINT_POSE,
            _vector(state.joint_velocities_radps, 14, "joint_velocities_radps"),
            _vector(state.previous_action, 14, "previous_action"),
            _vector(command.twist, 3, "twist"),
            _vector(command.head_pose, 4, "head_pose"),
            _vector(command.body_pose, 6, "body_pose"),
        )
    ).astype(np.float32, copy=False)
    if observation.shape != (61,) or not np.isfinite(observation).all():
        raise ValueError(
            "actor observation must be a finite float32 vector of shape (61,)"
        )
    return observation
