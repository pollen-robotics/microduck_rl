from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from mjlab_microduck.rom.observation import (
    DEFAULT_JOINT_POSE,
    OBSERVATION_NORMALIZATION,
    DeploymentCommand,
    DeploymentState,
    build_actor_observation,
)


def _sample_state(*, quaternion: np.ndarray | None = None) -> DeploymentState:
    return DeploymentState(
        base_angular_velocity_radps=np.array([1.0, 2.0, 3.0]),
        base_orientation_wxyz=(
            np.array([1.0, 0.0, 0.0, 0.0]) if quaternion is None else quaternion
        ),
        joint_positions_rad=DEFAULT_JOINT_POSE + np.arange(14, dtype=np.float32),
        joint_velocities_radps=np.arange(14, dtype=np.float32) + 20.0,
        previous_action=np.arange(14, dtype=np.float32) + 40.0,
    )


def test_actor_observation_is_shared_61d_contract() -> None:
    """Dropping or reordering a command slot would make hot-swapped policies incompatible."""
    observation = build_actor_observation(_sample_state(), DeploymentCommand.zero())

    expected = np.concatenate(
        (
            [1.0, 2.0, 3.0],
            [0.0, 0.0, -1.0],
            np.arange(14, dtype=np.float32),
            np.arange(14, dtype=np.float32) + 20.0,
            np.arange(14, dtype=np.float32) + 40.0,
            np.zeros(13, dtype=np.float32),
        )
    ).astype(np.float32)
    assert observation.shape == (61,)
    assert observation.dtype == np.float32
    np.testing.assert_array_equal(observation, expected)
    np.testing.assert_array_equal(observation[48:51], [0.0, 0.0, 0.0])
    assert OBSERVATION_NORMALIZATION == "BAKED_IN_ONNX"


def test_actor_observation_uses_mujoco_wxyz_quaternion_and_exact_command_slots() -> (
    None
):
    """Treating MuJoCo's quaternion as xyzw would rotate gravity into the wrong body axis."""
    half_sqrt = math.sqrt(0.5)
    command = DeploymentCommand(
        twist=np.array([1.0, 2.0, 3.0]),
        head_pose=np.array([4.0, 5.0, 6.0, 7.0]),
        body_pose=np.array([8.0, 9.0, 10.0, 11.0, 12.0, 13.0]),
    )

    observation = build_actor_observation(
        _sample_state(quaternion=np.array([half_sqrt, half_sqrt, 0.0, 0.0])),
        command,
    )

    np.testing.assert_allclose(observation[3:6], [0.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_array_equal(observation[48:51], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(observation[51:55], [4.0, 5.0, 6.0, 7.0])
    np.testing.assert_array_equal(
        observation[55:61], [8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
    )


@pytest.mark.parametrize("field", ["state", "command"])
def test_actor_observation_rejects_non_finite_inputs(field: str) -> None:
    """Allowing NaN intent or state to reach ONNX would make fail-safe behavior undefined."""
    state = _sample_state()
    command = DeploymentCommand.zero()
    if field == "state":
        state = DeploymentState(
            base_angular_velocity_radps=np.array([math.nan, 0.0, 0.0]),
            base_orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            joint_positions_rad=DEFAULT_JOINT_POSE,
            joint_velocities_radps=np.zeros(14),
            previous_action=np.zeros(14),
        )
    else:
        command = DeploymentCommand(
            twist=np.array([math.inf, 0.0, 0.0]),
            head_pose=np.zeros(4),
            body_pose=np.zeros(6),
        )

    with pytest.raises(ValueError, match="finite"):
        build_actor_observation(state, command)


def test_inference_rehearsal_uses_shared_finite_61d_builder() -> None:
    """Keeping a private concat path would let rehearsal accept state the runtime rejects."""
    script = Path(__file__).parents[1] / "scripts" / "infer_policy.py"
    spec = importlib.util.spec_from_file_location("infer_policy_for_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = module.PolicyInference.__new__(module.PolicyInference)
    policy.new_cmd_obs = True
    policy.use_projected_gravity = True
    policy.command = np.zeros(13, dtype=np.float32)
    policy.command[0] = np.nan
    policy.last_action = np.zeros(14, dtype=np.float32)
    policy.get_base_ang_vel = lambda: np.zeros(3, dtype=np.float32)
    policy.get_projected_gravity = lambda: np.array([0.0, 0.0, -1.0], dtype=np.float32)
    policy.get_joint_pos_relative = lambda: np.zeros(14, dtype=np.float32)
    policy.get_joint_vel = lambda: np.zeros(14, dtype=np.float32)

    with pytest.raises(ValueError, match="finite"):
        policy.get_observations()
