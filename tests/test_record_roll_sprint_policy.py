from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "record_roll_sprint_policy.py"
)
SPEC = importlib.util.spec_from_file_location("record_roll_sprint_policy", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _FakeRobot:
    def __init__(self) -> None:
        self.data = SimpleNamespace(
            root_link_pos_w=torch.tensor(
                [
                    [-0.2, -0.2, 0.11],
                    [-0.2, 0.2, 0.12],
                    [0.2, -0.2, 0.11],
                    [0.2, 0.2, 0.12],
                ]
            ),
            root_link_quat_w=torch.tensor(
                [[0.5, 0.5, 0.5, 0.5]] * 4, dtype=torch.float32
            ),
        )
        self.pose = None
        self.velocity = None

    def write_root_link_pose_to_sim(self, pose) -> None:
        self.pose = pose.clone()

    def write_root_link_velocity_to_sim(self, velocity) -> None:
        self.velocity = velocity.clone()


def test_canonical_video_arranges_four_parallel_deterministic_race_lanes() -> None:
    robot = _FakeRobot()
    terrain = SimpleNamespace(
        env_origins=torch.tensor(
            [
                [-0.2, -0.2, 0.0],
                [-0.2, 0.2, 0.0],
                [0.2, -0.2, 0.0],
                [0.2, 0.2, 0.0],
            ]
        )
    )

    class _Scene:
        def __init__(self) -> None:
            self.terrain = terrain

        def __getitem__(self, name):
            assert name == "robot"
            return robot

    sim = SimpleNamespace(forward_called=False)

    def forward() -> None:
        sim.forward_called = True

    sim.forward = forward
    env = SimpleNamespace(
        num_envs=4,
        device=torch.device("cpu"),
        scene=_Scene(),
        sim=sim,
    )

    MODULE._arrange_race_start(env)

    expected_y = torch.tensor([-0.42, -0.14, 0.14, 0.42])
    assert torch.equal(robot.pose[:, 0], torch.zeros(4))
    assert torch.allclose(robot.pose[:, 1], expected_y)
    assert torch.allclose(robot.pose[:, 2], torch.tensor([0.11, 0.12, 0.11, 0.12]))
    assert torch.equal(
        robot.pose[:, 3:],
        torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4),
    )
    assert torch.equal(robot.velocity, torch.zeros(4, 6))
    assert torch.equal(terrain.env_origins[:, 0], torch.zeros(4))
    assert torch.allclose(terrain.env_origins[:, 1], expected_y)
    assert sim.forward_called
    assert not env._roll_sprint_heading_ready.any()


def test_race_camera_is_centered_down_the_shared_forward_axis() -> None:
    assert MODULE.RACE_CAMERA_LOOKAT[0] > 0.0
    assert MODULE.RACE_CAMERA_LOOKAT[1] == 0.0
    assert MODULE.RACE_CAMERA_DISTANCE >= 2.5
    assert MODULE.RACE_CAMERA_ELEVATION <= -45.0
