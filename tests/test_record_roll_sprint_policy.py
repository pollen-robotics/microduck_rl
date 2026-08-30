from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
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
        self.data.root_link_pos_w = pose[:, :3].clone()
        self.data.root_link_quat_w = pose[:, 3:].clone()

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
    assert env._roll_sprint_heading_ready.all()

    headings = env._roll_sprint_heading_w
    projected_starts = MODULE.microduck_mdp._roll_sprint_forward_position(
        env, robot, headings
    )
    yaws = torch.atan2(
        2.0
        * (robot.pose[:, 3] * robot.pose[:, 6] + robot.pose[:, 4] * robot.pose[:, 5]),
        1.0 - 2.0 * (robot.pose[:, 5].square() + robot.pose[:, 6].square()),
    )
    assert torch.allclose(projected_starts, projected_starts[:1], atol=1.0e-7)
    assert torch.allclose(projected_starts, torch.zeros(4), atol=1.0e-7)
    assert torch.allclose(headings, torch.tensor([[1.0, 0.0]] * 4), atol=1.0e-7)
    assert torch.allclose(yaws, torch.zeros(4), atol=1.0e-7)

    # The reward-side projection is locked to the exact shared race heading.
    robot.data.root_link_pos_w[:, 0] += 0.25
    projected_advance = MODULE.microduck_mdp._roll_sprint_forward_position(
        env, robot, env._roll_sprint_heading_w
    )
    assert torch.allclose(projected_advance, torch.full((4,), 0.25), atol=1.0e-7)


def test_race_camera_is_centered_down_the_shared_forward_axis() -> None:
    assert MODULE.RACE_CAMERA_LOOKAT[0] > 0.0
    assert MODULE.RACE_CAMERA_LOOKAT[1] == 0.0
    assert MODULE.RACE_CAMERA_DISTANCE >= 2.5
    assert MODULE.RACE_CAMERA_ELEVATION <= -45.0


def test_race_label_reports_robot_max_speed_and_valid_distance() -> None:
    label = MODULE._race_label_text(0, 1.234, 4.56)

    assert label == "R1  MAX 1.23 m/s  |  4.6 m valid"


def test_world_projection_anchors_label_to_robot_screen_position() -> None:
    camera = SimpleNamespace(
        pos=[0.0, 0.0, 0.0],
        forward=[1.0, 0.0, 0.0],
        up=[0.0, 0.0, 1.0],
        frustum_near=1.0,
        frustum_top=1.0,
        frustum_bottom=-1.0,
        frustum_center=0.0,
        frustum_width=4.0,
        orthographic=False,
    )

    pixels, visible = MODULE._project_world_points(
        MODULE.np.array([[2.0, 0.0, 0.0], [2.0, -1.0, 0.0]]),
        camera,
        width=200,
        height=100,
    )

    assert visible.tolist() == [True, True]
    assert pixels[0].tolist() == pytest.approx([100.0, 50.0])
    assert pixels[1, 0] > pixels[0, 0]


def test_camera_follows_first_robot_only_along_forward_axis() -> None:
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[8.5, -0.42, 0.12], [12.0, -0.14, 0.12]])
        )
    )
    camera = SimpleNamespace(lookat=MODULE.np.zeros(3))

    class _Scene:
        def __getitem__(self, name):
            assert name == "robot"
            return robot

    env = SimpleNamespace(
        scene=_Scene(),
        _offline_renderer=SimpleNamespace(_cam=camera),
    )

    MODULE._follow_first_robot(env)

    assert camera.lookat.tolist() == pytest.approx([8.5, 0.0, 0.08])
