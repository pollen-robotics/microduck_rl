from __future__ import annotations

import importlib.util
from itertools import pairwise
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


def test_race_camera_is_close_and_keeps_all_lanes_centered() -> None:
    assert MODULE.RACE_CAMERA_LOOKAT[0] == pytest.approx(0.60)
    assert MODULE.RACE_CAMERA_LOOKAT[1] == 0.0
    assert 2.5 <= MODULE.RACE_CAMERA_DISTANCE <= 3.5
    assert MODULE.RACE_CAMERA_FOVY == 45.0
    assert MODULE.RACE_CAMERA_AZIMUTH == 90.0
    assert MODULE.RACE_CAMERA_ELEVATION == -45.0
    assert MODULE.ROAD_HALF_WIDTH_M == pytest.approx(0.56)
    assert MODULE.ROAD_SAFE_FULL_REWARD_HALF_WIDTH_M == pytest.approx(0.42)
    assert MODULE.ROAD_REPOSITION_TRIGGER_M == pytest.approx(0.50)
    assert MODULE.ROAD_REPOSITION_REARM_M == pytest.approx(0.46)


def test_follow_camera_advances_and_retreats_smoothly() -> None:
    dt_s = 0.02
    initial = MODULE.CameraFollowState(0.60)
    first = MODULE._camera_follow_x(initial, 1.00, dt_s)
    second = MODULE._camera_follow_x(first, 1.50, dt_s)

    assert 0.60 < first.x_m < second.x_m
    assert first.velocity_mps > 0.0
    assert second.velocity_mps > first.velocity_mps
    assert second.velocity_mps - first.velocity_mps <= (
        MODULE.RACE_CAMERA_MAX_ACCEL_MPS2 * dt_s + 1.0e-12
    )
    assert abs(second.velocity_mps) <= MODULE.RACE_CAMERA_MAX_SPEED_MPS
    assert MODULE._camera_follow_x(second, float("nan"), dt_s) == second


def test_follow_camera_does_not_snap_when_leader_resets_behind() -> None:
    dt_s = 0.02
    moving_forward = MODULE.CameraFollowState(x_m=3.0, velocity_mps=1.2)
    after_reset = MODULE._camera_follow_x(moving_forward, -5.0, dt_s)

    # The view keeps its momentum for one frame and brakes at the configured
    # acceleration bound instead of teleporting or reversing immediately.
    assert after_reset.x_m > moving_forward.x_m
    assert after_reset.velocity_mps == pytest.approx(
        moving_forward.velocity_mps - MODULE.RACE_CAMERA_MAX_ACCEL_MPS2 * dt_s
    )
    assert abs(after_reset.x_m - moving_forward.x_m) <= (
        MODULE.RACE_CAMERA_MAX_SPEED_MPS * dt_s
    )


def test_recording_fps_preserves_real_simulation_time() -> None:
    assert MODULE._recording_fps(0.02, 4) == pytest.approx(12.5)


def test_race_header_uses_requested_rollout_duration() -> None:
    assert MODULE._race_header_text(19.1, 20.0, 2) == (
        "10 m ROLL RACE  |  t 019.1 s / 20.0 s"
        "  |  camera follows on-road standing leader R3"
    )


def test_recovery_montage_uses_all_four_deterministic_orientations(
    monkeypatch,
) -> None:
    observed = {}

    def arrange(env, lane_spacing, *, seed, orientations):
        observed.update(
            env=env,
            lane_spacing=lane_spacing,
            seed=seed,
            orientations=orientations,
        )

    monkeypatch.setattr(
        MODULE.microduck_mdp,
        "arrange_roll_sprint_recovery_start",
        arrange,
    )
    env = object()
    MODULE._arrange_recovery_montage(env, seed=3)

    assert observed == {
        "env": env,
        "lane_spacing": MODULE.RACE_LANE_SPACING,
        "seed": 3,
        "orientations": ("face_down", "face_up", "left", "right"),
    }
    assert MODULE._recovery_header_text(11.5, 12.0) == (
        "SELF-RIGHT -> REPOSITION -> REROLL  |  t 011.5 s / 12.0 s"
    )


def test_camera_follows_furthest_credited_standing_robot_anywhere_on_road(
    monkeypatch,
) -> None:
    camera = SimpleNamespace(lookat=MODULE.np.zeros(3))
    env = SimpleNamespace(
        _offline_renderer=SimpleNamespace(_cam=camera),
        _roll_sprint_forward_position=torch.tensor([8.0, 3.0, 2.0, 1.0]),
        _roll_sprint_forward_frontier=torch.tensor([4.0, 3.0, 2.0, 1.0]),
        _roll_sprint_forward_origin=torch.zeros(4),
        _roll_sprint_lateral_displacement=torch.tensor([0.60, 0.10, 0.0, 0.0]),
        _roll_sprint_course_lateral_position=torch.tensor([0.18, -0.04, 0.14, 0.42]),
        scene=SimpleNamespace(
            terrain=SimpleNamespace(
                env_origins=torch.tensor(
                    [
                        [0.0, -0.42, 0.0],
                        [0.0, -0.14, 0.0],
                        [0.0, 0.14, 0.0],
                        [0.0, 0.42, 0.0],
                    ]
                )
            )
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "_launch_ready_mask",
        lambda _env: torch.ones(4, dtype=torch.bool),
    )

    initial = MODULE.CameraFollowState(0.60)
    next_state, next_y, leader_index = MODULE._follow_on_road_leader(
        env,
        initial,
        1,
        0.02,
    )

    # R1 crossed its original lane, but course y = -0.42 + 0.60 = +0.18 m.
    assert leader_index == 0
    assert next_state.x_m > initial.x_m
    assert next_state.velocity_mps == pytest.approx(
        MODULE.RACE_CAMERA_MAX_ACCEL_MPS2 * 0.02
    )
    assert next_y == 0.0
    assert camera.lookat.tolist() == pytest.approx(
        [next_state.x_m, next_y, MODULE.RACE_CAMERA_LOOKAT[2]]
    )


def test_leader_rejects_lying_or_off_road_robot_and_retains_safe_fallback(
    monkeypatch,
) -> None:
    env = SimpleNamespace(
        _roll_sprint_forward_position=torch.tensor([9.0, 4.0, 3.0, 2.0]),
        _roll_sprint_forward_frontier=torch.tensor([8.0, 4.0, 3.0, 2.0]),
        _roll_sprint_forward_origin=torch.zeros(4),
        _roll_sprint_lateral_displacement=torch.tensor([-0.20, 0.0, 0.0, 0.20]),
        _roll_sprint_course_lateral_position=torch.tensor([-0.62, -0.14, 0.14, 0.62]),
        scene=SimpleNamespace(
            terrain=SimpleNamespace(
                env_origins=torch.tensor(
                    [
                        [0.0, -0.42, 0.0],
                        [0.0, -0.14, 0.0],
                        [0.0, 0.14, 0.0],
                        [0.0, 0.42, 0.0],
                    ]
                )
            )
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "_launch_ready_mask",
        lambda _env: torch.tensor([True, True, False, True]),
    )

    # R1 and R4 are outside the shared road; standing on-road R2 wins.
    assert MODULE._select_on_road_leader(env) == 1

    monkeypatch.setattr(
        MODULE,
        "_launch_ready_mask",
        lambda _env: torch.zeros(4, dtype=torch.bool),
    )
    # With no standing robot, retain the previous on-road leader safely.
    assert MODULE._select_on_road_leader(env, previous_leader_index=2) == 2


def test_corridor_has_four_lanes_and_spans_exactly_ten_meters() -> None:
    segments = MODULE._race_corridor_segments()
    longitudinal = [
        (start, end) for start, end, _color, _radius in segments if start[1] == end[1]
    ]
    cross_track = [
        (start, end) for start, end, _color, _radius in segments if start[0] == end[0]
    ]

    assert MODULE.TARGET_DISTANCE_M == pytest.approx(10.0)
    assert len(longitudinal) == 5
    assert all(start[0] == 0.0 for start, _end in longitudinal)
    assert all(end[0] == MODULE.TARGET_DISTANCE_M for _start, end in longitudinal)
    assert sorted(start[1] for start, _end in longitudinal) == pytest.approx(
        [-0.56, -0.28, 0.0, 0.28, 0.56]
    )
    assert longitudinal[0][0][1] == pytest.approx(-MODULE.ROAD_HALF_WIDTH_M)
    assert longitudinal[-1][0][1] == pytest.approx(MODULE.ROAD_HALF_WIDTH_M)
    assert {start[0] for start, _end in cross_track} == set(
        MODULE.np.arange(0.0, MODULE.TARGET_DISTANCE_M + 1.0, 1.0)
    )


def test_corridor_visualizer_installs_when_no_existing_callback() -> None:
    env = SimpleNamespace()
    observed = []
    visualizer = SimpleNamespace(
        add_cylinder=lambda start, end, radius, color: observed.append(
            (start, end, radius, color)
        )
    )

    MODULE._install_race_corridor_visualizer(env)
    env.update_visualizers(visualizer)

    assert len(observed) == len(MODULE._race_corridor_segments())


def test_race_label_reports_robot_max_speed_and_valid_distance() -> None:
    label = MODULE._race_label_text(0, 1.234, 4.56)

    assert label == "R1  MAX 1s 1.23 m/s  |  4.6 m valid"


def test_video_overlay_uses_compact_fonts() -> None:
    assert MODULE.ROBOT_LABEL_FONT_SIZE <= 13
    assert MODULE.HEADER_FONT_SIZE <= 15


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


def test_speed_accumulator_uses_visible_forward_displacement_when_velocity_lags() -> (
    None
):
    max_speeds, current_position = MODULE._accumulate_max_forward_speed(
        torch.zeros(2),
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.02, 0.01]),
        torch.zeros(2, 3),
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        0.02,
    )

    assert current_position.tolist() == pytest.approx([0.02, 0.01])
    assert max_speeds.tolist() == pytest.approx([1.0, 0.5])


def test_speed_accumulator_keeps_peak_and_ignores_backward_motion() -> None:
    max_speeds, current_position = MODULE._accumulate_max_forward_speed(
        torch.tensor([1.2, 0.8]),
        torch.tensor([0.5, 0.5]),
        torch.tensor([0.49, 0.51]),
        torch.tensor([[-2.0, 0.0, 0.0], [0.9, 0.0, 0.0]]),
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        0.02,
    )

    assert current_position.tolist() == pytest.approx([0.49, 0.51])
    assert max_speeds.tolist() == pytest.approx([1.2, 0.9])


def test_label_layout_does_not_overlap_aligned_robot_labels() -> None:
    positions = MODULE._label_positions(
        MODULE.np.array([[100.0, 200.0]] * 4),
        MODULE.np.ones(4, dtype=bool),
        [(150, 22)] * 4,
        width=960,
        height=540,
    )

    boxes = [
        (x, y, x + 150, y + 22) for index in range(4) for x, y in [positions[index]]
    ]
    assert all(first[3] + 7.0 <= second[1] for first, second in pairwise(boxes))
