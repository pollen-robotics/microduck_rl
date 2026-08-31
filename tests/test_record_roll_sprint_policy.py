from __future__ import annotations

import importlib.util
import sys
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


def test_backroll_video_faces_away_but_credits_the_same_world_race_axis() -> None:
    robot = _FakeRobot()
    terrain = SimpleNamespace(env_origins=torch.zeros((4, 3)))

    class _Scene:
        def __init__(self) -> None:
            self.terrain = terrain

        def __getitem__(self, name):
            assert name == "robot"
            return robot

    sim = SimpleNamespace(forward=lambda: None)
    env = SimpleNamespace(
        num_envs=4,
        device=torch.device("cpu"),
        scene=_Scene(),
        sim=sim,
    )

    MODULE._arrange_race_start(
        env,
        task_id="Mjlab-Backroll-Sprint-Flat-MicroDuck",
    )

    expected_lanes = torch.tensor([-0.42, -0.14, 0.14, 0.42])
    assert torch.allclose(robot.pose[:, 0], torch.zeros(4), atol=1.0e-7)
    assert torch.allclose(robot.pose[:, 1], expected_lanes, atol=1.0e-7)
    assert torch.allclose(
        robot.pose[:, 3:],
        torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 4),
        atol=1.0e-7,
    )
    assert torch.allclose(
        env._roll_sprint_body_heading_w,
        torch.tensor([[-1.0, 0.0]] * 4),
        atol=1.0e-7,
    )
    assert torch.allclose(
        env._roll_sprint_heading_w,
        torch.tensor([[1.0, 0.0]] * 4),
        atol=1.0e-7,
    )
    robot.data.root_link_pos_w[:, 0] += 0.25
    projected_advance = MODULE.microduck_mdp._roll_sprint_forward_position(
        env,
        robot,
        env._roll_sprint_heading_w,
    )
    assert torch.allclose(projected_advance, torch.full((4,), 0.25), atol=1.0e-7)


def test_policy_load_precedes_final_race_arrangement_and_observation_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    base_env = SimpleNamespace(
        num_envs=4,
        forward=torch.full((4,), 9.0),
        lateral=torch.full((4,), 9.0),
        yaw=torch.full((4,), 1.0),
        heading=torch.zeros((4, 2)),
        reward_heading=torch.zeros((4, 2)),
    )

    class FakeWrapper:
        def __init__(self, env, *, clip_actions) -> None:
            assert env is base_env
            assert clip_actions == 1.0
            events.append("wrapper_reset")
            env.forward.fill_(7.0)
            env.lateral.fill_(7.0)
            env.yaw.fill_(0.7)

    class FakeRunner:
        def __init__(self, env, cfg, *, device) -> None:
            assert isinstance(env, FakeWrapper)
            assert cfg == {}
            assert device == "cpu"

        def load(self, path, *, load_cfg, strict, map_location) -> None:
            assert Path(path).name == "model_0.pt"
            assert load_cfg == {"actor": True}
            assert strict
            assert map_location == "cpu"
            events.append("policy_load")

        def get_inference_policy(self, *, device):
            assert device == "cpu"
            events.append("policy_ready")
            return "policy"

    expected_lateral = torch.tensor([-0.42, -0.14, 0.14, 0.42])

    def arrange(env, lane_spacing) -> None:
        assert lane_spacing == MODULE.RACE_LANE_SPACING
        events.append("arrange")
        env.forward.zero_()
        env.lateral.copy_(expected_lateral)
        env.yaw.zero_()
        env.heading[:] = torch.tensor([1.0, 0.0])

    def refresh(env) -> None:
        events.append("refresh")
        env.reward_heading.copy_(env.heading)

    checkpoint = tmp_path / "model_0.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(MODULE, "RslRlVecEnvWrapper", FakeWrapper)
    monkeypatch.setattr(MODULE, "load_runner_cls", lambda _task: FakeRunner)
    monkeypatch.setattr(MODULE, "asdict", lambda _cfg: {})
    monkeypatch.setattr(MODULE, "_arrange_race_start", arrange)
    monkeypatch.setattr(
        MODULE,
        "_install_race_corridor_visualizer",
        lambda _env, **_kwargs: events.append("corridor"),
    )
    monkeypatch.setattr(MODULE, "_refresh_manual_start_state", refresh)

    env, policy = MODULE._load_policy_then_arrange_start(
        base_env=base_env,
        agent_cfg=SimpleNamespace(clip_actions=1.0),
        task_id="task",
        checkpoint=checkpoint,
        device="cpu",
        recovery_montage=False,
        seed=0,
    )

    assert isinstance(env, FakeWrapper)
    assert policy == "policy"
    assert events == [
        "wrapper_reset",
        "policy_load",
        "policy_ready",
        "arrange",
        "corridor",
        "refresh",
    ]
    assert torch.equal(base_env.forward, torch.zeros(4))
    assert torch.allclose(base_env.lateral, expected_lateral)
    assert torch.equal(base_env.yaw, torch.zeros(4))
    assert torch.equal(base_env.heading, torch.tensor([[1.0, 0.0]] * 4))
    assert torch.equal(base_env.reward_heading, base_env.heading)


def test_manual_start_refresh_rebuilds_cached_observations() -> None:
    events: list[str] = []

    class Sim:
        def sense(self) -> None:
            events.append("sense")

    class ObservationManager:
        def reset(self, env_ids) -> None:
            assert torch.equal(env_ids, torch.arange(4))
            events.append("obs_reset")

        def compute(self, *, update_history):
            assert update_history
            events.append("obs_compute")
            return {"actor": torch.ones(4, 61)}

    env = SimpleNamespace(
        num_envs=4,
        device=torch.device("cpu"),
        sim=Sim(),
        observation_manager=ObservationManager(),
        obs_buf={"actor": torch.zeros(4, 61)},
    )

    MODULE._refresh_manual_start_state(env)

    assert events == ["sense", "obs_reset", "obs_compute"]
    assert torch.equal(env.obs_buf["actor"], torch.ones(4, 61))


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


def test_backroll_camera_can_follow_wrong_way_motion_before_policy_improves() -> None:
    state = MODULE.CameraFollowState(x_m=0.60, velocity_mps=0.0)

    for _ in range(200):
        state = MODULE._camera_follow_x(
            state,
            robot_x_m=-5.0,
            dt_s=0.02,
            minimum_x_m=None,
        )

    assert state.x_m < -0.60
    assert state.x_m > -5.0 + MODULE.RACE_CAMERA_LEAD_M


def test_follow_camera_continues_beyond_finish_line() -> None:
    state = MODULE.CameraFollowState(x_m=10.0, velocity_mps=0.0)

    after_finish = state
    for _ in range(100):
        after_finish = MODULE._camera_follow_x(after_finish, 14.0, 0.02)

    assert after_finish.x_m > MODULE.TARGET_DISTANCE_M
    assert after_finish.x_m < 14.0 + MODULE.RACE_CAMERA_LEAD_M


def test_recording_fps_preserves_real_simulation_time() -> None:
    assert MODULE._recording_fps(0.02, 4) == pytest.approx(12.5)


def test_canonical_video_defaults_to_clean_1080p_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["record_roll_sprint_policy.py", "model_0.pt", "race.mp4"],
    )

    args = MODULE._parse_args()
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert (args.width, args.height) == (1920, 1080)
    assert "_overlay_race_labels" not in source
    assert "ImageDraw" not in source


def test_ffmpeg_writes_hd_motion_at_constant_sixty_fps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {}

    def popen(command, *, stdin):
        observed["command"] = command
        observed["stdin"] = stdin
        return "writer"

    monkeypatch.setattr(MODULE.shutil, "which", lambda _name: "ffmpeg")
    monkeypatch.setattr(MODULE.subprocess, "Popen", popen)

    writer = MODULE._ffmpeg_writer(
        tmp_path / "race.mp4",
        width=1920,
        height=1080,
        input_fps=50.0,
        output_fps=60.0,
    )

    command = observed["command"]
    input_index = command.index("-i")
    assert writer == "writer"
    assert command[command.index("-s:v") + 1] == "1920x1080"
    assert command[command.index("-r") + 1] == "50"
    assert command[input_index + 1] == "pipe:0"
    assert command[command.index("-r", input_index) + 1] == "60"
    assert command[command.index("-crf") + 1] == "18"


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


def test_recovery_montage_is_arranged_after_policy_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    base_env = object()

    class FakeWrapper:
        def __init__(self, env, *, clip_actions) -> None:
            assert env is base_env
            assert clip_actions == 1.0
            events.append("wrapper_reset")

    class FakeRunner:
        def __init__(self, env, cfg, *, device) -> None:
            assert isinstance(env, FakeWrapper)
            assert cfg == {}
            assert device == "cpu"

        def load(self, *_args, **_kwargs) -> None:
            events.append("policy_load")

        def get_inference_policy(self, *, device):
            assert device == "cpu"
            return "policy"

    checkpoint = tmp_path / "model_0.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(MODULE, "RslRlVecEnvWrapper", FakeWrapper)
    monkeypatch.setattr(MODULE, "load_runner_cls", lambda _task: FakeRunner)
    monkeypatch.setattr(MODULE, "asdict", lambda _cfg: {})
    monkeypatch.setattr(
        MODULE,
        "_arrange_recovery_montage",
        lambda _env, *, seed: events.append(f"arrange_recovery_{seed}"),
    )
    monkeypatch.setattr(
        MODULE,
        "_refresh_manual_start_state",
        lambda _env: events.append("refresh"),
    )

    MODULE._load_policy_then_arrange_start(
        base_env=base_env,
        agent_cfg=SimpleNamespace(clip_actions=1.0),
        task_id="task",
        checkpoint=checkpoint,
        device="cpu",
        recovery_montage=True,
        seed=3,
    )

    assert events == [
        "wrapper_reset",
        "policy_load",
        "arrange_recovery_3",
        "refresh",
    ]


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


def test_five_robot_showcase_uses_aligned_safe_road_starts() -> None:
    origins = MODULE._race_lane_origins(
        5,
        MODULE.FIVE_RACER_LANE_SPACING,
        device="cpu",
        dtype=torch.float32,
    )

    assert torch.equal(origins[:, 0], torch.zeros(5))
    assert origins[:, 1].tolist() == pytest.approx([-0.42, -0.21, 0.0, 0.21, 0.42])
    assert origins[:, 1].abs().max().item() <= MODULE.ROAD_SAFE_FULL_REWARD_HALF_WIDTH_M


def test_five_robot_corridor_has_five_visible_lanes() -> None:
    segments = MODULE._race_corridor_segments(
        num_lanes=5,
        lane_spacing=MODULE.FIVE_RACER_LANE_SPACING,
    )
    longitudinal = [
        (start, end) for start, end, _color, _radius in segments if start[1] == end[1]
    ]

    assert len(longitudinal) == 6
    assert [start[1] for start, _end in longitudinal] == pytest.approx(
        [-0.56, -0.315, -0.105, 0.105, 0.315, 0.56]
    )


def test_showcase_fireworks_latch_only_valid_ten_meter_frontier() -> None:
    state = MODULE.FinishCelebrationState(5)
    state.update(torch.tensor([19.0, 9.99, 10.0, 2.0, 11.0]), elapsed_s=12.0)

    # Raw position is never passed here. Only the credited frontier can trigger.
    assert state.finish_times_s == [12.0, None, 12.0, None, 12.0]

    state.update(torch.tensor([20.0, 10.0, 12.0, 10.0, 13.0]), elapsed_s=13.5)
    assert state.finish_times_s == [12.0, 13.5, 12.0, 13.5, 12.0]


def test_showcase_draws_finish_arch_and_animated_celebration() -> None:
    state = MODULE.FinishCelebrationState(5)
    state.update(torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0]), elapsed_s=20.0)
    state.current_time_s = 20.4
    lane_centers = MODULE.np.array([-0.42, -0.21, 0.0, 0.21, 0.42])

    arch = MODULE._finish_arch_segments()
    celebration = MODULE._finish_celebration_segments(state, lane_centers)

    assert len(arch) == 13
    assert len(celebration) == 32
    assert all(segment[3] > 0.0 for segment in celebration)


def test_five_robot_showcase_stops_after_third_finish_and_short_hold() -> None:
    state = MODULE.FinishCelebrationState(5)
    state.update(torch.tensor([0.0, 10.0, 10.0, 0.0, 0.0]), elapsed_s=21.9)
    state.arm_stop_after_finishers(
        MODULE.SHOWCASE_FINISHER_TARGET,
        MODULE.SHOWCASE_POST_FINISH_HOLD_SECONDS,
    )
    assert state.finished_count == 2
    assert state.stop_time_s is None

    state.update(torch.tensor([0.0, 10.0, 10.0, 10.0, 0.0]), elapsed_s=23.24)
    state.arm_stop_after_finishers(
        MODULE.SHOWCASE_FINISHER_TARGET,
        MODULE.SHOWCASE_POST_FINISH_HOLD_SECONDS,
    )
    assert state.finished_count == 3
    assert state.stop_time_s == pytest.approx(24.74)
    assert not state.stop_due

    state.update(torch.tensor([10.0] * 5), elapsed_s=24.74)
    state.arm_stop_after_finishers(
        MODULE.SHOWCASE_FINISHER_TARGET,
        MODULE.SHOWCASE_POST_FINISH_HOLD_SECONDS,
    )
    assert state.finished_count == 5
    assert state.stop_time_s == pytest.approx(24.74)
    assert state.stop_due
