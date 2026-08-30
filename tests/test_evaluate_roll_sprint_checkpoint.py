from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_roll_sprint_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_roll_sprint_checkpoint", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_canonical_evaluation_defaults_to_twenty_seconds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model_0.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), str(checkpoint)])

    args = MODULE._parse_args()

    assert MODULE.CANONICAL_RACE_DURATION_S == pytest.approx(20.0)
    assert args.duration == pytest.approx(20.0)


def test_canonical_evaluation_rejects_forty_second_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model_0.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT_PATH), str(checkpoint), "--duration", "40"],
    )

    with pytest.raises(SystemExit, match="--duration 20"):
        MODULE.main()


def test_evaluator_arranges_and_refreshes_after_wrapper_reset_and_policy_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    base_env = SimpleNamespace(state="initial")

    class FakeWrapper:
        def __init__(self, env, *, clip_actions) -> None:
            assert env is base_env
            assert clip_actions == 1.0
            env.state = "scrambled_by_wrapper_reset"
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

    expected_origins = torch.tensor(
        [
            [0.0, -0.42, 0.0],
            [0.0, -0.14, 0.0],
            [0.0, 0.14, 0.0],
            [0.0, 0.42, 0.0],
        ]
    )

    def arrange(env, lane_spacing):
        assert env.state == "scrambled_by_wrapper_reset"
        assert lane_spacing == MODULE.RACE_LANE_SPACING
        env.state = "canonical"
        events.append("arrange")
        return expected_origins

    def refresh(env) -> None:
        assert env.state == "canonical"
        events.append("refresh")

    checkpoint = tmp_path / "model_0.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(MODULE, "RslRlVecEnvWrapper", FakeWrapper)
    monkeypatch.setattr(MODULE, "load_runner_cls", lambda _task: FakeRunner)
    monkeypatch.setattr(MODULE, "asdict", lambda _cfg: {})
    monkeypatch.setattr(
        MODULE.microduck_mdp,
        "arrange_roll_sprint_race_start",
        arrange,
    )
    monkeypatch.setattr(MODULE, "_refresh_manual_start_state", refresh)

    env, policy, origins = MODULE._load_policy_then_arrange_race(
        base_env=base_env,
        agent_cfg=SimpleNamespace(clip_actions=1.0),
        checkpoint=checkpoint,
        device="cpu",
    )

    assert isinstance(env, FakeWrapper)
    assert policy == "policy"
    assert torch.equal(origins, expected_origins)
    assert events == ["wrapper_reset", "policy_load", "arrange", "refresh"]
    assert base_env.state == "canonical"


def test_canonical_alignment_checks_actual_state_and_reward_projection() -> None:
    lateral = torch.tensor([-0.42, -0.14, 0.14, 0.42])
    origins = torch.stack((torch.zeros(4), lateral, torch.zeros(4)), dim=-1)
    headings = torch.tensor([[1.0, 0.0]] * 4)
    centers = torch.zeros(4, 2)
    kwargs = {
        "forward_starts": torch.zeros(4),
        "lateral_starts": lateral,
        "race_origins": origins,
        "reward_headings": headings,
        "body_yaws": torch.zeros(4),
        "reward_forward_origins": torch.zeros(4),
        "reward_course_lateral": lateral,
        "reward_course_centers": centers,
    }

    assert MODULE._canonical_race_alignment_pass(**kwargs)

    wrong_lateral = {**kwargs, "lateral_starts": lateral + 0.01}
    assert not MODULE._canonical_race_alignment_pass(**wrong_lateral)

    wrong_heading = {**kwargs, "reward_headings": headings.roll(1, dims=1)}
    assert not MODULE._canonical_race_alignment_pass(**wrong_heading)

    wrong_reward_origin = {
        **kwargs,
        "reward_forward_origins": torch.full((4,), 0.01),
    }
    assert not MODULE._canonical_race_alignment_pass(**wrong_reward_origin)


def test_recovery_case_refreshes_after_reset_and_final_arrangement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    terrain = SimpleNamespace(env_origins=torch.full((4, 3), 9.0))
    base_env = SimpleNamespace(scene=SimpleNamespace(terrain=terrain))

    class FakeEnv:
        def reset(self) -> None:
            events.append("reset")
            terrain.env_origins.fill_(7.0)

    expected_lateral = torch.tensor([-0.42, -0.14, 0.14, 0.42])

    def arrange(env, lane_spacing, *, seed, orientations) -> None:
        assert env is base_env
        assert lane_spacing == MODULE.RACE_LANE_SPACING
        assert seed == 3
        assert orientations == MODULE.RECOVERY_ORIENTATIONS
        assert torch.equal(terrain.env_origins, torch.full((4, 3), 7.0))
        events.append("arrange")
        terrain.env_origins.zero_()
        terrain.env_origins[:, 1].copy_(expected_lateral)

    def refresh(env) -> None:
        assert env is base_env
        assert torch.allclose(terrain.env_origins[:, 1], expected_lateral)
        events.append("refresh")

    monkeypatch.setattr(
        MODULE.microduck_mdp,
        "arrange_roll_sprint_recovery_start",
        arrange,
    )
    monkeypatch.setattr(MODULE, "_refresh_manual_start_state", refresh)

    initial_lateral = MODULE._reset_and_arrange_recovery_case(
        base_env=base_env,
        env=FakeEnv(),
        seed=3,
    )

    assert events == ["reset", "arrange", "refresh"]
    assert torch.allclose(initial_lateral, expected_lateral)


def _auditor(initial_course_y: float = 0.0) -> object:
    return MODULE.RollCycleAuditor(
        initial_position_xy=torch.tensor([[0.0, initial_course_y]]),
        initial_root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        initial_vertical_velocity=torch.zeros(1),
        step_dt=0.02,
        course_center_xy=torch.zeros(2),
    )


def _observe(
    auditor,
    *,
    omega: float,
    forward_position: float = 0.0,
    lateral_position: float = 0.0,
    forward_velocity: float = 0.0,
    support: bool = True,
    foot_support: bool = True,
    head_contact: bool = False,
    head_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    root_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> None:
    auditor.observe(
        position_xy=torch.tensor([[forward_position, lateral_position]]),
        root_quat=torch.tensor([root_quat]),
        head_quat=torch.tensor([head_quat]),
        linear_velocity_w=torch.tensor([[forward_velocity, 0.0, 0.0]]),
        angular_velocity_b=torch.tensor([[0.0, omega, 0.0]]),
        support=torch.tensor([support]),
        foot_support=torch.tensor([foot_support]),
        head_contact=torch.tensor([head_contact]),
    )


def _recover(auditor, *, forward_position: float) -> None:
    for _ in range(MODULE.RECOVERY_HOLD_STEPS):
        _observe(
            auditor,
            omega=0.0,
            forward_position=forward_position,
            foot_support=True,
            head_contact=False,
        )


def _yaw_quat(degrees: float) -> tuple[float, float, float, float]:
    half = math.radians(degrees) / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _pitched_yaw_quat(degrees: float) -> tuple[float, float, float, float]:
    yaw_half = math.radians(degrees) / 2.0
    pitch_half = math.pi / 4.0
    cy, sy = math.cos(yaw_half), math.sin(yaw_half)
    cp, sp = math.cos(pitch_half), math.sin(pitch_half)
    return (cy * cp, -sy * sp, cy * sp, sy * cp)


def test_shared_road_thresholds_match_canonical_geometry() -> None:
    assert MODULE.ROAD_HALF_WIDTH_M == pytest.approx(0.56)
    assert MODULE.ROAD_SAFE_FULL_REWARD_HALF_WIDTH_M == pytest.approx(0.42)
    assert MODULE.ROAD_REPOSITION_TRIGGER_M == pytest.approx(0.50)
    assert MODULE.ROAD_REPOSITION_REARM_M == pytest.approx(0.46)
    assert MODULE.ROAD_MAX_YAW_DEVIATION_DEG == pytest.approx(20.0)


def test_auditor_backward_rocking_cannot_complete_or_repay_angle() -> None:
    auditor = _auditor()

    for _ in range(80):
        _observe(auditor, omega=5.0)
        _observe(auditor, omega=-5.0)

    assert auditor.accum.item() == pytest.approx(0.0, abs=1.0e-6)
    assert auditor.valid_count.item() == 0
    assert auditor.invalid_count.item() == 0


def test_auditor_missing_head_contact_marks_full_cycle_invalid() -> None:
    auditor = _auditor()
    auditor.accum[:] = MODULE.TARGET_ANGLE - 0.01

    _observe(auditor, omega=1.0)

    assert auditor.valid_count.item() == 0
    assert auditor.invalid_count.item() == 1
    assert auditor.linked_distance.item() == 0.0


def test_auditor_releases_only_rotation_capped_distance_on_valid_cycle() -> None:
    auditor = _auditor()
    auditor.accum[:] = 1.0
    auditor.head_latch[:] = True
    auditor.accum[:] = MODULE.TARGET_ANGLE - 0.02

    _observe(auditor, omega=2.0, forward_position=100.0)

    expected_cap = MODULE.MAX_DISTANCE_PER_RAD * MODULE.TARGET_ANGLE
    assert auditor.valid_count.item() == 1
    assert auditor.linked_distance.item() == pytest.approx(expected_cap)


def test_auditor_requires_shared_road_reposition_before_roll_restart() -> None:
    auditor = _auditor()
    _observe(
        auditor,
        omega=1.0,
        lateral_position=MODULE.ROAD_REPOSITION_TRIGGER_M + 0.01,
    )
    assert auditor.awaiting_reposition.item()

    auditor.accum[:] = MODULE.TARGET_ANGLE - 0.01
    auditor.head_latch[:] = True
    _observe(
        auditor,
        omega=1.0,
        forward_position=0.35,
        lateral_position=MODULE.ROAD_REPOSITION_TRIGGER_M + 0.01,
    )

    assert auditor.valid_count.item() == 0
    assert auditor.invalid_count.item() == 0
    assert auditor.linked_distance.item() == 0.0
    assert auditor.forward_frontier.item() == 0.0

    for _ in range(MODULE.RECOVERY_HOLD_STEPS):
        _observe(
            auditor,
            omega=0.0,
            forward_position=0.35,
            lateral_position=MODULE.ROAD_REPOSITION_REARM_M + 0.01,
        )
    assert auditor.awaiting_reposition.item()

    _recover(auditor, forward_position=0.35)
    assert not auditor.awaiting_reposition.item()
    assert auditor.reposition_count.item() == 1
    assert auditor.recovery_count.item() == 0

    auditor.accum[:] = MODULE.TARGET_ANGLE - 0.01
    auditor.head_latch[:] = True
    _observe(auditor, omega=1.0, forward_position=0.55)
    assert auditor.valid_count.item() == 1
    assert auditor.linked_distance.item() == pytest.approx(0.20)
    report = auditor.summary(6.0)
    assert report["mean_lane_reposition_count"] == pytest.approx(1.0)
    assert report["mean_lane_reposition_latency_s"] == pytest.approx(
        (2 * MODULE.RECOVERY_HOLD_STEPS + 1) * 0.02
    )


def test_auditor_allows_crossing_original_lanes_inside_shared_road() -> None:
    auditor = _auditor(initial_course_y=-0.42)
    auditor.accum[:] = MODULE.TARGET_ANGLE - 0.01
    auditor.head_latch[:] = True

    _observe(
        auditor,
        omega=1.0,
        forward_position=0.30,
        lateral_position=0.42,
    )

    assert auditor.valid_count.item() == 1
    assert auditor.invalid_count.item() == 0
    assert auditor.linked_distance.item() == pytest.approx(0.30)
    assert auditor.max_abs_course_lateral.item() == pytest.approx(0.42)
    assert auditor.max_road_boundary_overshoot.item() == 0.0


def test_auditor_shared_road_boundary_is_inclusive_and_outer_exit_fails() -> None:
    boundary = _auditor()
    _observe(
        boundary,
        omega=1.0,
        lateral_position=MODULE.ROAD_HALF_WIDTH_M,
    )
    assert not boundary.lateral_invalid.item()
    assert boundary.road_exit_steps.item() == 0

    outside = _auditor()
    _observe(
        outside,
        omega=1.0,
        lateral_position=(
            MODULE.ROAD_HALF_WIDTH_M + MODULE.ROAD_BOUNDARY_TOLERANCE_M + 1.0e-4
        ),
    )
    assert outside.awaiting_reposition.item()
    assert outside.road_exit_steps.item() == 1
    assert outside.max_road_boundary_overshoot.item() > 0.0


def test_auditor_requires_heading_reposition_before_roll_restart() -> None:
    auditor = _auditor()
    auditor.accum[:] = 1.25

    _observe(auditor, omega=0.0, root_quat=_yaw_quat(21.0))

    assert auditor.awaiting_reposition.item()
    assert auditor.accum.item() == 0.0
    for _ in range(MODULE.RECOVERY_HOLD_STEPS):
        _observe(auditor, omega=0.0, root_quat=_yaw_quat(11.0))
    assert auditor.awaiting_reposition.item()
    assert auditor.reposition_count.item() == 0

    for _ in range(MODULE.RECOVERY_HOLD_STEPS):
        _observe(auditor, omega=0.0, root_quat=_yaw_quat(0.0))
    assert not auditor.awaiting_reposition.item()
    assert auditor.reposition_count.item() == 1


def test_auditor_heading_violation_invalidates_active_cycle() -> None:
    auditor = _auditor()
    auditor.accum[:] = MODULE.TARGET_ANGLE - 0.01
    auditor.head_latch[:] = True

    _observe(
        auditor,
        omega=1.0,
        forward_position=0.30,
        root_quat=_pitched_yaw_quat(21.0),
    )

    assert auditor.valid_count.item() == 0
    assert auditor.invalid_count.item() == 1
    assert auditor.linked_distance.item() == 0.0


def test_auditor_completion_outside_heading_rearm_starts_reposition() -> None:
    auditor = _auditor()
    auditor.accum[:] = MODULE.TARGET_ANGLE - 0.01
    auditor.head_latch[:] = True

    _observe(
        auditor,
        omega=1.0,
        forward_position=0.20,
        root_quat=_yaw_quat(15.0),
    )

    assert auditor.valid_count.item() == 1
    assert auditor.awaiting_recovery.item()
    assert auditor.awaiting_reposition.item()
    _recover(auditor, forward_position=0.20)
    assert auditor.recovery_count.item() == 1
    assert auditor.reposition_count.item() == 1


def test_auditor_uses_net_cycle_advance_and_global_frontier() -> None:
    auditor = _auditor()
    _observe(auditor, omega=1.0, forward_position=0.40)
    auditor.accum[:] = MODULE.TARGET_ANGLE - 0.01
    auditor.head_latch[:] = True
    _observe(auditor, omega=1.0, forward_position=0.20)
    assert auditor.linked_distance.item() == pytest.approx(0.20)

    _recover(auditor, forward_position=0.20)
    auditor.accum[:] = MODULE.TARGET_ANGLE - 0.01
    auditor.head_latch[:] = True
    _observe(auditor, omega=1.0, forward_position=0.20)
    assert auditor.linked_distance.item() == pytest.approx(0.20)

    _recover(auditor, forward_position=0.20)
    auditor.accum[:] = MODULE.TARGET_ANGLE - 0.01
    auditor.head_latch[:] = False
    _observe(auditor, omega=1.0, forward_position=0.0)
    assert auditor.invalid_count.item() == 1

    auditor.accum[:] = MODULE.TARGET_ANGLE - 0.01
    auditor.head_latch[:] = True
    _observe(auditor, omega=1.0, forward_position=0.35)
    assert auditor.linked_distance.item() == pytest.approx(0.35)


def test_auditor_requires_recovery_before_second_credited_roll() -> None:
    auditor = _auditor()
    auditor.accum[:] = MODULE.TARGET_ANGLE - 0.01
    auditor.head_latch[:] = True
    _observe(
        auditor,
        omega=1.0,
        forward_position=0.20,
        forward_velocity=1.25,
    )

    auditor.accum[:] = MODULE.TARGET_ANGLE - 0.01
    auditor.head_latch[:] = True
    _observe(auditor, omega=1.0, forward_position=0.40)
    assert auditor.valid_count.item() == 1
    assert auditor.linked_distance.item() == pytest.approx(0.20)

    _recover(auditor, forward_position=0.20)
    auditor.accum[:] = MODULE.TARGET_ANGLE - 0.01
    auditor.head_latch[:] = True
    _observe(auditor, omega=1.0, forward_position=0.45)

    assert auditor.valid_count.item() == 2
    assert auditor.recovery_count.item() == 1
    assert auditor.recovered_and_rerolled_count.item() == 1
    report = auditor.summary(6.0)
    assert report["mean_recovery_latency_s"] == pytest.approx(
        MODULE.RECOVERY_HOLD_STEPS * 0.02
    )
    assert report["repeated_roll_rate"] == pytest.approx(1.0)
    assert report["maximum_forward_speed_mps"] == pytest.approx(1.25)
    assert report["per_robot"][0]["maximum_forward_speed_mps"] == pytest.approx(1.25)
    assert not report["per_robot"][0]["target_10m_pass"]
    assert report["target_distance_reach_rate"] == pytest.approx(0.0)
    assert not report["four_robot_batch_target_10m_pass"]
    assert report["recovery_gate_diagnostics"] == {
        "awaiting_steps": MODULE.RECOVERY_HOLD_STEPS,
        "foot_supported_head_released_steps": MODULE.RECOVERY_HOLD_STEPS,
        "upright_ready_steps": MODULE.RECOVERY_HOLD_STEPS,
        "sagittal_ready_steps": MODULE.RECOVERY_HOLD_STEPS,
        "rate_ready_steps": MODULE.RECOVERY_HOLD_STEPS,
        "candidate_steps": MODULE.RECOVERY_HOLD_STEPS,
        "max_consecutive_candidate_steps": MODULE.RECOVERY_HOLD_STEPS,
        "foot_release_fraction": 1.0,
        "upright_given_foot_release_fraction": 1.0,
        "sagittal_given_upright_fraction": 1.0,
        "rate_given_sagittal_fraction": 1.0,
    }


def test_auditor_stalled_fall_discards_partial_cycle_before_full_reroll() -> None:
    auditor = _auditor()
    auditor.accum[:] = MODULE.TARGET_ANGLE - 0.01
    auditor.head_latch[:] = True
    _observe(auditor, omega=1.0, forward_position=0.10)
    _recover(auditor, forward_position=0.10)

    for index in range(5):
        _observe(
            auditor,
            omega=5.0,
            forward_position=0.10 + 0.04 * (index + 1),
        )
    assert auditor.accum.item() == pytest.approx(0.50)
    assert auditor.phase_frontier.item() == pytest.approx(0.50)

    fallen_quat = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)
    stall_steps = max(
        1,
        round(MODULE.SELF_RIGHT_STALL_SECONDS / auditor.step_dt),
    )
    for _ in range(stall_steps):
        _observe(
            auditor,
            omega=0.0,
            forward_position=0.30,
            support=True,
            foot_support=False,
            root_quat=fallen_quat,
        )

    assert auditor.self_righting.item()
    assert auditor.awaiting_recovery.item()
    assert auditor.accum.item() == 0.0
    assert auditor.phase_frontier.item() == 0.0
    assert auditor.forward_frontier.item() == pytest.approx(0.10)
    assert auditor.linked_distance.item() == pytest.approx(0.10)

    _observe(
        auditor,
        omega=5.0,
        forward_position=0.45,
        support=True,
        foot_support=False,
        head_contact=True,
        root_quat=fallen_quat,
    )
    assert auditor.accum.item() == 0.0
    assert auditor.phase_frontier.item() == 0.0
    assert auditor.valid_count.item() == 1
    assert auditor.forward_frontier.item() == pytest.approx(0.10)

    _recover(auditor, forward_position=0.45)
    assert not auditor.self_righting.item()
    assert not auditor.awaiting_recovery.item()
    assert auditor.recovery_count.item() == 2
    assert auditor.cycle_start_forward.item() == pytest.approx(0.45)

    auditor.accum[:] = MODULE.TARGET_ANGLE - 0.01
    auditor.phase_frontier[:] = auditor.accum
    auditor.head_latch[:] = True
    _observe(auditor, omega=1.0, forward_position=0.75)

    assert auditor.valid_count.item() == 2
    assert auditor.recovered_and_rerolled_count.item() == 1
    assert auditor.linked_distance.item() == pytest.approx(0.40)
    assert auditor.forward_frontier.item() == pytest.approx(0.75)


def test_evaluator_ranks_standing_on_road_robot_by_credited_frontier() -> None:
    auditor = MODULE.RollCycleAuditor(
        initial_position_xy=torch.tensor([[0.0, -0.14], [0.0, 0.14]]),
        initial_root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2),
        initial_vertical_velocity=torch.zeros(2),
        step_dt=0.02,
        course_center_xy=torch.zeros(2),
    )
    auditor.last_position[:] = torch.tensor([[10.0, -0.10], [10.5, 0.10]])
    auditor.final_course_lateral[:] = torch.tensor([-0.10, 0.10])
    auditor.linked_distance[:] = torch.tensor([10.0, 9.9])
    auditor.launch_ready[:] = True

    report = auditor.summary(20.0)

    assert report["standing_on_road_winner_robot_index"] == 0
    assert report["target_distance_m"] == pytest.approx(10.0)
    assert report["per_robot"][0]["standing_on_road_rank"] == 1
    assert report["per_robot"][0]["target_10m_pass"]
    assert report["per_robot"][0]["standing_on_road_finish_pass"]
    assert report["per_robot"][1]["standing_on_road_rank"] == 2
    assert not report["per_robot"][1]["target_10m_pass"]
    assert not report["per_robot"][1]["standing_on_road_finish_pass"]


def test_ten_meter_finish_boundary_is_scored_at_twenty_seconds() -> None:
    lane_centers = torch.tensor([-0.42, -0.14, 0.14, 0.42])
    auditor = MODULE.RollCycleAuditor(
        initial_position_xy=torch.stack((torch.zeros(4), lane_centers), dim=-1),
        initial_root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4),
        initial_vertical_velocity=torch.zeros(4),
        step_dt=0.02,
        course_center_xy=torch.zeros(2),
    )
    auditor.last_position[:] = torch.stack(
        (torch.tensor([10.0, 10.0, 10.0, 9.99]), lane_centers), dim=-1
    )
    auditor.final_course_lateral[:] = lane_centers
    auditor.linked_distance[:] = torch.tensor([10.0, 10.0, 10.0, 9.99])
    auditor.launch_ready[:] = True

    report = auditor.summary(MODULE.CANONICAL_RACE_DURATION_S)

    assert report["target_distance_reach_rate"] == pytest.approx(0.75)
    assert report["standing_on_road_target_reach_rate"] == pytest.approx(0.75)
    assert [robot["target_10m_pass"] for robot in report["per_robot"]] == [
        True,
        True,
        True,
        False,
    ]
    assert report["mean_roll_linked_speed_mps"] == pytest.approx(39.99 / 4 / 20)


def test_heading_uses_lateral_axis_at_vertical_pitch() -> None:
    half = math.sqrt(0.5)
    heading = MODULE.heading_from_quat(
        torch.tensor([[half, 0.0, half, 0.0]], dtype=torch.float32)
    )

    assert heading[0, 0].item() == pytest.approx(1.0, abs=1.0e-6)
    assert heading[0, 1].item() == pytest.approx(0.0, abs=1.0e-6)


def test_absolute_race_goal_requires_every_gate() -> None:
    assert MODULE.TARGET_DISTANCE_M == pytest.approx(10.0)
    assert MODULE.PROMOTION["mean_valid_roll_count"] == pytest.approx(14.0)
    assert MODULE.PROMOTION["mean_recovered_and_rerolled_count"] == pytest.approx(13.0)
    report = {
        **MODULE.PROMOTION,
        "road_exit_env_count": 0,
        "nan_env_count": 0,
        "out_of_bounds_env_count": 0,
    }
    assert MODULE.absolute_race_goal_pass(report)

    for key in MODULE.PROMOTION:
        failing = dict(report)
        if key in {
            "maximum_road_boundary_overshoot_m",
            "mean_uncredited_positive_displacement_m",
        }:
            failing[key] = float(report[key]) + 0.001
        else:
            failing[key] = float(report[key]) - 0.001
        assert not MODULE.absolute_race_goal_pass(failing), key

    for key in ("road_exit_env_count", "nan_env_count", "out_of_bounds_env_count"):
        failing = dict(report)
        failing[key] = 1
        assert not MODULE.absolute_race_goal_pass(failing), key


def _recovery_case(
    orientation: str,
    seed: int,
    *,
    success: bool = True,
    rerolled: bool = True,
    latency_s: float | None = 2.0,
) -> dict[str, object]:
    return {
        "orientation": orientation,
        "seed": seed,
        "success": success,
        "recovery_latency_s": latency_s if success else None,
        "self_right_then_reroll": rerolled if success else False,
        "frontier_after_recovery_m": 0.25 if rerolled else 0.0,
        "lane_reposition_count": 1,
        "lane_reposition_latency_s": 0.4,
        "maximum_lateral_drift_m": 0.10,
        "maximum_road_boundary_overshoot_m": 0.0,
        "road_exit_steps": 0,
        "nan_seen": False,
        "out_of_bounds": False,
    }


def _race_report() -> dict[str, object]:
    return {
        "mean_credited_forward_frontier_m": 10.0,
        "four_robot_batch_road_corridor_pass": True,
        "nan_env_count": 0,
        "out_of_bounds_env_count": 0,
    }


def test_recovery_battery_reports_sixteen_orientation_seed_cases() -> None:
    cases = [
        _recovery_case(orientation, seed)
        for orientation in MODULE.RECOVERY_ORIENTATIONS
        for seed in MODULE.RECOVERY_SEEDS
    ]

    report = MODULE.summarize_recovery_battery(
        cases,
        race_report=_race_report(),
        parent_frontier_m=9.5,
    )

    assert report["total_attempts"] == 16
    assert report["total_successes"] == 16
    assert report["success_rate"] == pytest.approx(1.0)
    assert report["self_right_then_reroll_count"] == 16
    assert report["self_right_then_reroll_rate"] == pytest.approx(1.0)
    assert report["frontier_after_recovery_m"] == pytest.approx(4.0)
    assert report["lane_reposition_count"] == 16
    assert report["lane_reposition_latency_mean_s"] == pytest.approx(0.4)
    assert report["road_exit_case_count"] == 0
    assert report["maximum_road_boundary_overshoot_m"] == 0.0
    assert report["race_frontier_ratio_to_parent"] == pytest.approx(10 / 9.5)
    assert report["race_frontier_delta_to_parent_m"] == pytest.approx(0.5)
    assert report["race_frontier_improved_over_parent"]
    assert report["overall_pass"]
    assert set(report["by_orientation"]) == set(MODULE.RECOVERY_ORIENTATIONS)
    assert all(
        orientation_report["attempts"] == 4
        and orientation_report["successes"] == 4
        and orientation_report["pass"]
        for orientation_report in report["by_orientation"].values()
    )


def test_recovery_battery_fails_weak_orientation_and_parent_regression() -> None:
    cases = [
        _recovery_case(
            orientation,
            seed,
            success=not (orientation == "left" and seed >= 1),
            rerolled=not (orientation == "left" and seed >= 1),
        )
        for orientation in MODULE.RECOVERY_ORIENTATIONS
        for seed in MODULE.RECOVERY_SEEDS
    ]

    report = MODULE.summarize_recovery_battery(
        cases,
        race_report=_race_report(),
        parent_frontier_m=12.0,
    )

    assert report["by_orientation"]["left"]["success_rate"] == pytest.approx(0.25)
    assert not report["by_orientation"]["left"]["pass"]
    assert not report["race_frontier_at_least_90pct_parent"]
    assert not report["race_frontier_improved_over_parent"]
    assert not report["overall_pass"]


def test_recovery_battery_rejects_any_shared_road_exit() -> None:
    cases = [
        _recovery_case(orientation, seed)
        for orientation in MODULE.RECOVERY_ORIENTATIONS
        for seed in MODULE.RECOVERY_SEEDS
    ]
    cases[0]["maximum_road_boundary_overshoot_m"] = 0.01
    cases[0]["road_exit_steps"] = 1

    report = MODULE.summarize_recovery_battery(
        cases,
        race_report=_race_report(),
        parent_frontier_m=9.5,
    )

    assert report["road_exit_case_count"] == 1
    assert report["maximum_road_boundary_overshoot_m"] == pytest.approx(0.01)
    assert not report["overall_pass"]


def test_incremental_promotion_requires_strict_frontier_improvement() -> None:
    cases = [
        _recovery_case(orientation, seed)
        for orientation in MODULE.RECOVERY_ORIENTATIONS
        for seed in MODULE.RECOVERY_SEEDS
    ]

    report = MODULE.summarize_recovery_battery(
        cases,
        race_report=_race_report(),
        parent_frontier_m=10.0,
    )

    assert report["race_frontier_delta_to_parent_m"] == pytest.approx(0.0)
    assert report["race_frontier_at_least_90pct_parent"]
    assert not report["race_frontier_improved_over_parent"]
    assert not report["overall_pass"]
