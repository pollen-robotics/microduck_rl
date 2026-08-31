import math
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_roll_sprint_env_cfg import (
    EPISODE_LENGTH_S,
    TARGET_DISTANCE_M,
    MicroduckRollSprintRlCfg,
    make_microduck_roll_sprint_env_cfg,
)
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    COM_RANDOMIZATION_RANGE,
    HEAD_COM_RANDOMIZATION_RANGE,
    make_microduck_roulade_env_cfg,
)


class _FakeScene:
    def __init__(self, asset, sensors, origins):
        self.robot = asset
        self.sensors = sensors
        self.terrain = SimpleNamespace(env_origins=origins)

    def __getitem__(self, name):
        assert name == "robot"
        return self.robot


def _fake_env(num_envs: int = 1):
    identity = torch.zeros(num_envs, 4)
    identity[:, 0] = 1.0
    data = SimpleNamespace(
        root_link_ang_vel_b=torch.zeros(num_envs, 3),
        root_link_lin_vel_b=torch.zeros(num_envs, 3),
        root_link_lin_vel_w=torch.zeros(num_envs, 3),
        root_link_pos_w=torch.zeros(num_envs, 3),
        root_link_quat_w=identity,
    )
    asset = SimpleNamespace(data=data)
    sensors = {
        "robot_ground_contact": SimpleNamespace(
            data=SimpleNamespace(found=torch.ones(num_envs, 1))
        ),
        "head_ground_contact": SimpleNamespace(
            data=SimpleNamespace(found=torch.zeros(num_envs, 1))
        ),
        "feet_ground_contact": SimpleNamespace(
            data=SimpleNamespace(found=torch.ones(num_envs, 1))
        ),
    }
    env = SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        step_dt=0.02,
        common_step_counter=0,
        scene=_FakeScene(asset, sensors, torch.zeros(num_envs, 3)),
    )
    return env, asset


def _enable_flat_valid_roll(monkeypatch, env) -> None:
    monkeypatch.setattr(
        mdp,
        "_lateral_axis_z",
        lambda quat: torch.zeros(env.num_envs),
    )
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda env, asset: torch.ones(env.num_envs, dtype=torch.bool),
    )


def _prime_roll_heading(env, asset) -> None:
    asset.data.root_link_ang_vel_b[:] = 0.0
    mdp._update_roll_sprint_state(env, asset)
    env.common_step_counter += 1


def _complete_valid_roll(env, asset, *, forward: float, lateral: float = 0.0) -> None:
    asset.data.root_link_pos_w[:, 0] = forward
    asset.data.root_link_pos_w[:, 1] = lateral
    asset.data.root_link_ang_vel_b[:, 1] = 1.0
    env._roll_sprint_accum[:] = 2.0 * torch.pi - 0.01
    env._roll_sprint_phase_frontier[:] = env._roll_sprint_accum
    env._roll_sprint_head_latch[:] = True
    env._roll_sprint_lateral_invalid[:] = False
    mdp._update_roll_sprint_state(env, asset)


def _recover(env, asset, *, yaw_radians: float = 0.0) -> None:
    env.scene.sensors["head_ground_contact"].data.found[:] = 0.0
    env.scene.sensors["feet_ground_contact"].data.found[:] = 1.0
    asset.data.root_link_quat_w.zero_()
    asset.data.root_link_quat_w[:, 0] = math.cos(0.5 * yaw_radians)
    asset.data.root_link_quat_w[:, 3] = math.sin(0.5 * yaw_radians)
    asset.data.root_link_pos_w[:, 2] = 0.115
    asset.data.root_link_ang_vel_b[:] = 0.0
    for _ in range(mdp._ROLL_SPRINT_RECOVERY_HOLD_STEPS):
        env.common_step_counter += 1
        mdp._update_roll_sprint_state(env, asset)


def _set_yaw(asset, yaw_radians: torch.Tensor) -> None:
    asset.data.root_link_quat_w.zero_()
    asset.data.root_link_quat_w[:, 0] = torch.cos(0.5 * yaw_radians)
    asset.data.root_link_quat_w[:, 3] = torch.sin(0.5 * yaw_radians)


def test_roll_sprint_is_separate_long_distance_61d_policy():
    cfg = make_microduck_roll_sprint_env_cfg()
    roulade = make_microduck_roulade_env_cfg()

    assert cfg.episode_length_s == EPISODE_LENGTH_S == 40.0
    assert TARGET_DISTANCE_M == 10.0
    assert mdp._ROLL_SPRINT_RECOVERY_MAX_FORWARD_RATE == 6.0
    assert mdp._ROLL_SPRINT_RECOVERY_HOLD_STEPS == 5
    assert mdp._ROLL_SPRINT_RECOVERY_MIN_HEIGHT_M == 0.09
    assert mdp._ROLL_SPRINT_SELF_RIGHT_STALL_SECONDS == 0.30
    assert mdp._ROLL_SPRINT_ROAD_HALF_WIDTH == 0.56
    assert mdp._ROLL_SPRINT_ROAD_SAFE_HALF_WIDTH == 0.42
    assert mdp._ROLL_SPRINT_REPOSITION_TRIGGER_M == 0.50
    assert mdp._ROLL_SPRINT_REPOSITION_REARM_M == 0.46
    assert mdp._ROLL_SPRINT_REPOSITION_LATERAL_COMMAND_MPS == 0.20
    assert mdp._ROLL_SPRINT_REPOSITION_LATERAL_KP == 2.0
    assert mdp._ROLL_SPRINT_REPOSITION_HEADING_TRIGGER_RAD == pytest.approx(
        math.radians(20.0)
    )
    assert mdp._ROLL_SPRINT_REPOSITION_HEADING_REARM_RAD == pytest.approx(
        math.radians(10.0)
    )
    assert mdp._ROLL_SPRINT_REPOSITION_YAW_COMMAND_RPS == 0.05
    assert cfg.scene.terrain.terrain_type == "plane"
    assert list(cfg.observations["actor"].terms) == list(
        roulade.observations["actor"].terms
    )
    assert (
        cfg.observations["actor"].terms["command"].func
        is mdp.roll_sprint_reposition_command
    )
    assert (
        cfg.observations["critic"].terms["command"].func
        is mdp.roll_sprint_reposition_command
    )
    assert (
        cfg.observations["critic"].terms["roll_sprint_critic_padding"].params["dim"]
        == 16
    )
    assert cfg.actions["joint_pos"].scale == 1.0
    assert "roll_sprint_distance" in cfg.rewards
    assert (
        cfg.rewards["roll_sprint_distance"].weight
        > cfg.rewards["roll_sprint_progress"].weight
    )
    assert cfg.rewards["roll_sprint_progress"].params["road_half_width"] == (
        mdp._ROLL_SPRINT_ROAD_HALF_WIDTH
    )
    assert cfg.rewards["roll_sprint_cycle_rate"].weight == 1.0
    assert cfg.rewards["roll_sprint_recovery"].weight == 1.0
    assert cfg.rewards["roll_sprint_reposition"].weight == 2.0
    assert cfg.rewards["roll_sprint_recovered_reroll"].weight == 4.0
    assert cfg.rewards["roll_sprint_self_right_upright"].weight == 5.0
    assert cfg.rewards["roll_sprint_self_right_height"].weight == 30.0
    assert cfg.rewards["roll_sprint_self_right_upward"].weight == 1.0
    assert cfg.rewards["roll_sprint_self_right_fallen_tax"].weight == -0.25
    assert cfg.rewards["roll_sprint_self_right_success"].weight == 1.0
    assert (
        cfg.rewards["roll_sprint_recovered_reroll"].weight
        < cfg.rewards["roll_sprint_distance"].weight
    )
    assert cfg.rewards["roll_sprint_invalid_cycle"].weight == 0.0
    assert cfg.rewards["roll_sprint_sagittal"].weight == -0.05
    assert cfg.rewards["roll_sprint_flatness"].weight == -0.25
    assert cfg.rewards["roll_sprint_lateral_vel"].weight == -0.35
    assert cfg.rewards["roll_sprint_straightness"].weight == -3.0
    road_return_weight = cfg.rewards["roll_sprint_road_return"].weight
    distance_weight = cfg.rewards["roll_sprint_distance"].weight
    assert road_return_weight == 4.0
    assert road_return_weight < distance_weight
    full_edge_cost = (
        mdp._ROLL_SPRINT_ROAD_HALF_WIDTH
        - mdp._ROLL_SPRINT_ROAD_SAFE_HALF_WIDTH
    ) * road_return_weight
    maximum_valid_cycle_reward = (
        mdp._ROLL_SPRINT_TARGET_ANGLE
        * mdp._ROLL_SPRINT_MAX_DISTANCE_PER_RAD
        * distance_weight
    )
    assert full_edge_cost < 0.1 * maximum_valid_cycle_reward
    assert cfg.rewards["roll_sprint_heading_alignment"].weight == 1.0
    for name in (
        "roll_sprint_overspeed",
        "roll_sprint_impact",
        "body_ang_vel",
        "angular_momentum",
        "action_rate_l2",
        "joint_torque_rate_l2",
        "self_collisions",
    ):
        assert cfg.rewards[name].weight == 0.0
    assert not any(
        name.startswith("roulade_") or name in {"upright", "height", "pose"}
        for name in cfg.rewards
    )
    assert "set_roll_sprint_state" in cfg.events
    assert "set_roulade_state" not in cfg.events
    assert cfg.terminations["nan_state"].time_out is False
    assert MicroduckRollSprintRlCfg.experiment_name == "microduck_roll_sprint"
    assert MicroduckRollSprintRlCfg.save_interval == 100
    assert MicroduckRollSprintRlCfg.algorithm.entropy_coef == 0.0
    assert MicroduckRollSprintRlCfg.algorithm.clip_param == pytest.approx(0.1)
    assert MicroduckRollSprintRlCfg.algorithm.learning_rate == pytest.approx(2.5e-5)
    assert MicroduckRollSprintRlCfg.algorithm.desired_kl == pytest.approx(0.005)
    reset_params = cfg.events["set_roll_sprint_state"].params
    assert reset_params["standing_prob"] == 0.45
    assert reset_params["midroll_prob"] == 0.10
    assert reset_params["postroll_prob"] == 0.15
    assert reset_params["crouch_prob"] == 0.10
    assert reset_params["ground_recovery_prob"] == 0.20
    assert (
        reset_params["road_interior_prob"],
        reset_params["road_edge_prob"],
        reset_params["road_return_prob"],
    ) == (0.70, 0.20, 0.10)
    assert reset_params["recovery_road_return_prob"] == 0.35
    assert reset_params["heading_return_prob"] == 0.10
    assert reset_params["heading_return_min_rad"] == pytest.approx(math.radians(20.0))
    assert reset_params["heading_return_max_rad"] == pytest.approx(math.radians(40.0))
    assert tuple(
        reset_params[name]
        for name in (
            "ground_face_down_prob",
            "ground_face_up_prob",
            "ground_left_prob",
            "ground_right_prob",
        )
    ) == (0.25, 0.25, 0.25, 0.25)
    assert cfg.curriculum["roll_sprint_road_half_width"].params[
        "width_stages"
    ] == [{"step": 0, "width": 0.56}]
    play_cfg = make_microduck_roll_sprint_env_cfg(play=True)
    assert play_cfg.curriculum["roll_sprint_road_half_width"].params[
        "width_stages"
    ] == [{"step": 0, "width": 0.56}]
    assert cfg.curriculum["roll_sprint_invalid_cycle_weight"].params[
        "weight_stages"
    ] == [
        {"step": 0, "weight": 0.0},
        {"step": 2000 * 24, "weight": -0.5},
        {"step": 3000 * 24, "weight": -1.0},
        {"step": 3750 * 24, "weight": -2.0},
    ]
    spawn_stages = cfg.curriculum["roll_sprint_spawn_mix"].params["param_stages"]
    assert [stage["step"] for stage in spawn_stages] == [
        0,
        400 * 24,
        1000 * 24,
        2000 * 24,
    ]
    assert [
        (
            stage["params"]["standing_prob"],
            stage["params"]["midroll_prob"],
            stage["params"]["postroll_prob"],
            stage["params"]["crouch_prob"],
            stage["params"]["ground_recovery_prob"],
        )
        for stage in spawn_stages
    ] == [
        (0.45, 0.10, 0.15, 0.10, 0.20),
        (0.45, 0.05, 0.15, 0.15, 0.20),
        (0.55, 0.05, 0.10, 0.10, 0.20),
        (0.65, 0.00, 0.10, 0.05, 0.20),
    ]
    assert [
        stage["params"]["recovery_road_return_prob"] for stage in spawn_stages
    ] == [0.35, 0.30, 0.20, 0.10]
    assert [
        stage["params"]["heading_return_prob"] for stage in spawn_stages
    ] == [0.10, 0.10, 0.08, 0.05]
    assert [
        tuple(
            stage["params"][name]
            for name in (
                "ground_face_down_prob",
                "ground_face_up_prob",
                "ground_left_prob",
                "ground_right_prob",
            )
        )
        for stage in spawn_stages
    ] == [
        (0.25, 0.25, 0.25, 0.25),
        (0.25, 0.25, 0.25, 0.25),
        (0.25, 0.25, 0.25, 0.25),
        (0.25, 0.25, 0.25, 0.25),
    ]
    assert cfg.curriculum["roll_sprint_progress_weight"].params[
        "weight_stages"
    ] == [
        {"step": 0, "weight": 1.5},
        {"step": 1250 * 24, "weight": 1.0},
        {"step": 2500 * 24, "weight": 0.25},
        {"step": 3500 * 24, "weight": 0.0},
    ]
    assert cfg.curriculum["roll_sprint_distance_weight"].params[
        "weight_stages"
    ] == [
        {"step": 0, "weight": 32.0},
    ]
    assert cfg.curriculum["roll_sprint_self_right_upward_weight"].params[
        "weight_stages"
    ] == [
        {"step": 0, "weight": 1.0},
        {"step": 1000 * 24, "weight": 2.0},
    ]
    assert cfg.curriculum["roll_sprint_self_right_fallen_tax_weight"].params[
        "weight_stages"
    ][-1] == {"step": 1000 * 24, "weight": -0.5}
    assert cfg.curriculum["roll_sprint_self_right_success_weight"].params[
        "weight_stages"
    ] == [
        {"step": 0, "weight": 1.0},
        {"step": 1000 * 24, "weight": 10.0},
    ]
    assert cfg.curriculum["roll_sprint_head_pivot_weight"].params[
        "weight_stages"
    ] == [
        {"step": 0, "weight": 0.25},
        {"step": 3000 * 24, "weight": 0.10},
    ]
    assert cfg.curriculum["com_range"].params["range_stages"] == [
        {"step": 0, "range": COM_RANDOMIZATION_RANGE},
        {"step": 2000 * 24, "range": 0.005},
        {"step": 3000 * 24, "range": 0.01},
        {"step": 3750 * 24, "range": 0.015},
    ]
    assert cfg.curriculum["head_com_range"].params["range_stages"] == [
        {"step": 0, "range": HEAD_COM_RANDOMIZATION_RANGE},
        {"step": 2000 * 24, "range": 0.005},
        {"step": 3000 * 24, "range": 0.01},
    ]
    assert {
        "roll_sprint_recovery_count",
        "roll_sprint_recovered_reroll_count",
        "roll_sprint_mean_recovery_latency_s",
        "roll_sprint_reposition_count",
        "roll_sprint_mean_reposition_latency_s",
        "roll_sprint_self_right_attempt_count",
        "roll_sprint_self_right_success_count",
        "roll_sprint_self_right_success_rate",
        "roll_sprint_mean_self_right_latency_s",
        "roll_sprint_frontier_after_self_right_m",
    }.issubset(cfg.metrics)


def test_roll_sprint_road_return_weight_stays_fixed_at_proven_level():
    cfg = make_microduck_roll_sprint_env_cfg()
    term = cfg.curriculum["roll_sprint_road_return_weight"]

    assert term.func is mdp.reward_weight
    assert term.params["reward_name"] == "roll_sprint_road_return"
    assert term.params["weight_stages"] == [
        {"step": 0, "weight": 4.0},
    ]


def test_roll_sprint_and_backlash_tasks_are_registered():
    from mjlab.tasks.registry import list_tasks

    import mjlab_microduck.tasks  # noqa: F401

    tasks = list_tasks()
    assert "Mjlab-Roll-Sprint-Flat-MicroDuck" in tasks
    assert "Mjlab-Roll-Sprint-Flat-Backlash-MicroDuck" in tasks


def test_roll_sprint_buffer_reset_is_per_environment():
    env, _ = _fake_env(3)
    mdp._roll_sprint_state(env)
    env._roll_sprint_accum[:] = torch.tensor([0.5, 1.0, 1.5])
    env._roll_sprint_phase_frontier[:] = torch.tensor([0.6, 1.1, 1.6])
    env._roll_sprint_forward_frontier[:] = torch.tensor([0.2, 0.3, 0.4])
    env._roll_sprint_completed[:] = torch.tensor([2.0, 3.0, 4.0])
    env._roll_sprint_head_latch[:] = True
    env._roll_sprint_cycle_eligible[:] = True
    env._roll_sprint_awaiting_recovery[:] = True
    env._roll_sprint_awaiting_reposition[:] = True
    env._roll_sprint_reposition_count[:] = 2.0
    env._roll_sprint_recovery_count[:] = 2.0
    env._roll_sprint_lateral_invalid[:] = True

    mdp._reset_roll_sprint_buffers(
        env,
        torch.tensor([1]),
        spawn_accum=torch.tensor([0.25]),
        spawn_head_latch=torch.tensor([False]),
    )

    assert torch.allclose(env._roll_sprint_accum, torch.tensor([0.5, 0.25, 1.5]))
    assert torch.allclose(
        env._roll_sprint_phase_frontier, torch.tensor([0.6, 0.25, 1.6])
    )
    assert torch.allclose(
        env._roll_sprint_forward_frontier, torch.tensor([0.2, 0.0, 0.4])
    )
    assert torch.equal(env._roll_sprint_completed, torch.tensor([2.0, 0.0, 4.0]))
    assert torch.equal(env._roll_sprint_head_latch, torch.tensor([True, False, True]))
    assert torch.equal(
        env._roll_sprint_lateral_invalid, torch.tensor([True, False, True])
    )
    assert torch.equal(
        env._roll_sprint_awaiting_recovery, torch.tensor([True, False, True])
    )
    assert torch.equal(
        env._roll_sprint_awaiting_reposition, torch.tensor([True, False, True])
    )
    assert torch.equal(env._roll_sprint_reposition_count, torch.tensor([2.0, 0.0, 2.0]))
    assert torch.equal(env._roll_sprint_recovery_count, torch.tensor([2.0, 0.0, 2.0]))


def test_first_post_reset_update_freezes_actual_course_frame(monkeypatch):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    env.scene.terrain.env_origins[0, :2] = torch.tensor([-9.0, 4.0])
    mdp._reset_roll_sprint_buffers(
        env,
        torch.tensor([0]),
        spawn_course_lateral_position=torch.tensor([0.31]),
    )

    asset.data.root_link_pos_w[0] = torch.tensor([3.0, -2.0, 0.115])
    _set_yaw(asset, torch.tensor([math.pi / 2.0]))
    mdp._update_roll_sprint_state(env, asset)

    assert env._roll_sprint_heading_ready[0]
    assert env._roll_sprint_heading_w[0].tolist() == pytest.approx(
        [0.0, 1.0], abs=1.0e-6
    )
    assert env._roll_sprint_course_center_xy_w[0].tolist() == pytest.approx(
        [3.31, -2.0]
    )
    assert env._roll_sprint_course_lateral_position[0] == pytest.approx(0.31)
    assert env._roll_sprint_forward_origin[0] == pytest.approx(0.0, abs=1.0e-6)


def test_heading_return_reset_exposes_signed_error_and_requires_reposition(monkeypatch):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    angle = math.radians(30.0)
    mdp._reset_roll_sprint_buffers(
        env,
        torch.tensor([0]),
        spawn_cycle_eligible=torch.tensor([False]),
        spawn_awaiting_reposition=torch.tensor([True]),
        spawn_heading_error_rad=torch.tensor([angle]),
    )
    env.command_manager = SimpleNamespace(
        get_command=lambda command_name: torch.zeros((1, 3))
    )

    _set_yaw(asset, torch.tensor([0.0]))
    mdp._update_roll_sprint_state(env, asset)
    command = mdp.roll_sprint_reposition_command(env)

    assert env._roll_sprint_heading_w[0].tolist() == pytest.approx(
        [math.cos(angle), math.sin(angle)], abs=1.0e-6
    )
    assert env._roll_sprint_heading_error_rad[0] == pytest.approx(angle)
    assert env._roll_sprint_awaiting_reposition[0]
    assert not env._roll_sprint_cycle_eligible[0]
    assert command[0, 2] == pytest.approx(
        mdp._ROLL_SPRINT_REPOSITION_YAW_COMMAND_RPS
    )
    assert env._roll_sprint_spawn_heading_error_rad[0] == 0.0


@pytest.mark.parametrize("left_prob,right_prob", [(1.0, 0.0), (0.0, 1.0)])
def test_side_recovery_reset_preserves_sampled_course_heading(
    monkeypatch,
    left_prob,
    right_prob,
):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    env.sim = SimpleNamespace(
        data=SimpleNamespace(qpos=torch.zeros(1, 21), qvel=torch.zeros(1, 20))
    )
    env._roulade_accum = torch.zeros(1)
    monkeypatch.setattr(mdp, "reset_roulade_state", lambda *args, **kwargs: None)

    yaw = 0.73
    mdp.reset_roll_sprint_state(
        env,
        torch.tensor([0]),
        standing_prob=0.0,
        midroll_prob=0.0,
        postroll_prob=0.0,
        crouch_prob=0.0,
        ground_recovery_prob=1.0,
        ground_face_down_prob=0.0,
        ground_face_up_prob=0.0,
        ground_left_prob=left_prob,
        ground_right_prob=right_prob,
        yaw_range=(yaw, yaw),
        road_interior_prob=1.0,
        road_edge_prob=0.0,
        road_return_prob=0.0,
    )

    intended_lateral = env._roll_sprint_course_lateral_position.clone()
    expected_heading = torch.tensor([math.cos(yaw), math.sin(yaw)])
    assert env._roll_sprint_reset_heading_override_valid[0]
    assert not env._roll_sprint_heading_ready[0]
    assert env._roll_sprint_reset_heading_override_w[0].tolist() == pytest.approx(
        expected_heading.tolist()
    )

    asset.data.root_link_pos_w[0] = torch.tensor([2.0, -1.0, 0.22])
    asset.data.root_link_quat_w[0] = env.sim.data.qpos[0, 3:7]
    mdp._update_roll_sprint_state(env, asset)

    assert env._roll_sprint_heading_ready[0]
    assert not env._roll_sprint_reset_heading_override_valid[0]
    assert env._roll_sprint_heading_w[0].tolist() == pytest.approx(
        expected_heading.tolist()
    )
    assert env._roll_sprint_course_lateral_position[0].item() == pytest.approx(
        intended_lateral[0].item(), abs=1.0e-6
    )


def test_recovery_curriculum_forces_integrated_road_return_starts(monkeypatch):
    env, _asset = _fake_env(4)
    _enable_flat_valid_roll(monkeypatch, env)
    env.sim = SimpleNamespace(
        data=SimpleNamespace(qpos=torch.zeros(4, 21), qvel=torch.zeros(4, 20))
    )
    env._roulade_accum = torch.zeros(4)
    monkeypatch.setattr(mdp, "reset_roulade_state", lambda *args, **kwargs: None)

    mdp.reset_roll_sprint_state(
        env,
        torch.arange(4),
        standing_prob=0.0,
        midroll_prob=0.0,
        postroll_prob=0.0,
        crouch_prob=0.0,
        ground_recovery_prob=1.0,
        ground_face_down_prob=1.0,
        ground_face_up_prob=0.0,
        ground_left_prob=0.0,
        ground_right_prob=0.0,
        road_interior_prob=1.0,
        road_edge_prob=0.0,
        road_return_prob=0.0,
        recovery_road_return_prob=1.0,
    )

    assert torch.all(env._roll_sprint_self_righting)
    assert torch.all(env._roll_sprint_awaiting_reposition)
    assert torch.all(
        env._roll_sprint_course_lateral_position.abs()
        > mdp._ROLL_SPRINT_REPOSITION_TRIGGER_M
    )


def test_roll_sprint_requires_support_and_sagittal_forward_rotation(monkeypatch):
    env, asset = _fake_env(3)
    asset.data.root_link_ang_vel_b[:, 1] = torch.tensor([5.0, 5.0, -5.0])
    env.scene.sensors["robot_ground_contact"].data.found[:, 0] = torch.tensor(
        [0.0, 1.0, 1.0]
    )
    monkeypatch.setattr(
        mdp,
        "_lateral_axis_z",
        lambda quat: torch.tensor([0.0, 0.99, 0.0]),
    )
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda env, asset: torch.ones(env.num_envs, dtype=torch.bool),
    )

    mdp._update_roll_sprint_state(env, asset)

    assert env._roll_sprint_accum[0] == 0.0
    assert env._roll_sprint_accum[1] == 0.0
    assert env._roll_sprint_accum[2] == 0.0
    assert not env._roll_sprint_completed_now.any()


def test_roll_sprint_missing_support_sensor_has_zero_credit(monkeypatch):
    env, asset = _fake_env(1)
    del env.scene.sensors["robot_ground_contact"]
    asset.data.root_link_ang_vel_b[:, 1] = 5.0
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda env, asset: torch.ones(env.num_envs, dtype=torch.bool),
    )

    mdp._update_roll_sprint_state(env, asset)

    assert env._roll_sprint_accum[0] == 0.0
    assert env._roll_sprint_progress_delta[0] == 0.0
    assert env._roll_sprint_completed_distance[0] == 0.0


def test_roll_sprint_backward_rocking_never_completes_or_repays_progress(monkeypatch):
    env, asset = _fake_env(1)
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda env, asset: torch.ones(env.num_envs, dtype=torch.bool),
    )
    paid_progress = 0.0

    for step in range(40):
        env.common_step_counter = step
        asset.data.root_link_ang_vel_b[:, 1] = 5.0 if step % 2 == 0 else -5.0
        mdp._update_roll_sprint_state(env, asset)
        paid_progress += float(env._roll_sprint_progress_delta[0])

    assert torch.isclose(env._roll_sprint_accum[0], torch.tensor(0.0))
    assert torch.isclose(torch.tensor(paid_progress), torch.tensor(0.1))
    assert env._roll_sprint_completed[0] == 0.0
    assert not env._roll_sprint_completed_now[0]


def test_roll_sprint_head_top_latch_and_invalid_cycle_zero_credit(monkeypatch):
    env, asset = _fake_env(2)
    asset.data.root_link_ang_vel_b[:, 1] = 1.0
    env.scene.sensors["head_ground_contact"].data.found[:] = 1.0
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda env, asset: torch.tensor([True, False]),
    )

    mdp._roll_sprint_state(env)
    env._roll_sprint_accum[:] = 0.5
    mdp._update_roll_sprint_state(env, asset)
    assert torch.equal(env._roll_sprint_head_latch, torch.tensor([True, False]))

    env.common_step_counter = 1
    env._roll_sprint_accum[:] = 2.0 * torch.pi - 0.01
    env._roll_sprint_head_latch[:] = False
    mdp._update_roll_sprint_state(env, asset)

    assert not env._roll_sprint_completed_now.any()
    assert env._roll_sprint_invalid_now.all()
    assert env._roll_sprint_completed.sum() == 0.0
    assert env._roll_sprint_completed_distance.sum() == 0.0
    assert env._roll_sprint_accum.sum() == 0.0


def test_midroll_bootstrap_segment_cannot_receive_cycle_credit(monkeypatch):
    env, asset = _fake_env(1)
    asset.data.root_link_ang_vel_b[:, 1] = 1.0
    env.scene.sensors["head_ground_contact"].data.found[:] = 1.0
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda env, asset: torch.ones(env.num_envs, dtype=torch.bool),
    )
    mdp._reset_roll_sprint_buffers(
        env,
        torch.tensor([0]),
        spawn_accum=torch.tensor([2.0 * torch.pi - 0.01]),
        spawn_head_latch=torch.tensor([True]),
        spawn_cycle_eligible=torch.tensor([False]),
    )

    mdp._update_roll_sprint_state(env, asset)

    assert env._roll_sprint_bootstrap_completed_now[0]
    assert not env._roll_sprint_completed_now[0]
    assert not env._roll_sprint_invalid_now[0]
    assert env._roll_sprint_completed[0] == 0.0
    assert env._roll_sprint_completed_distance[0] == 0.0
    assert env._roll_sprint_awaiting_recovery[0]
    assert not env._roll_sprint_cycle_eligible[0]

    _recover(env, asset)
    assert not env._roll_sprint_awaiting_recovery[0]
    assert env._roll_sprint_cycle_eligible[0]

    env.common_step_counter += 1
    _complete_valid_roll(env, asset, forward=0.20)

    assert env._roll_sprint_completed_now[0]
    assert env._roll_sprint_completed[0] == 1.0


def test_valid_roll_enters_recovery_required_and_blocks_second_roll(monkeypatch):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    _prime_roll_heading(env, asset)
    _complete_valid_roll(env, asset, forward=0.20)

    assert env._roll_sprint_awaiting_recovery[0]
    assert env._roll_sprint_self_righting[0]
    assert env._roll_sprint_self_right_attempt_count[0] == 1.0
    assert not env._roll_sprint_cycle_eligible[0]
    assert env._roll_sprint_completed[0] == 1.0

    env.common_step_counter += 1
    _complete_valid_roll(env, asset, forward=0.40)

    assert env._roll_sprint_completed[0] == 1.0
    assert env._roll_sprint_completed_distance[0] == 0.0
    assert env._roll_sprint_accum[0] == 0.0
    assert env._roll_sprint_forward_frontier[0] == pytest.approx(0.20)


def test_feet_supported_recovery_rearms_once_without_standing_annuity(monkeypatch):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    _prime_roll_heading(env, asset)
    _complete_valid_roll(env, asset, forward=0.20)

    _recover(env, asset)

    assert env._roll_sprint_recovered_now[0]
    assert env._roll_sprint_recovery_count[0] == 1.0
    assert env._roll_sprint_cycle_eligible[0]
    assert not env._roll_sprint_awaiting_recovery[0]

    env.common_step_counter += 1
    reward = mdp.roll_sprint_recovery_rate(env)
    assert reward[0] == 0.0
    assert env._roll_sprint_recovery_count[0] == 1.0


def test_standing_without_completed_roll_cannot_farm_recovery_reward(monkeypatch):
    env, _asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)

    for _ in range(8):
        reward = mdp.roll_sprint_recovery_rate(env)
        assert reward[0] == 0.0
        env.common_step_counter += 1

    assert env._roll_sprint_recovery_count[0] == 0.0


def test_stalled_fall_enters_self_righting_after_point_three_seconds(monkeypatch):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    asset.data.root_link_quat_w[:] = torch.tensor(
        [[2.0**-0.5, 0.0, 2.0**-0.5, 0.0]]
    )
    asset.data.root_link_ang_vel_b.zero_()
    stall_steps = round(mdp._ROLL_SPRINT_SELF_RIGHT_STALL_SECONDS / env.step_dt)

    for step in range(stall_steps - 1):
        env.common_step_counter = step
        mdp._update_roll_sprint_state(env, asset)
        assert not env._roll_sprint_self_righting[0]

    env.common_step_counter = stall_steps - 1
    mdp._update_roll_sprint_state(env, asset)

    assert env._roll_sprint_self_righting[0]
    assert env._roll_sprint_self_right_started_now[0]
    assert env._roll_sprint_awaiting_recovery[0]
    assert not env._roll_sprint_cycle_eligible[0]


def test_actively_progressing_roll_never_triggers_stalled_fall(monkeypatch):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    asset.data.root_link_quat_w[:] = torch.tensor(
        [[2.0**-0.5, 0.0, 2.0**-0.5, 0.0]]
    )
    asset.data.root_link_ang_vel_b[:, 1] = 2.0

    for step in range(25):
        env.common_step_counter = step
        mdp._update_roll_sprint_state(env, asset)

    assert env._roll_sprint_phase_frontier[0] > 0.0
    assert not env._roll_sprint_self_righting[0]
    assert env._roll_sprint_self_right_attempt_count[0] == 0.0


@pytest.mark.parametrize(
    ("kind", "quaternion"),
    [
        (1, [2.0**-0.5, 0.0, 2.0**-0.5, 0.0]),
        (2, [2.0**-0.5, 0.0, -(2.0**-0.5), 0.0]),
        (3, [2.0**-0.5, 2.0**-0.5, 0.0, 0.0]),
        (4, [2.0**-0.5, -(2.0**-0.5), 0.0, 0.0]),
    ],
)
def test_each_lying_orientation_is_an_ineligible_self_right_start(
    monkeypatch, kind, quaternion
):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    asset.data.root_link_quat_w[:] = torch.tensor([quaternion])
    mdp._reset_roll_sprint_buffers(
        env,
        torch.tensor([0]),
        spawn_cycle_eligible=torch.tensor([False]),
        spawn_awaiting_recovery=torch.tensor([True]),
        spawn_self_righting=torch.tensor([True]),
        spawn_recovery_kind=torch.tensor([kind]),
    )

    mdp._update_roll_sprint_state(env, asset)
    assert env._roll_sprint_self_righting[0]
    assert not env._roll_sprint_cycle_eligible[0]
    assert env._roll_sprint_completed_distance[0] == 0.0

    _recover(env, asset)
    assert env._roll_sprint_self_right_success_count[0] == 1.0
    assert env._roll_sprint_recovery_count[0] == 1.0
    assert env._roll_sprint_cycle_eligible[0]


def test_self_right_rewards_are_one_shot_and_lying_cannot_farm(monkeypatch):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    asset.data.root_link_quat_w[:] = torch.tensor(
        [[2.0**-0.5, 0.0, 2.0**-0.5, 0.0]]
    )
    asset.data.root_link_pos_w[:, 2] = 0.06
    mdp._reset_roll_sprint_buffers(
        env,
        torch.tensor([0]),
        spawn_cycle_eligible=torch.tensor([False]),
        spawn_awaiting_recovery=torch.tensor([True]),
        spawn_self_righting=torch.tensor([True]),
    )

    for _ in range(8):
        assert mdp.roll_sprint_self_right_upright_progress(env)[0] == 0.0
        assert mdp.roll_sprint_self_right_height_progress(env)[0] == 0.0
        assert mdp.roll_sprint_self_right_success_rate(env)[0] == 0.0
        assert mdp.roll_sprint_distance(env)[0] == 0.0
        env.common_step_counter += 1

    env.scene.sensors["head_ground_contact"].data.found[:] = 0.0
    env.scene.sensors["feet_ground_contact"].data.found[:] = 1.0
    asset.data.root_link_quat_w[:] = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    asset.data.root_link_pos_w[:, 2] = 0.115
    asset.data.root_link_ang_vel_b.zero_()
    upright_progress = 0.0
    for _ in range(mdp._ROLL_SPRINT_RECOVERY_HOLD_STEPS):
        env.common_step_counter += 1
        upright_progress += float(
            mdp.roll_sprint_self_right_upright_progress(env)[0].item()
        )
    assert mdp.roll_sprint_self_right_success_rate(env)[0] == pytest.approx(
        1.0 / env.step_dt
    )
    assert upright_progress > 0.0
    env.common_step_counter += 1
    assert mdp.roll_sprint_self_right_success_rate(env)[0] == 0.0
    assert mdp.roll_sprint_self_right_upright_progress(env)[0] == 0.0
    assert mdp.roll_sprint_self_right_height_progress(env)[0] == 0.0


def test_self_right_potentials_are_rates_and_roll_costs_pause_during_recovery(
    monkeypatch,
):
    env, asset = _fake_env(4)
    mdp._roll_sprint_state(env)
    monkeypatch.setattr(mdp, "_update_roll_sprint_state", lambda env, asset: None)
    monkeypatch.setattr(
        mdp,
        "_lateral_axis_z",
        lambda quat: torch.ones(env.num_envs),
    )
    env._roll_sprint_self_righting[:] = torch.tensor([True, False, False, False])
    env._roll_sprint_self_righted_now[:] = False
    env._roll_sprint_awaiting_recovery[:] = torch.tensor(
        [True, False, True, False]
    )
    env._roll_sprint_awaiting_reposition[:] = torch.tensor(
        [False, True, False, False]
    )
    env._roll_sprint_self_right_upright_delta[:] = torch.tensor(
        [0.20, 0.0, 0.0, 0.0]
    )
    env._roll_sprint_self_right_height_delta[:] = torch.tensor(
        [0.01, 0.0, 0.0, 0.0]
    )
    env._roll_sprint_lane_centering_delta[:] = torch.tensor(
        [-0.04, 0.04, -0.04, -0.04]
    )
    asset.data.root_link_ang_vel_b[:, 0] = 2.0
    asset.data.root_link_lin_vel_b[:, 1] = 0.5

    assert mdp.roll_sprint_self_right_upright_progress(env).tolist() == pytest.approx(
        [10.0, 0.0, 0.0, 0.0]
    )
    assert mdp.roll_sprint_self_right_height_progress(env).tolist() == pytest.approx(
        [0.5, 0.0, 0.0, 0.0]
    )
    assert mdp.roll_sprint_sagittal_penalty(env).tolist() == pytest.approx(
        [0.0, 0.0, 0.0, 4.0]
    )
    assert mdp.roll_sprint_lateral_velocity_penalty(env).tolist() == pytest.approx(
        [0.0, 0.0, 0.0, 0.25]
    )
    assert mdp.roll_sprint_flatness_penalty(env).tolist() == pytest.approx(
        [0.0, 0.0, 0.0, 1.0]
    )
    assert mdp.roll_sprint_lane_centering_progress(env).tolist() == pytest.approx(
        [0.0, 2.0, 0.0, -2.0]
    )


def test_roll_recover_reroll_credits_two_cycles(monkeypatch):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    _prime_roll_heading(env, asset)
    _complete_valid_roll(env, asset, forward=0.20)
    _recover(env, asset)

    env.common_step_counter += 1
    _complete_valid_roll(env, asset, forward=0.45)

    assert env._roll_sprint_completed[0] == 2.0
    assert env._roll_sprint_recovered_and_rerolled[0] == 1.0
    assert env._roll_sprint_forward_frontier[0] == pytest.approx(0.45)
    assert env._roll_sprint_completed_distance[0] == pytest.approx(0.25)
    assert mdp.roll_sprint_recovered_reroll_rate(env)[0] == pytest.approx(
        1.0 / env.step_dt
    )

    env.common_step_counter += 1
    assert mdp.roll_sprint_recovered_reroll_rate(env)[0] == 0.0


def test_roll_sprint_side_violation_invalidates_whole_cycle(monkeypatch):
    env, asset = _fake_env(1)
    asset.data.root_link_ang_vel_b[:, 1] = 1.0
    env.scene.sensors["head_ground_contact"].data.found[:] = 1.0
    lateral = torch.tensor([0.90])
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda quat: lateral)
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda env, asset: torch.ones(env.num_envs, dtype=torch.bool),
    )

    mdp._update_roll_sprint_state(env, asset)
    assert env._roll_sprint_lateral_invalid[0]

    env.common_step_counter = 1
    lateral[:] = 0.0
    env._roll_sprint_accum[:] = 2.0 * torch.pi - 0.01
    env._roll_sprint_phase_frontier[:] = env._roll_sprint_accum
    env._roll_sprint_head_latch[:] = True
    mdp._update_roll_sprint_state(env, asset)

    assert env._roll_sprint_invalid_now[0]
    assert not env._roll_sprint_completed_now[0]
    assert env._roll_sprint_completed_distance[0] == 0.0


def test_roll_sprint_heading_survives_vertical_pitch():
    _, asset = _fake_env(1)
    half = torch.tensor(torch.pi / 4.0)
    # ZYX yaw=90 degrees, pitch=90 degrees, roll=0.
    asset.data.root_link_quat_w[:] = torch.tensor(
        [
            [
                half.cos() * half.cos(),
                -half.sin() * half.sin(),
                half.cos() * half.sin(),
                half.sin() * half.cos(),
            ]
        ]
    )

    heading = mdp._roll_sprint_heading(asset)

    assert torch.allclose(heading, torch.tensor([[0.0, 1.0]]), atol=1.0e-5)
    error = mdp._roll_sprint_signed_heading_error(
        heading,
        torch.tensor([[1.0, 0.0]]),
    )
    assert error[0] == pytest.approx(-math.pi / 2.0, abs=1.0e-5)


def test_roll_sprint_heading_error_is_wrap_safe():
    current_angle = math.radians(-179.0)
    target_angle = math.radians(179.0)
    current = torch.tensor([[math.cos(current_angle), math.sin(current_angle)]])
    target = torch.tensor([[math.cos(target_angle), math.sin(target_angle)]])

    error = mdp._roll_sprint_signed_heading_error(current, target)

    assert error[0] == pytest.approx(math.radians(-2.0), abs=1.0e-6)


def test_roll_sprint_per_env_step_guard_updates_only_reset_environment(monkeypatch):
    env, asset = _fake_env(2)
    asset.data.root_link_ang_vel_b[:, 1] = 1.0
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda env, asset: torch.ones(env.num_envs, dtype=torch.bool),
    )
    mdp._update_roll_sprint_state(env, asset)
    before = env._roll_sprint_accum.clone()

    mdp._reset_roll_sprint_buffers(env, torch.tensor([0]))
    asset.data.root_link_ang_vel_b[:, 1] = 2.0
    mdp._update_roll_sprint_state(env, asset)

    assert env._roll_sprint_accum[0] > 0.0
    assert env._roll_sprint_accum[1] == before[1]


def test_roll_sprint_releases_heading_aligned_distance_on_completion(monkeypatch):
    env, asset = _fake_env(1)
    yaw_quat = torch.tensor([[2.0**-0.5, 0.0, 0.0, 2.0**-0.5]])
    asset.data.root_link_quat_w[:] = yaw_quat
    env.scene.sensors["head_ground_contact"].data.found[:] = 1.0
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda env, asset: torch.ones(env.num_envs, dtype=torch.bool),
    )
    mdp._update_roll_sprint_state(env, asset)
    env.common_step_counter = 1
    asset.data.root_link_pos_w[:, 1] = 0.25
    asset.data.root_link_ang_vel_b[:, 1] = 1.0
    env._roll_sprint_accum[:] = 2.0 * torch.pi - 0.01
    env._roll_sprint_phase_frontier[:] = env._roll_sprint_accum
    env._roll_sprint_head_latch[:] = True

    mdp._update_roll_sprint_state(env, asset)

    assert env._roll_sprint_completed[0] == 1.0
    assert env._roll_sprint_completed_now[0]
    assert torch.allclose(
        env._roll_sprint_completed_distance, torch.tensor([0.25]), atol=1.0e-6
    )


def test_roll_sprint_distance_is_rotation_linked_and_capped(monkeypatch):
    env, asset = _fake_env(1)
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda env, asset: torch.ones(env.num_envs, dtype=torch.bool),
    )
    mdp._update_roll_sprint_state(env, asset)
    env.common_step_counter = 1
    asset.data.root_link_pos_w[:, 0] = 100.0
    asset.data.root_link_ang_vel_b[:, 1] = 1.0
    env._roll_sprint_accum[:] = 2.0 * torch.pi - 0.01
    env._roll_sprint_phase_frontier[:] = env._roll_sprint_accum
    env._roll_sprint_head_latch[:] = True

    mdp._update_roll_sprint_state(env, asset)

    expected_cap = 0.12 * 2.0 * torch.pi
    assert torch.allclose(
        env._roll_sprint_completed_distance,
        torch.tensor([expected_cap]),
        atol=1.0e-6,
    )
    assert torch.allclose(
        env._roll_sprint_credited_distance,
        torch.tensor([expected_cap]),
        atol=1.0e-6,
    )
    assert env._roll_sprint_forward_frontier[0] == pytest.approx(100.0)


def test_roll_sprint_cumulative_credit_stays_bounded_while_raw_frontier_advances(
    monkeypatch,
):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    _prime_roll_heading(env, asset)

    _complete_valid_roll(env, asset, forward=4.0)
    expected_cap = 0.12 * 2.0 * torch.pi
    assert env._roll_sprint_credited_distance[0] == pytest.approx(expected_cap)
    assert env._roll_sprint_forward_frontier[0] == pytest.approx(4.0)

    _recover(env, asset)
    env.common_step_counter += 1
    _complete_valid_roll(env, asset, forward=8.0)

    assert env._roll_sprint_credited_distance[0] == pytest.approx(2.0 * expected_cap)
    assert env._roll_sprint_forward_frontier[0] == pytest.approx(8.0)

    mdp._reset_roll_sprint_buffers(env, torch.tensor([0]))
    assert env._roll_sprint_credited_distance[0] == 0.0


def test_roll_sprint_forward_then_backward_translation_credits_only_net_advance(
    monkeypatch,
):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    _prime_roll_heading(env, asset)

    asset.data.root_link_pos_w[:, 0] = 0.40
    asset.data.root_link_ang_vel_b[:, 1] = 1.0
    mdp._update_roll_sprint_state(env, asset)

    env.common_step_counter += 1
    _complete_valid_roll(env, asset, forward=0.20)

    assert torch.allclose(
        env._roll_sprint_completed_distance, torch.tensor([0.20]), atol=1.0e-6
    )


def test_roll_sprint_revisiting_credited_forward_point_earns_zero(monkeypatch):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    _prime_roll_heading(env, asset)
    _complete_valid_roll(env, asset, forward=0.30)
    assert env._roll_sprint_completed_distance[0] == pytest.approx(0.30)

    _recover(env, asset)
    env.common_step_counter += 1
    _complete_valid_roll(env, asset, forward=0.30)

    assert env._roll_sprint_completed_distance[0] == 0.0
    assert env._roll_sprint_forward_frontier[0] == pytest.approx(0.30)


def test_roll_sprint_only_credits_extension_beyond_global_frontier(monkeypatch):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    _prime_roll_heading(env, asset)
    _complete_valid_roll(env, asset, forward=0.20)

    _recover(env, asset)
    env.common_step_counter += 1
    asset.data.root_link_pos_w[:, 0] = 0.0
    asset.data.root_link_ang_vel_b[:, 1] = 1.0
    env._roll_sprint_accum[:] = 2.0 * torch.pi - 0.01
    env._roll_sprint_phase_frontier[:] = env._roll_sprint_accum
    env._roll_sprint_head_latch[:] = False
    mdp._update_roll_sprint_state(env, asset)
    assert env._roll_sprint_invalid_now[0]

    env.common_step_counter += 1
    _complete_valid_roll(env, asset, forward=0.35)

    assert torch.allclose(
        env._roll_sprint_completed_distance, torch.tensor([0.15]), atol=1.0e-6
    )
    assert env._roll_sprint_forward_frontier[0] == pytest.approx(0.35)


def test_roll_sprint_lateral_crossing_has_no_credit_and_only_road_edge_costs(
    monkeypatch,
):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    _prime_roll_heading(env, asset)
    _complete_valid_roll(env, asset, forward=0.0, lateral=0.49)
    env._roll_sprint_progress_delta[:] = env.step_dt * mdp._ROLL_SPRINT_TARGET_ANGLE

    penalty = mdp.roll_sprint_straightness_penalty(env)

    assert env._roll_sprint_completed_distance[0] == 0.0
    assert penalty[0] == pytest.approx(0.07)
    env._roll_sprint_progress_delta.zero_()
    assert mdp.roll_sprint_straightness_penalty(env)[0] == 0.0


def test_crossing_internal_lane_guides_remains_a_valid_road_cycle(monkeypatch):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    asset.data.root_link_pos_w[:, 1] = -0.42
    mdp._reset_roll_sprint_buffers(
        env,
        torch.tensor([0]),
        spawn_course_lateral_position=torch.tensor([-0.42]),
    )
    _prime_roll_heading(env, asset)

    env.common_step_counter += 1
    _complete_valid_roll(env, asset, forward=0.30, lateral=0.42)

    assert env._roll_sprint_completed_now[0]
    assert env._roll_sprint_completed_distance[0] == pytest.approx(0.30)
    assert not env._roll_sprint_awaiting_reposition[0]
    assert not env._roll_sprint_lateral_invalid[0]


def test_roll_sprint_excessive_drift_requires_reposition_before_restart(monkeypatch):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    _prime_roll_heading(env, asset)

    asset.data.root_link_pos_w[:, 1] = mdp._ROLL_SPRINT_REPOSITION_TRIGGER_M + 0.01
    asset.data.root_link_ang_vel_b[:, 1] = 5.0
    mdp._update_roll_sprint_state(env, asset)

    assert env._roll_sprint_awaiting_reposition[0]
    assert env._roll_sprint_accum[0] == 0.0
    assert env._roll_sprint_progress_delta[0] == 0.0

    # Even a nominally complete roll is locked out before the lane return.
    env.common_step_counter += 1
    _complete_valid_roll(
        env,
        asset,
        forward=0.35,
        lateral=mdp._ROLL_SPRINT_REPOSITION_TRIGGER_M + 0.01,
    )

    assert not env._roll_sprint_completed_now[0]
    assert env._roll_sprint_completed[0] == 0.0
    assert env._roll_sprint_completed_distance[0] == 0.0
    assert env._roll_sprint_forward_frontier[0] == 0.0

    # Crossing back below the trigger is insufficient. The robot must return
    # inside the tighter rearm band and hold a launch-ready feet-supported pose.
    asset.data.root_link_pos_w[:, 1] = mdp._ROLL_SPRINT_REPOSITION_REARM_M + 0.01
    _recover(env, asset)
    assert env._roll_sprint_awaiting_reposition[0]

    asset.data.root_link_pos_w[:, 1] = 0.0
    _recover(env, asset)
    assert env._roll_sprint_repositioned_now[0]
    assert not env._roll_sprint_awaiting_reposition[0]
    assert env._roll_sprint_reposition_count[0] == 1.0
    assert env._roll_sprint_recovery_count[0] == 0.0
    assert env._roll_sprint_cycle_eligible[0]

    env.common_step_counter += 1
    _complete_valid_roll(env, asset, forward=0.45)
    assert env._roll_sprint_completed_now[0]
    assert env._roll_sprint_completed_distance[0] == pytest.approx(0.10)
    assert env._roll_sprint_forward_frontier[0] == pytest.approx(0.45)


def test_leaving_shared_road_discards_partial_cycle_and_frontier_credit(monkeypatch):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    _prime_roll_heading(env, asset)
    env._roll_sprint_accum[:] = 2.0 * torch.pi - 0.01
    env._roll_sprint_phase_frontier[:] = env._roll_sprint_accum
    env._roll_sprint_head_latch[:] = True
    asset.data.root_link_pos_w[:, 0] = 0.40
    asset.data.root_link_pos_w[:, 1] = mdp._ROLL_SPRINT_ROAD_HALF_WIDTH + 0.01
    asset.data.root_link_ang_vel_b[:, 1] = 1.0

    mdp._update_roll_sprint_state(env, asset)

    assert env._roll_sprint_awaiting_reposition[0]
    assert env._roll_sprint_accum[0] == 0.0
    assert not env._roll_sprint_completed_now[0]
    assert env._roll_sprint_completed_distance[0] == 0.0
    assert env._roll_sprint_forward_frontier[0] == 0.0


def test_roll_sprint_yaw_error_requires_alignment_before_restart(monkeypatch):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    _prime_roll_heading(env, asset)
    env._roll_sprint_accum[:] = 1.25
    env._roll_sprint_phase_frontier[:] = 1.25
    _set_yaw(asset, torch.tensor([math.radians(21.0)]))

    mdp._update_roll_sprint_state(env, asset)

    assert env._roll_sprint_awaiting_reposition[0]
    assert env._roll_sprint_accum[0] == 0.0
    assert env._roll_sprint_completed[0] == 0.0
    assert env._roll_sprint_forward_frontier[0] == 0.0

    _recover(env, asset, yaw_radians=math.radians(11.0))
    assert env._roll_sprint_awaiting_reposition[0]
    assert env._roll_sprint_reposition_count[0] == 0.0

    _recover(env, asset, yaw_radians=0.0)
    assert not env._roll_sprint_awaiting_reposition[0]
    assert env._roll_sprint_reposition_count[0] == 1.0
    assert env._roll_sprint_cycle_eligible[0]


def test_roll_sprint_heading_violation_invalidates_active_cycle(monkeypatch):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    _prime_roll_heading(env, asset)
    yaw = math.radians(21.0)
    monkeypatch.setattr(
        mdp,
        "_roll_sprint_heading",
        lambda asset: torch.tensor([[math.cos(yaw), math.sin(yaw)]]),
    )
    half = 2.0**-0.5
    asset.data.root_link_quat_w[:] = torch.tensor([[half, 0.0, half, 0.0]])
    asset.data.root_link_ang_vel_b[:, 1] = 1.0
    env._roll_sprint_accum[:] = 2.0 * torch.pi - 0.01
    env._roll_sprint_phase_frontier[:] = env._roll_sprint_accum
    env._roll_sprint_head_latch[:] = True

    mdp._update_roll_sprint_state(env, asset)

    assert env._roll_sprint_invalid_now[0]
    assert not env._roll_sprint_completed_now[0]
    assert env._roll_sprint_completed_distance[0] == 0.0
    assert env._roll_sprint_forward_frontier[0] == 0.0


def test_roll_sprint_reposition_command_points_to_nearest_safe_road_edge(monkeypatch):
    env, asset = _fake_env(3)
    _enable_flat_valid_roll(monkeypatch, env)
    base_command = torch.tensor(
        [[0.01, 0.02, 0.03], [0.01, 0.02, 0.03], [0.01, 0.02, 0.03]]
    )
    env.command_manager = SimpleNamespace(
        get_command=lambda command_name: base_command,
    )
    _prime_roll_heading(env, asset)
    asset.data.root_link_pos_w[:, 1] = torch.tensor([0.60, -0.60, 0.60])
    _set_yaw(
        asset,
        torch.tensor(
            [math.radians(15.0), math.radians(-15.0), math.radians(90.0)]
        ),
    )
    env._roll_sprint_awaiting_reposition[:] = True

    command = mdp.roll_sprint_reposition_command(env)

    assert torch.allclose(command[0], torch.tensor([0.0, -0.20, -0.05]))
    assert torch.allclose(command[1], torch.tensor([0.0, 0.20, 0.05]))
    # A large heading error must not suppress the simultaneous road-return
    # command. Side recoveries otherwise remain outside the road indefinitely.
    assert torch.allclose(command[2], torch.tensor([0.0, -0.20, -0.05]))

    env._roll_sprint_self_righting[0] = True
    assert torch.allclose(
        mdp.roll_sprint_reposition_command(env)[0],
        torch.tensor([1.0, 0.0, -0.05]),
    )


def test_roll_sprint_lateral_return_command_is_reposition_only(monkeypatch):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    env.command_manager = SimpleNamespace(
        get_command=lambda command_name: torch.tensor([[0.01, 0.02, 0.03]]),
    )
    _prime_roll_heading(env, asset)
    asset.data.root_link_pos_w[:, 1] = 0.45

    assert torch.allclose(
        mdp.roll_sprint_reposition_command(env),
        torch.tensor([[0.0, 0.0, 0.0]]),
    )

    env._roll_sprint_awaiting_reposition[:] = True
    assert torch.allclose(
        mdp.roll_sprint_reposition_command(env),
        torch.tensor([[0.0, -0.06, 0.0]]),
    )


def test_roll_recovery_waits_for_lane_reposition_before_reroll(monkeypatch):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    _prime_roll_heading(env, asset)
    _complete_valid_roll(env, asset, forward=0.20)

    env.common_step_counter += 1
    asset.data.root_link_pos_w[:, 1] = mdp._ROLL_SPRINT_REPOSITION_TRIGGER_M + 0.01
    mdp._update_roll_sprint_state(env, asset)
    assert env._roll_sprint_awaiting_recovery[0]
    assert env._roll_sprint_self_righting[0]
    assert not env._roll_sprint_awaiting_reposition[0]

    _recover(env, asset)
    assert env._roll_sprint_awaiting_recovery[0]
    assert not env._roll_sprint_self_righting[0]
    assert env._roll_sprint_awaiting_reposition[0]
    assert env._roll_sprint_recovery_count[0] == 0.0

    asset.data.root_link_pos_w[:, 1] = 0.0
    _recover(env, asset)
    assert env._roll_sprint_recovered_now[0]
    assert env._roll_sprint_repositioned_now[0]
    assert mdp.roll_sprint_reposition_rate(env)[0] == pytest.approx(1.0 / env.step_dt)
    assert env._roll_sprint_recovery_count[0] == 1.0
    assert env._roll_sprint_reposition_count[0] == 1.0
    assert env._roll_sprint_forward_frontier[0] == pytest.approx(0.20)

    env.common_step_counter += 1
    _complete_valid_roll(env, asset, forward=0.45)
    assert env._roll_sprint_completed[0] == 2.0
    assert env._roll_sprint_recovered_and_rerolled[0] == 1.0


def test_roll_completion_outside_heading_rearm_repositions_and_preserves_frontier(
    monkeypatch,
):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    _prime_roll_heading(env, asset)
    asset.data.root_link_pos_w[:, 2] = 0.115
    _set_yaw(asset, torch.tensor([math.radians(15.0)]))

    _complete_valid_roll(env, asset, forward=0.20)

    assert env._roll_sprint_completed_now[0]
    assert env._roll_sprint_forward_frontier[0] == pytest.approx(0.20)
    assert env._roll_sprint_awaiting_recovery[0]
    assert env._roll_sprint_awaiting_reposition[0]

    _recover(env, asset, yaw_radians=0.0)
    assert env._roll_sprint_recovery_count[0] == 1.0
    assert env._roll_sprint_reposition_count[0] == 1.0
    assert env._roll_sprint_forward_frontier[0] == pytest.approx(0.20)

    env.common_step_counter += 1
    _complete_valid_roll(env, asset, forward=0.40)
    assert env._roll_sprint_completed[0] == 2.0
    assert env._roll_sprint_completed_distance[0] == pytest.approx(0.20)


def test_roll_sprint_shared_road_width_is_fixed():
    env, _asset = _fake_env(1)
    stages = [{"step": 0, "width": 0.56}]

    for step, expected in ((0, 0.56), (2000 * 24, 0.56)):
        env.common_step_counter = step
        reported = mdp.roll_sprint_lane_half_width_curriculum(
            env,
            torch.tensor([0]),
            stages,
        )
        assert reported[0] == pytest.approx(expected)
        assert env._roll_sprint_lane_half_width_m == pytest.approx(expected)


def test_roll_progress_is_full_across_lanes_and_fades_only_at_road_edge():
    env, _asset = _fake_env(3)
    mdp._roll_sprint_state(env)
    env._roll_sprint_last_update_step[:] = env.common_step_counter
    env._roll_sprint_progress_delta[:] = env.step_dt * mdp._ROLL_SPRINT_TARGET_ANGLE
    env._roll_sprint_course_lateral_position[:] = torch.tensor([0.42, 0.49, 0.56])

    reward = mdp.roll_sprint_progress(
        env,
        max_paid_rate=10.0,
        road_half_width=0.56,
        road_safe_half_width=0.42,
    )

    assert reward.tolist() == pytest.approx([1.0, 0.75, 0.0])
    env._roll_sprint_progress_delta.zero_()
    assert torch.equal(
        mdp.roll_sprint_progress(
            env,
            max_paid_rate=10.0,
            road_half_width=0.56,
            road_safe_half_width=0.42,
        ),
        torch.zeros(3),
    )


def test_roll_progress_uses_active_shared_road_width():
    env, _asset = _fake_env(3)
    mdp._roll_sprint_state(env)
    env._roll_sprint_last_update_step[:] = env.common_step_counter
    env._roll_sprint_progress_delta[:] = env.step_dt * mdp._ROLL_SPRINT_TARGET_ANGLE
    env._roll_sprint_course_lateral_position[:] = torch.tensor([0.42, 0.49, 0.56])
    env._roll_sprint_lane_half_width_m = 0.56

    reward = mdp.roll_sprint_progress(
        env,
        max_paid_rate=10.0,
        road_half_width=0.56,
        road_safe_half_width=0.42,
    )

    assert reward.tolist() == pytest.approx([1.0, 0.75, 0.0])


def test_roll_progress_fades_to_zero_at_heading_gate():
    env, _asset = _fake_env(4)
    mdp._roll_sprint_state(env)
    env._roll_sprint_last_update_step[:] = env.common_step_counter
    env._roll_sprint_progress_delta[:] = env.step_dt * mdp._ROLL_SPRINT_TARGET_ANGLE
    env._roll_sprint_lateral_displacement.zero_()
    env._roll_sprint_heading_error_rad[:] = torch.tensor(
        [0.0, math.radians(10.0), math.radians(20.0), math.radians(30.0)]
    )

    reward = mdp.roll_sprint_progress(env, max_paid_rate=10.0)

    assert reward.tolist() == pytest.approx([1.0, 0.75, 0.0, 0.0])


def test_roll_progress_stops_after_missed_head_phase_or_cycle_violation():
    env, _asset = _fake_env(5)
    mdp._roll_sprint_state(env)
    env._roll_sprint_last_update_step[:] = env.common_step_counter
    env._roll_sprint_progress_delta[:] = env.step_dt * mdp._ROLL_SPRINT_TARGET_ANGLE
    env._roll_sprint_lateral_displacement.zero_()
    env._roll_sprint_phase_frontier[:] = torch.tensor(
        [
            mdp._HEAD_LATCH_HI - 0.01,
            mdp._HEAD_LATCH_HI + 0.01,
            mdp._HEAD_LATCH_HI + 0.01,
            mdp._HEAD_LATCH_HI - 0.01,
            0.0,
        ]
    )
    env._roll_sprint_head_latch[:] = torch.tensor(
        [False, False, True, False, False]
    )
    env._roll_sprint_lateral_invalid[3] = True
    env._roll_sprint_invalid_now[4] = True

    reward = mdp.roll_sprint_progress(
        env,
        max_paid_rate=10.0,
            road_half_width=0.56,
            road_safe_half_width=0.42,
    )

    assert reward.tolist() == pytest.approx([1.0, 0.0, 1.0, 0.0, 0.0])


def test_road_return_progress_has_no_center_pull_and_no_idle_annuity(
    monkeypatch,
):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    mdp._update_roll_sprint_state(env, asset)

    env.common_step_counter += 1
    asset.data.root_link_lin_vel_w[:, 0] = 25.0
    asset.data.root_link_pos_w[:, 1] = 0.49
    departure = mdp.roll_sprint_road_return_progress(env)

    env.common_step_counter += 1
    asset.data.root_link_lin_vel_w[:, 0] = -25.0
    asset.data.root_link_pos_w[:, 1] = 0.44
    correction = mdp.roll_sprint_road_return_progress(env)

    env.common_step_counter += 1
    asset.data.root_link_pos_w[:, 1] = 0.20
    interior = mdp.roll_sprint_road_return_progress(env)

    env.common_step_counter += 1
    idle = mdp.roll_sprint_road_return_progress(env)

    assert departure[0] == pytest.approx(-0.07 / env.step_dt)
    assert correction[0] == pytest.approx(0.05 / env.step_dt)
    assert interior[0] == pytest.approx(0.02 / env.step_dt)
    assert idle[0] == 0.0
    assert env.step_dt * (
        departure[0] + correction[0] + interior[0] + idle[0]
    ) == pytest.approx(
        0.0, abs=1.0e-6
    )


def test_road_return_shaping_remains_active_during_reposition():
    env, _asset = _fake_env(1)
    mdp._roll_sprint_state(env)
    env._roll_sprint_awaiting_recovery[:] = True
    env._roll_sprint_awaiting_reposition[:] = True
    env._roll_sprint_self_righting[:] = False
    env._roll_sprint_self_righted_now[:] = False

    assert mdp._roll_sprint_lane_reward_mask(env)[0] == 1.0

    env._roll_sprint_self_righting[:] = True
    assert mdp._roll_sprint_lane_reward_mask(env)[0] == 0.0


def test_heading_alignment_progress_is_signed_potential_without_annuity(monkeypatch):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    _prime_roll_heading(env, asset)

    unchanged = mdp.roll_sprint_heading_alignment_progress(env)

    env.common_step_counter += 1
    _set_yaw(asset, torch.tensor([math.radians(10.0)]))
    departure = mdp.roll_sprint_heading_alignment_progress(env)

    env.common_step_counter += 1
    _set_yaw(asset, torch.tensor([math.radians(4.0)]))
    partial_return = mdp.roll_sprint_heading_alignment_progress(env)

    env.common_step_counter += 1
    _set_yaw(asset, torch.tensor([0.0]))
    full_return = mdp.roll_sprint_heading_alignment_progress(env)

    assert unchanged[0] == 0.0
    assert departure[0] == pytest.approx(-math.radians(10.0) / env.step_dt)
    assert partial_return[0] == pytest.approx(math.radians(6.0) / env.step_dt)
    assert full_return[0] == pytest.approx(math.radians(4.0) / env.step_dt)
    integrated = env.step_dt * (
        departure[0] + partial_return[0] + full_return[0]
    )
    assert integrated == pytest.approx(0.0, abs=1.0e-6)

    env.common_step_counter += 1
    assert mdp.roll_sprint_heading_alignment_progress(env)[0] == 0.0


def test_roll_sprint_backward_rotation_and_translation_earn_no_credit(monkeypatch):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    _prime_roll_heading(env, asset)

    env._roll_sprint_accum[:] = 0.50
    env._roll_sprint_phase_frontier[:] = 0.50
    asset.data.root_link_pos_w[:, 0] = -0.10
    asset.data.root_link_ang_vel_b[:, 1] = -5.0
    mdp._update_roll_sprint_state(env, asset)
    assert env._roll_sprint_accum[0] < 0.50
    assert not env._roll_sprint_completed_now[0]

    env.common_step_counter += 1
    _complete_valid_roll(env, asset, forward=-0.20)

    assert env._roll_sprint_completed_now[0]
    assert env._roll_sprint_completed_distance[0] == 0.0
    assert env._roll_sprint_forward_frontier[0] == 0.0


def test_roll_sprint_completion_discards_overshoot_until_recovery(monkeypatch):
    env, asset = _fake_env(1)
    asset.data.root_link_ang_vel_b[:, 1] = 2.0
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda env, asset: torch.ones(env.num_envs, dtype=torch.bool),
    )
    mdp._roll_sprint_state(env)
    env._roll_sprint_accum[:] = 2.0 * torch.pi - 0.01
    env._roll_sprint_phase_frontier[:] = env._roll_sprint_accum
    env._roll_sprint_head_latch[:] = True

    mdp._update_roll_sprint_state(env, asset)

    assert env._roll_sprint_completed_now[0]
    assert env._roll_sprint_awaiting_recovery[0]
    assert torch.equal(env._roll_sprint_accum, torch.zeros(1))
    assert torch.equal(env._roll_sprint_phase_frontier, torch.zeros(1))
