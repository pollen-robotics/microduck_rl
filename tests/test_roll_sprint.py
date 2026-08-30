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


def _recover(env, asset) -> None:
    env.scene.sensors["head_ground_contact"].data.found[:] = 0.0
    env.scene.sensors["feet_ground_contact"].data.found[:] = 1.0
    asset.data.root_link_quat_w[:] = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    asset.data.root_link_pos_w[:, 2] = 0.115
    asset.data.root_link_ang_vel_b[:] = 0.0
    for _ in range(mdp._ROLL_SPRINT_RECOVERY_HOLD_STEPS):
        env.common_step_counter += 1
        mdp._update_roll_sprint_state(env, asset)


def test_roll_sprint_is_separate_long_distance_61d_policy():
    cfg = make_microduck_roll_sprint_env_cfg()
    roulade = make_microduck_roulade_env_cfg()

    assert cfg.episode_length_s == EPISODE_LENGTH_S == 40.0
    assert TARGET_DISTANCE_M == 20.0
    assert mdp._ROLL_SPRINT_RECOVERY_MAX_FORWARD_RATE == 6.0
    assert mdp._ROLL_SPRINT_RECOVERY_HOLD_STEPS == 5
    assert mdp._ROLL_SPRINT_RECOVERY_MIN_HEIGHT_M == 0.09
    assert mdp._ROLL_SPRINT_SELF_RIGHT_STALL_SECONDS == 0.30
    assert mdp._ROLL_SPRINT_REPOSITION_TRIGGER_M == 0.14
    assert mdp._ROLL_SPRINT_REPOSITION_REARM_M == 0.07
    assert mdp._ROLL_SPRINT_REPOSITION_LATERAL_COMMAND_MPS == 0.20
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
    assert cfg.rewards["roll_sprint_progress"].params["lane_half_width"] == (
        mdp._ROLL_SPRINT_LANE_HALF_WIDTH
    )
    assert cfg.rewards["roll_sprint_cycle_rate"].weight == 1.0
    assert cfg.rewards["roll_sprint_recovery"].weight == 1.0
    assert cfg.rewards["roll_sprint_recovered_reroll"].weight == 4.0
    assert cfg.rewards["roll_sprint_self_right_upright"].weight == 5.0
    assert cfg.rewards["roll_sprint_self_right_height"].weight == 30.0
    assert cfg.rewards["roll_sprint_self_right_upward"].weight == 0.0
    assert cfg.rewards["roll_sprint_self_right_fallen_tax"].weight == 0.0
    assert cfg.rewards["roll_sprint_self_right_success"].weight == 0.0
    assert (
        cfg.rewards["roll_sprint_recovered_reroll"].weight
        < cfg.rewards["roll_sprint_distance"].weight
    )
    assert cfg.rewards["roll_sprint_invalid_cycle"].weight == 0.0
    assert cfg.rewards["roll_sprint_sagittal"].weight == -0.05
    assert cfg.rewards["roll_sprint_flatness"].weight == -0.25
    assert cfg.rewards["roll_sprint_lateral_vel"].weight == -0.35
    assert cfg.rewards["roll_sprint_straightness"].weight == -3.0
    assert cfg.rewards["roll_sprint_lane_centering"].weight == 4.0
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
    reset_params = cfg.events["set_roll_sprint_state"].params
    assert reset_params["standing_prob"] == 0.45
    assert reset_params["midroll_prob"] == 0.10
    assert reset_params["postroll_prob"] == 0.15
    assert reset_params["crouch_prob"] == 0.15
    assert reset_params["ground_recovery_prob"] == 0.15
    assert cfg.curriculum["roll_sprint_lane_half_width"].params[
        "width_stages"
    ] == [
        {"step": 0, "width": 2.0},
        {"step": 1000 * 24, "width": 0.60},
        {"step": 2000 * 24, "width": 0.40},
        {"step": 2800 * 24, "width": 0.28},
        {"step": 3400 * 24, "width": 0.20},
        {"step": 3750 * 24, "width": 0.14},
    ]
    play_cfg = make_microduck_roll_sprint_env_cfg(play=True)
    assert play_cfg.curriculum["roll_sprint_lane_half_width"].params[
        "width_stages"
    ] == [{"step": 0, "width": 0.14}]
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
        (0.45, 0.10, 0.15, 0.15, 0.15),
        (0.45, 0.05, 0.15, 0.15, 0.20),
        (0.55, 0.05, 0.10, 0.10, 0.20),
        (0.65, 0.00, 0.10, 0.05, 0.20),
    ]
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
        (0.70, 0.10, 0.10, 0.10),
        (0.55, 0.15, 0.15, 0.15),
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
        {"step": 0, "weight": 0.0},
        {"step": 400 * 24, "weight": 1.0},
        {"step": 1000 * 24, "weight": 2.0},
    ]
    assert cfg.curriculum["roll_sprint_self_right_fallen_tax_weight"].params[
        "weight_stages"
    ][-1] == {"step": 1000 * 24, "weight": -0.5}
    assert cfg.curriculum["roll_sprint_self_right_success_weight"].params[
        "weight_stages"
    ][-1] == {"step": 1000 * 24, "weight": 10.0}
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


def test_roll_sprint_lateral_displacement_has_no_credit_and_costs_straightness(
    monkeypatch,
):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    _prime_roll_heading(env, asset)
    _complete_valid_roll(env, asset, forward=0.0, lateral=0.25)
    env._roll_sprint_progress_delta[:] = env.step_dt * mdp._ROLL_SPRINT_TARGET_ANGLE

    penalty = mdp.roll_sprint_straightness_penalty(env, deadband=0.01)

    assert env._roll_sprint_completed_distance[0] == 0.0
    assert penalty[0] == pytest.approx(0.24)
    env._roll_sprint_progress_delta.zero_()
    assert mdp.roll_sprint_straightness_penalty(env, deadband=0.01)[0] == 0.0


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


def test_roll_sprint_reposition_command_points_back_to_lane_center(monkeypatch):
    env, asset = _fake_env(3)
    _enable_flat_valid_roll(monkeypatch, env)
    base_command = torch.tensor(
        [[0.01, 0.02, 0.03], [0.01, 0.02, 0.03], [0.01, 0.02, 0.03]]
    )
    env.command_manager = SimpleNamespace(
        get_command=lambda command_name: base_command,
    )
    _prime_roll_heading(env, asset)
    asset.data.root_link_pos_w[:, 1] = torch.tensor([0.15, -0.15, 0.02])

    command = mdp.roll_sprint_reposition_command(env)

    assert torch.allclose(command[0], torch.tensor([0.0, -0.20, 0.0]))
    assert torch.allclose(command[1], torch.tensor([0.0, 0.20, 0.0]))
    assert torch.allclose(command[2], torch.zeros(3))


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
    assert env._roll_sprint_recovery_count[0] == 1.0
    assert env._roll_sprint_reposition_count[0] == 1.0
    assert env._roll_sprint_forward_frontier[0] == pytest.approx(0.20)

    env.common_step_counter += 1
    _complete_valid_roll(env, asset, forward=0.45)
    assert env._roll_sprint_completed[0] == 2.0
    assert env._roll_sprint_recovered_and_rerolled[0] == 1.0


def test_roll_sprint_training_lane_gate_tightens_to_canonical_width():
    env, _asset = _fake_env(1)
    stages = [
        {"step": 0, "width": 2.0},
        {"step": 250 * 24, "width": 0.40},
        {"step": 1750 * 24, "width": 0.28},
        {"step": 2750 * 24, "width": 0.20},
        {"step": 3500 * 24, "width": 0.14},
    ]

    for step, expected in (
        (0, 2.0),
        (250 * 24, 0.40),
        (1750 * 24, 0.28),
        (2750 * 24, 0.20),
        (3500 * 24, 0.14),
    ):
        env.common_step_counter = step
        reported = mdp.roll_sprint_lane_half_width_curriculum(
            env,
            torch.tensor([0]),
            stages,
        )
        assert reported[0] == pytest.approx(expected)
        assert env._roll_sprint_lane_half_width_m == pytest.approx(expected)


def test_roll_progress_reward_fades_to_zero_at_lane_edge_without_idle_annuity():
    env, _asset = _fake_env(4)
    mdp._roll_sprint_state(env)
    env._roll_sprint_last_update_step[:] = env.common_step_counter
    env._roll_sprint_progress_delta[:] = env.step_dt * mdp._ROLL_SPRINT_TARGET_ANGLE
    env._roll_sprint_lateral_displacement[:] = torch.tensor([0.0, 0.07, 0.14, 0.25])

    reward = mdp.roll_sprint_progress(
        env,
        max_paid_rate=10.0,
        lane_half_width=0.14,
    )

    assert reward.tolist() == pytest.approx([1.0, 0.75, 0.0, 0.0])
    env._roll_sprint_progress_delta.zero_()
    assert torch.equal(
        mdp.roll_sprint_progress(
            env,
            max_paid_rate=10.0,
            lane_half_width=0.14,
        ),
        torch.zeros(4),
    )


def test_roll_progress_uses_active_training_lane_width_curriculum():
    env, _asset = _fake_env(3)
    mdp._roll_sprint_state(env)
    env._roll_sprint_last_update_step[:] = env.common_step_counter
    env._roll_sprint_progress_delta[:] = env.step_dt * mdp._ROLL_SPRINT_TARGET_ANGLE
    env._roll_sprint_lateral_displacement[:] = torch.tensor([0.0, 0.20, 0.40])
    env._roll_sprint_lane_half_width_m = 0.40

    reward = mdp.roll_sprint_progress(
        env,
        max_paid_rate=10.0,
        lane_half_width=0.14,
    )

    assert reward.tolist() == pytest.approx([1.0, 0.75, 0.0])


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
        lane_half_width=0.14,
    )

    assert reward.tolist() == pytest.approx([1.0, 0.0, 1.0, 0.0, 0.0])


def test_lane_centering_progress_penalizes_departure_rewards_correction_and_not_idle(
    monkeypatch,
):
    env, asset = _fake_env(1)
    _enable_flat_valid_roll(monkeypatch, env)
    mdp._update_roll_sprint_state(env, asset)

    env.common_step_counter += 1
    asset.data.root_link_pos_w[:, 1] = 0.10
    departure = mdp.roll_sprint_lane_centering_progress(env)

    env.common_step_counter += 1
    asset.data.root_link_pos_w[:, 1] = 0.04
    correction = mdp.roll_sprint_lane_centering_progress(env)

    env.common_step_counter += 1
    idle = mdp.roll_sprint_lane_centering_progress(env)

    assert departure[0] == pytest.approx(-0.10 / env.step_dt)
    assert correction[0] == pytest.approx(0.06 / env.step_dt)
    assert idle[0] == 0.0
    assert env.step_dt * (departure[0] + correction[0] + idle[0]) == pytest.approx(
        -0.04
    )


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
