import math
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_backroll_env_cfg import (
    BACKROLL_CURRICULUM_STAGES,
    EPISODE_LENGTH_S,
    REPEATED_BACKROLL_CURRICULUM_STAGES,
    REPEATED_EPISODE_LENGTH_S,
    MicroduckBackrollRlCfg,
    MicroduckRepeatedBackrollRlCfg,
    make_microduck_backroll_env_cfg,
    make_microduck_repeated_backroll_env_cfg,
)
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    make_microduck_roulade_env_cfg,
)


class _Scene:
    def __init__(self, asset, sensors):
        self.asset = asset
        self.sensors = sensors
        self.terrain = SimpleNamespace(env_origins=torch.zeros(1, 3))

    def __getitem__(self, name):
        assert name == "robot"
        return self.asset


def _fake_env():
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    asset = SimpleNamespace(
        data=SimpleNamespace(
            root_link_ang_vel_b=torch.zeros(1, 3),
            root_link_lin_vel_b=torch.zeros(1, 3),
            root_link_lin_vel_w=torch.zeros(1, 3),
            root_link_pos_w=torch.tensor([[0.0, 0.0, 0.115]]),
            root_link_quat_w=quat,
        )
    )
    sensors = {
        "robot_ground_contact": SimpleNamespace(
            data=SimpleNamespace(found=torch.ones(1, 1))
        ),
        "trunk_ground_contact": SimpleNamespace(
            data=SimpleNamespace(found=torch.zeros(1, 1))
        ),
        "head_ground_contact": SimpleNamespace(
            data=SimpleNamespace(found=torch.zeros(1, 1))
        ),
        "left_foot_ground_contact": SimpleNamespace(
            data=SimpleNamespace(found=torch.ones(1, 1))
        ),
        "right_foot_ground_contact": SimpleNamespace(
            data=SimpleNamespace(found=torch.ones(1, 1))
        ),
    }
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        step_dt=0.02,
        common_step_counter=0,
        scene=_Scene(asset, sensors),
    )
    mdp._grounded_backroll_state(env)
    env._roulade_roll_direction[:] = -1.0
    return env, asset


def _next_step(env, asset, omega_y):
    asset.data.root_link_ang_vel_b[:, 1] = omega_y
    value = mdp.grounded_backroll_progress(env)
    env.common_step_counter += 1
    return value


def test_backroll_is_one_shot_roulade_without_sprint_objectives():
    cfg = make_microduck_backroll_env_cfg()
    roulade = make_microduck_roulade_env_cfg()

    assert cfg.episode_length_s == EPISODE_LENGTH_S == 5.0
    assert list(cfg.observations["actor"].terms) == list(
        roulade.observations["actor"].terms
    )
    assert set(cfg.rewards) == {
        "backroll_progress",
        "backroll_head_pivot",
        "backroll_completion_progress",
        "backroll_upright_progress",
        "backroll_height_progress",
        "backroll_success",
        "backroll_invalid",
        "backroll_overspeed",
        "backroll_sagittal",
        "backroll_lateral_velocity",
        "backroll_flatness",
        "action_rate_l2",
        "gentle_landing",
        "self_collisions",
    }
    forbidden = ("sprint", "distance", "lane", "road", "recovery", "reposition")
    assert not any(token in name for name in cfg.rewards for token in forbidden)
    assert cfg.events["set_grounded_backroll_state"].func is mdp.reset_grounded_backroll_state
    assert cfg.curriculum["backroll_phase"].func is mdp.grounded_backroll_curriculum
    assert MicroduckBackrollRlCfg.experiment_name == "microduck_backroll"
    assert MicroduckBackrollRlCfg.max_iterations == 4000
    assert MicroduckBackrollRlCfg.save_interval == 50
    assert MicroduckBackrollRlCfg.algorithm.learning_rate == pytest.approx(1.0e-3)
    assert MicroduckBackrollRlCfg.actor.distribution_cfg["init_std"] == 1.0


def test_backroll_play_is_deterministic_standing_start():
    cfg = make_microduck_backroll_env_cfg(play=True)
    reset = cfg.events["set_grounded_backroll_state"].params

    assert reset["standing_prob"] == 1.0
    assert reset["midroll_prob"] == 0.0
    assert reset["yaw_range"] == (0.0, 0.0)
    assert reset["joint_noise_std"] == 0.0
    assert "backroll_phase" not in cfg.curriculum


def test_repeated_backroll_rearms_without_adding_course_objectives():
    cfg = make_microduck_repeated_backroll_env_cfg()

    assert cfg.episode_length_s == REPEATED_EPISODE_LENGTH_S == 12.0
    assert "backroll_success" not in cfg.terminations
    assert cfg.events["set_grounded_backroll_state"].params["repeat_mode"] is True
    assert cfg.rewards["backroll_speed_progress"].func is mdp.grounded_backroll_speed_progress
    assert cfg.rewards["backroll_rise_velocity"].func is mdp.grounded_backroll_rise_velocity
    assert (
        cfg.rewards["backroll_contact_sequence"].func
        is mdp.grounded_backroll_contact_sequence
    )
    assert cfg.rewards["backroll_success"].func is mdp.grounded_backroll_repeat_success_rate
    assert cfg.metrics["backroll_cycle_count"].func is mdp.grounded_backroll_cycle_count
    forbidden = ("sprint", "distance", "lane", "road", "recovery", "reposition")
    assert not any(token in name for name in cfg.rewards for token in forbidden)
    assert MicroduckRepeatedBackrollRlCfg.experiment_name == "microduck_repeated_backroll"
    assert MicroduckRepeatedBackrollRlCfg.algorithm.learning_rate == pytest.approx(2.5e-5)
    assert MicroduckRepeatedBackrollRlCfg.algorithm.entropy_coef == pytest.approx(1.0e-3)
    assert MicroduckRepeatedBackrollRlCfg.actor.distribution_cfg[
        "init_std"
    ] == pytest.approx(0.25)
    assert [
        stage["params"]["standing_prob"]
        for stage in REPEATED_BACKROLL_CURRICULUM_STAGES
    ] == [0.65, 0.70, 0.75, 0.85, 0.95, 1.0]
    assert [
        stage["params"]["mastery_cycles"]
        for stage in REPEATED_BACKROLL_CURRICULUM_STAGES
    ] == [1, 1, 1, 2, 2, 3]
    assert REPEATED_BACKROLL_CURRICULUM_STAGES[0]["params"][
        "midroll_pitch_min"
    ] == pytest.approx(math.radians(20.0))
    assert REPEATED_BACKROLL_CURRICULUM_STAGES[0]["params"][
        "midroll_pitch_max"
    ] == pytest.approx(math.radians(29.0))
    assert REPEATED_BACKROLL_CURRICULUM_STAGES[0]["params"][
        "midroll_omega_range"
    ] == (1.0, 3.0)
    assert REPEATED_BACKROLL_CURRICULUM_STAGES[0]["params"][
        "midroll_z_min"
    ] == pytest.approx(0.075)
    assert REPEATED_BACKROLL_CURRICULUM_STAGES[0]["params"][
        "midroll_z_max"
    ] == pytest.approx(0.075)
    assert REPEATED_BACKROLL_CURRICULUM_STAGES[0]["params"][
        "tuck_factor_range"
    ] == (1.0, 1.0)
    assert REPEATED_BACKROLL_CURRICULUM_STAGES[0]["params"][
        "midroll_pitch_max"
    ] < math.radians(30.0)
    assert cfg.curriculum["backroll_phase"].params["success_threshold"] == pytest.approx(
        0.55
    )
    assert cfg.rewards["backroll_completion_progress"].weight > cfg.rewards[
        "backroll_progress"
    ].weight
    assert cfg.rewards["backroll_success"].params["later_cycle_bonus"] == pytest.approx(
        1.0
    )
    assert cfg.rewards["backroll_speed_progress"].weight == pytest.approx(1.0)
    assert cfg.rewards["backroll_speed_progress"].params == {
        "minimum_rate": 2.0,
        "target_rate": 6.0,
    }
    assert cfg.rewards["backroll_invalid"].weight == pytest.approx(-2.0)
    assert cfg.curriculum["backroll_phase"].params["speed_reward_weights"] == [
        1.0,
        1.0,
        1.5,
        2.0,
        3.0,
        3.0,
    ]
    assert cfg.curriculum["backroll_phase"].params["invalid_reward_weights"] == [
        -2.0,
        -3.0,
        -4.0,
        -6.0,
        -8.0,
        -10.0,
    ]


def test_backroll_curriculum_matches_mastery_stages():
    assert [stage["params"]["standing_prob"] for stage in BACKROLL_CURRICULUM_STAGES] == [
        0.20,
        0.30,
        0.40,
        0.60,
        0.85,
    ]
    assert [stage["params"]["midroll_prob"] for stage in BACKROLL_CURRICULUM_STAGES] == [
        0.80,
        0.70,
        0.60,
        0.40,
        0.15,
    ]
    assert BACKROLL_CURRICULUM_STAGES[0]["params"]["midroll_pitch_min"] == pytest.approx(
        math.radians(180.0)
    )
    assert BACKROLL_CURRICULUM_STAGES[0]["params"]["midroll_omega_range"] == (
        1.0,
        3.0,
    )
    cfg = make_microduck_backroll_env_cfg()
    params = cfg.curriculum["backroll_phase"].params
    assert params["window_episodes"] == 4096
    assert params["success_threshold"] == pytest.approx(0.70)


def test_reverse_phase_reset_uses_negative_pitch_rate_and_contact_prerequisites(
    monkeypatch,
):
    env, _asset = _fake_env()
    env.sim = SimpleNamespace(
        data=SimpleNamespace(
            qpos=torch.zeros(1, 21),
            qvel=torch.zeros(1, 20),
        )
    )
    env.sim.data.qpos[:, 3] = 1.0
    monkeypatch.setattr(mdp, "_servo_joint_ids", lambda _env, _asset: list(range(14)))
    pitch = math.radians(200.0)

    mdp.reset_grounded_backroll_state(
        env,
        torch.tensor([0]),
        standing_prob=0.0,
        midroll_prob=1.0,
        midroll_pitch_min=pitch,
        midroll_pitch_max=pitch,
        midroll_omega_range=(3.0, 3.0),
        standing_tilt_max=0.0,
        yaw_range=(0.0, 0.0),
        joint_noise_std=0.0,
    )

    assert env._roulade_roll_direction.item() == -1.0
    assert env.sim.data.qpos[0, 5].item() < 0.0
    assert env.sim.data.qvel[0, 4].item() == pytest.approx(-3.0)
    assert env._backroll_trunk_latch.item()
    assert env._backroll_head_latch.item()


def test_repeated_curriculum_counts_the_stage_mastery_cycle_target(monkeypatch):
    env, _asset = _fake_env()
    env.sim = SimpleNamespace(
        data=SimpleNamespace(qpos=torch.zeros(1, 21), qvel=torch.zeros(1, 20))
    )
    env.sim.data.qpos[:, 3] = 1.0
    monkeypatch.setattr(mdp, "_servo_joint_ids", lambda _env, _asset: list(range(14)))
    env._backroll_started[:] = True
    env._backroll_repeat_mode[:] = True
    env._backroll_cycle_count[:] = 1
    env._backroll_mastery_cycles[:] = 2

    mdp.reset_grounded_backroll_state(
        env,
        torch.tensor([0]),
        standing_prob=1.0,
        midroll_prob=0.0,
        repeat_mode=True,
        mastery_cycles=1,
        yaw_range=(0.0, 0.0),
        joint_noise_std=0.0,
    )
    assert env._backroll_window_successes.item() == 0

    env._backroll_cycle_count[:] = 1
    mdp.reset_grounded_backroll_state(
        env,
        torch.tensor([0]),
        standing_prob=1.0,
        midroll_prob=0.0,
        repeat_mode=True,
        mastery_cycles=1,
        yaw_range=(0.0, 0.0),
        joint_noise_std=0.0,
    )
    assert env._backroll_window_successes.item() == 1


def test_negative_body_y_advances_but_forward_rocking_cannot_farm(monkeypatch):
    env, asset = _fake_env()
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )

    first = _next_step(env, asset, -1.0)
    frontier = env._roulade_max.clone()
    backward = _next_step(env, asset, 1.0)
    revisit = _next_step(env, asset, -1.0)
    extension = _next_step(env, asset, -1.0)

    assert first.item() > 0.0
    assert backward.item() == 0.0
    assert revisit.item() == 0.0
    assert torch.equal(env._roulade_max - frontier, torch.tensor([0.02]))
    assert extension.item() > 0.0


def test_completion_push_requires_head_latch_and_only_pays_new_frontier(monkeypatch):
    env, asset = _fake_env()
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )
    env._roulade_accum[:] = math.radians(180.0)
    env._roulade_max[:] = math.radians(180.0)
    env._backroll_completion_paid[:] = math.radians(180.0)

    asset.data.root_link_ang_vel_b[:, 1] = -2.0
    assert mdp.grounded_backroll_completion_progress(env).item() == 0.0
    env.common_step_counter += 1

    env._backroll_trunk_latch[:] = True
    env._backroll_head_latch[:] = True
    asset.data.root_link_ang_vel_b[:, 1] = -2.0
    first_extension = mdp.grounded_backroll_completion_progress(env)
    env.common_step_counter += 1
    asset.data.root_link_ang_vel_b[:, 1] = 2.0
    forward_rock = mdp.grounded_backroll_completion_progress(env)
    env.common_step_counter += 1
    asset.data.root_link_ang_vel_b[:, 1] = -2.0
    revisit = mdp.grounded_backroll_completion_progress(env)
    env.common_step_counter += 1
    asset.data.root_link_ang_vel_b[:, 1] = -2.0
    second_extension = mdp.grounded_backroll_completion_progress(env)

    assert first_extension.item() > 0.0
    assert forward_rock.item() == 0.0
    assert revisit.item() == 0.0
    assert second_extension.item() > 0.0


def test_completion_push_rejects_airborne_rotation(monkeypatch):
    env, asset = _fake_env()
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )
    env._backroll_trunk_latch[:] = True
    env._backroll_head_latch[:] = True
    env._roulade_accum[:] = math.radians(180.0)
    env._roulade_max[:] = math.radians(180.0)
    env._backroll_completion_paid[:] = math.radians(180.0)
    env.scene.sensors["robot_ground_contact"].data.found[:] = 0.0
    asset.data.root_link_ang_vel_b[:, 1] = -3.0

    assert mdp.grounded_backroll_completion_progress(env).item() == 0.0


def test_speed_progress_requires_fast_new_backward_frontier(monkeypatch):
    env, asset = _fake_env()
    env._backroll_repeat_mode[:] = True
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )

    asset.data.root_link_ang_vel_b[:, 1] = -4.5
    first = mdp.grounded_backroll_speed_progress(env)
    env.common_step_counter += 1
    asset.data.root_link_ang_vel_b[:, 1] = 4.5
    rocking = mdp.grounded_backroll_speed_progress(env)
    env.common_step_counter += 1
    asset.data.root_link_ang_vel_b[:, 1] = -4.5
    revisit = mdp.grounded_backroll_speed_progress(env)

    assert first.item() > 0.0
    assert rocking.item() == 0.0
    assert revisit.item() == 0.0


def test_repeated_progress_bridges_contact_windows_then_requires_latches(monkeypatch):
    env, asset = _fake_env()
    env._backroll_repeat_mode[:] = True
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )

    asset.data.root_link_ang_vel_b[:, 1] = -4.0
    # The policy still receives a local gradient while trunk contact remains
    # physically attainable inside its valid phase window.
    env._roulade_accum[:] = math.radians(40.0)
    env._roulade_max[:] = math.radians(40.0)
    env._roulade_paid[:] = math.radians(35.0)
    env._backroll_previous_frontier[:] = math.radians(35.0)
    assert mdp.grounded_backroll_progress(env).item() > 0.0
    assert mdp.grounded_backroll_speed_progress(env).item() > 0.0

    # Passing the trunk window without a latch closes all further progress.
    env._roulade_accum[:] = math.radians(265.0)
    env._roulade_max[:] = math.radians(265.0)
    env._roulade_paid[:] = math.radians(260.0)
    env._backroll_previous_frontier[:] = math.radians(260.0)
    env.common_step_counter += 1
    assert mdp.grounded_backroll_progress(env).item() == 0.0
    assert mdp.grounded_backroll_speed_progress(env).item() == 0.0

    # A trunk latch reopens the bridge through the head-contact window.
    env._backroll_trunk_latch[:] = True
    env._roulade_accum[:] = math.radians(275.0)
    env._roulade_max[:] = math.radians(275.0)
    env._roulade_paid[:] = math.radians(270.0)
    env._backroll_previous_frontier[:] = math.radians(270.0)
    env.common_step_counter += 1
    assert mdp.grounded_backroll_progress(env).item() > 0.0
    assert mdp.grounded_backroll_speed_progress(env).item() > 0.0

    # Passing the head window without the ordered flat-top latch closes it.
    env._roulade_accum[:] = math.radians(305.0)
    env._roulade_max[:] = math.radians(305.0)
    env._roulade_paid[:] = math.radians(300.0)
    env._backroll_previous_frontier[:] = math.radians(300.0)
    env.common_step_counter += 1
    assert mdp.grounded_backroll_progress(env).item() == 0.0
    assert mdp.grounded_backroll_speed_progress(env).item() == 0.0

    env._backroll_head_latch[:] = True
    env._roulade_accum[:] = math.radians(310.0)
    env._roulade_max[:] = math.radians(310.0)
    env._roulade_paid[:] = math.radians(305.0)
    env._backroll_previous_frontier[:] = math.radians(305.0)
    env.common_step_counter += 1
    assert mdp.grounded_backroll_progress(env).item() > 0.0
    assert mdp.grounded_backroll_speed_progress(env).item() > 0.0


def test_repeated_positive_rewards_stop_after_cumulative_offaxis_escape(monkeypatch):
    env, asset = _fake_env()
    env._backroll_repeat_mode[:] = True
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )
    asset.data.root_link_ang_vel_b[:, 1] = -6.0
    env._roulade_accum[:] = math.radians(200.0)
    env._roulade_max[:] = math.radians(200.0)
    env._roulade_paid[:] = math.radians(195.0)
    env._backroll_previous_frontier[:] = math.radians(195.0)
    env._backroll_trunk_latch[:] = True
    env._backroll_head_latch[:] = True
    env._backroll_cycle_offaxis_rotation[:] = (
        mdp._BACKROLL_REPEAT_MAX_OFFAXIS_ROTATION + math.radians(1.0)
    )

    assert mdp.grounded_backroll_progress(env).item() == 0.0
    assert mdp.grounded_backroll_invalid_termination(env).item()
    assert mdp.grounded_backroll_speed_progress(env).item() == 0.0

    env._backroll_completion_paid[:] = math.radians(195.0)
    assert mdp.grounded_backroll_completion_progress(env).item() == 0.0

    env.scene.sensors["trunk_ground_contact"].data.found[:] = 1.0
    env._backroll_trunk_latch[:] = False
    env._backroll_head_latch[:] = False
    env._roulade_accum[:] = math.radians(40.0)
    env._roulade_max[:] = math.radians(40.0)
    env.common_step_counter += 1
    assert mdp.grounded_backroll_contact_sequence(env).item() == 0.0


def test_backward_progress_is_scaled_by_instantaneous_sagittal_purity():
    _env, asset = _fake_env()
    asset.data.root_link_quat_w[:] = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    asset.data.root_link_ang_vel_b[:] = torch.tensor([[0.0, -6.0, 0.0]])
    pure = mdp._grounded_backroll_sagittal_purity(asset)
    asset.data.root_link_ang_vel_b[:] = torch.tensor([[6.0, -6.0, 0.0]])
    mixed = mdp._grounded_backroll_sagittal_purity(asset)
    asset.data.root_link_ang_vel_b[:] = torch.tensor([[6.0, 0.0, 0.0]])
    wrong_axis = mdp._grounded_backroll_sagittal_purity(asset)

    assert pure.item() == pytest.approx(1.0)
    assert mixed.item() == pytest.approx(0.5)
    assert wrong_axis.item() == 0.0


def test_backward_purity_fades_before_the_robot_can_side_roll():
    _env, asset = _fake_env()
    asset.data.root_link_ang_vel_b[:] = torch.tensor([[0.0, -6.0, 0.0]])

    def roll_quaternion(degrees: float) -> torch.Tensor:
        half = math.radians(degrees) * 0.5
        return torch.tensor([[math.cos(half), math.sin(half), 0.0, 0.0]])

    asset.data.root_link_quat_w[:] = roll_quaternion(10.0)
    clean = mdp._grounded_backroll_sagittal_purity(asset)
    asset.data.root_link_quat_w[:] = roll_quaternion(20.0)
    fading = mdp._grounded_backroll_sagittal_purity(asset)
    asset.data.root_link_quat_w[:] = roll_quaternion(30.0)
    escaped = mdp._grounded_backroll_sagittal_purity(asset)

    assert clean.item() == pytest.approx(1.0)
    assert 0.0 < fading.item() < 1.0
    assert escaped.item() == pytest.approx(0.0, abs=1.0e-6)


def test_ordered_contact_rewards_are_one_shot_latches(monkeypatch):
    env, _asset = _fake_env()
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )
    env.scene.sensors["trunk_ground_contact"].data.found[:] = 1.0
    env._roulade_accum[:] = math.radians(40.0)
    env._roulade_max[:] = math.radians(40.0)

    trunk_pulse = mdp.grounded_backroll_contact_sequence(env)
    env.common_step_counter += 1
    trunk_repeat = mdp.grounded_backroll_contact_sequence(env)
    assert trunk_pulse.item() > 0.0
    assert trunk_repeat.item() == 0.0

    env.scene.sensors["head_ground_contact"].data.found[:] = 1.0
    env._roulade_accum[:] = math.radians(120.0)
    env._roulade_max[:] = math.radians(120.0)
    env.common_step_counter += 1
    head_pulse = mdp.grounded_backroll_contact_sequence(env)
    env.common_step_counter += 1
    head_repeat = mdp.grounded_backroll_contact_sequence(env)
    assert head_pulse.item() > trunk_pulse.item()
    assert head_repeat.item() == 0.0


def test_airborne_and_sideways_rotation_receive_no_progress(monkeypatch):
    env, asset = _fake_env()
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )
    env.scene.sensors["robot_ground_contact"].data.found[:] = 0.0
    assert _next_step(env, asset, -2.0).item() == 0.0

    env.scene.sensors["robot_ground_contact"].data.found[:] = 1.0
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.ones(1))
    assert _next_step(env, asset, -2.0).item() == 0.0


def test_head_contact_only_latches_after_trunk_contact(monkeypatch):
    env, _asset = _fake_env()
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )
    env._roulade_accum[:] = math.radians(150.0)
    env._roulade_max[:] = math.radians(150.0)
    env.scene.sensors["head_ground_contact"].data.found[:] = 1.0
    mdp.grounded_backroll_progress(env)
    assert not env._backroll_head_latch.item()

    env.common_step_counter += 1
    env._roulade_accum[:] = math.radians(50.0)
    env._roulade_max[:] = math.radians(50.0)
    env.scene.sensors["head_ground_contact"].data.found[:] = 0.0
    env.scene.sensors["trunk_ground_contact"].data.found[:] = 1.0
    mdp.grounded_backroll_progress(env)
    assert env._backroll_trunk_latch.item()

    env.common_step_counter += 1
    env._roulade_accum[:] = math.radians(150.0)
    env._roulade_max[:] = math.radians(150.0)
    env.scene.sensors["head_ground_contact"].data.found[:] = 1.0
    mdp.grounded_backroll_progress(env)
    assert env._backroll_head_latch.item()


def test_airborne_gap_and_wrong_way_completion_are_invalid(monkeypatch):
    env, asset = _fake_env()
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )
    env.scene.sensors["robot_ground_contact"].data.found[:] = 0.0
    for _ in range(math.ceil(mdp._BACKROLL_MAX_AIR_SECONDS / env.step_dt) + 1):
        _next_step(env, asset, -2.0)
    assert mdp.grounded_backroll_invalid_termination(env).item()

    env, asset = _fake_env()
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    for _ in range(46):
        _next_step(env, asset, 2.0)
    assert mdp.grounded_backroll_invalid_termination(env).item()


def test_invalid_terminal_cost_is_one_shot(monkeypatch):
    env, _asset = _fake_env()
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )
    env._roulade_accum[:] = -math.radians(91.0)
    env._roulade_max[:] = 0.0

    first = mdp.grounded_backroll_invalid_rate(env)
    env.common_step_counter += 1
    second = mdp.grounded_backroll_invalid_rate(env)

    assert first.item() == pytest.approx(1.0 / env.step_dt)
    assert second.item() == 0.0


def test_success_requires_ordered_contacts_and_is_one_shot(monkeypatch):
    env, _asset = _fake_env()
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )
    env._roulade_accum[:] = math.radians(355.0)
    env._roulade_max[:] = math.radians(355.0)
    env._roulade_paid[:] = math.radians(355.0)
    env._backroll_previous_frontier[:] = math.radians(355.0)

    hold_steps = math.ceil(mdp._BACKROLL_LANDING_HOLD_SECONDS / env.step_dt)
    for _ in range(hold_steps + 1):
        value = mdp.grounded_backroll_success_rate(env)
        assert value.item() == 0.0
        env.common_step_counter += 1

    env._backroll_trunk_latch[:] = True
    env._backroll_head_latch[:] = True
    pulses = []
    for _ in range(hold_steps + 2):
        pulses.append(mdp.grounded_backroll_success_rate(env).item())
        env.common_step_counter += 1

    assert sum(value > 0.0 for value in pulses) == 1
    assert env._backroll_success.item()


def test_repeated_backroll_rearms_and_credits_two_distinct_cycles(monkeypatch):
    env, _asset = _fake_env()
    env._backroll_repeat_mode[:] = True
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )
    hold_steps = math.ceil(
        mdp._BACKROLL_REPEAT_LANDING_HOLD_SECONDS / env.step_dt
    )

    pulses = []
    for expected_count in (1, 2):
        env._roulade_accum[:] = math.radians(355.0)
        env._roulade_max[:] = math.radians(355.0)
        env._roulade_paid[:] = math.radians(355.0)
        env._backroll_previous_frontier[:] = math.radians(355.0)
        env._backroll_trunk_latch[:] = True
        env._backroll_head_latch[:] = True
        for _ in range(hold_steps + 1):
            pulses.append(mdp.grounded_backroll_repeat_success_rate(env).item())
            env.common_step_counter += 1
        assert env._backroll_cycle_count.item() == expected_count
        assert env._roulade_max.item() == 0.0
        assert not env._backroll_trunk_latch.item()
        assert not env._backroll_head_latch.item()
        assert not env._backroll_success.item()

    assert sum(value > 0.0 for value in pulses) == 2


def test_repeated_backroll_rejects_side_landing(monkeypatch):
    env, _asset = _fake_env()
    env._backroll_repeat_mode[:] = True
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.full((1,), 0.7))
    env._roulade_accum[:] = math.radians(355.0)
    env._roulade_max[:] = math.radians(355.0)
    env._backroll_previous_frontier[:] = math.radians(355.0)
    env._backroll_trunk_latch[:] = True
    env._backroll_head_latch[:] = True

    hold_steps = math.ceil(
        mdp._BACKROLL_REPEAT_LANDING_HOLD_SECONDS / env.step_dt
    )
    for _ in range(hold_steps + 2):
        assert mdp.grounded_backroll_repeat_success_rate(env).item() == 0.0
        env.common_step_counter += 1

    assert env._backroll_cycle_count.item() == 0


def test_repeated_backroll_cannot_park_on_trunk_mid_cycle(monkeypatch):
    env, asset = _fake_env()
    env._backroll_repeat_mode[:] = True
    env.scene.sensors["left_foot_ground_contact"].data.found[:] = 0.0
    env.scene.sensors["right_foot_ground_contact"].data.found[:] = 0.0
    env.scene.sensors["trunk_ground_contact"].data.found[:] = 1.0
    env._roulade_accum[:] = math.radians(180.0)
    env._roulade_max[:] = math.radians(180.0)
    env._backroll_previous_frontier[:] = math.radians(180.0)
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )

    stall_steps = math.ceil(
        mdp._BACKROLL_REPEAT_PRE_EXIT_STALL_SECONDS / env.step_dt
    )
    for _ in range(stall_steps):
        asset.data.root_link_ang_vel_b.zero_()
        mdp.grounded_backroll_progress(env)
        env.common_step_counter += 1

    assert mdp.grounded_backroll_invalid_termination(env).item()


def test_repeated_backroll_gets_a_bounded_post_350_landing_budget(monkeypatch):
    env, asset = _fake_env()
    env._backroll_repeat_mode[:] = True
    env.scene.sensors["left_foot_ground_contact"].data.found[:] = 0.0
    env.scene.sensors["right_foot_ground_contact"].data.found[:] = 0.0
    env._roulade_accum[:] = math.radians(355.0)
    env._roulade_max[:] = math.radians(355.0)
    env._backroll_previous_frontier[:] = math.radians(355.0)
    env._backroll_trunk_latch[:] = True
    env._backroll_head_latch[:] = True
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )

    timeout_steps = math.ceil(
        mdp._BACKROLL_REPEAT_LANDING_TIMEOUT_SECONDS / env.step_dt
    )
    for _ in range(timeout_steps - 2):
        asset.data.root_link_ang_vel_b.zero_()
        mdp.grounded_backroll_progress(env)
        env.common_step_counter += 1
    assert not mdp.grounded_backroll_invalid_termination(env).item()

    env.common_step_counter += 1
    mdp.grounded_backroll_progress(env)
    assert mdp.grounded_backroll_invalid_termination(env).item()


def test_standing_cannot_farm_backroll_success(monkeypatch):
    env, _asset = _fake_env()
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )
    for _ in range(25):
        assert mdp.grounded_backroll_success_rate(env).item() == 0.0
        env.common_step_counter += 1
    assert not env._backroll_success.item()


def test_forward_roulade_direction_default_is_unchanged(monkeypatch):
    env, asset = _fake_env()
    env._roulade_roll_direction[:] = 1.0
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )
    asset.data.root_link_ang_vel_b[:, 1] = 1.0

    mdp._update_roulade_accum(env, asset)

    assert env._roulade_accum.item() == pytest.approx(0.02)
