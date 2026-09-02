import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from mjlab.tasks.registry import load_runner_cls

from mjlab_microduck.tasks import MicroduckFrozenActorNormRunner, mdp
from mjlab_microduck.tasks.backroll_ppo import AnchoredPPO
from mjlab_microduck.tasks.microduck_backroll_env_cfg import (
    BACKROLL_CURRICULUM_STAGES,
    BACKROLL_TUCK_OVERRIDES,
    EPISODE_LENGTH_S,
    REPEATED_BACKROLL_CURRICULUM_STAGES,
    REPEATED_EPISODE_LENGTH_S,
    MicroduckBackrollRlCfg,
    MicroduckRepeatedBackrollRlCfg,
    make_microduck_backroll_env_cfg,
    make_microduck_repeated_backroll_env_cfg,
)
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    TUCK_OVERRIDES,
    make_microduck_roulade_env_cfg,
)


class _Scene:
    def __init__(self, asset, sensors, num_envs=1):
        self.asset = asset
        self.sensors = sensors
        self.terrain = SimpleNamespace(env_origins=torch.zeros(num_envs, 3))

    def __getitem__(self, name):
        assert name == "robot"
        return self.asset


def _fake_env(num_envs=1):
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_envs, 1)
    asset = SimpleNamespace(
        data=SimpleNamespace(
            root_link_ang_vel_b=torch.zeros(num_envs, 3),
            root_link_lin_vel_b=torch.zeros(num_envs, 3),
            root_link_lin_vel_w=torch.zeros(num_envs, 3),
            root_link_pos_w=torch.tensor([[0.0, 0.0, 0.115]]).repeat(num_envs, 1),
            root_link_quat_w=quat,
            joint_pos=torch.zeros(num_envs, 14),
            default_joint_pos=torch.zeros(num_envs, 14),
        )
    )
    asset.find_joints = lambda _pattern: (list(range(14)), [])
    sensors = {
        "robot_ground_contact": SimpleNamespace(
            data=SimpleNamespace(found=torch.ones(num_envs, 1))
        ),
        "trunk_ground_contact": SimpleNamespace(
            data=SimpleNamespace(found=torch.zeros(num_envs, 1))
        ),
        "head_ground_contact": SimpleNamespace(
            data=SimpleNamespace(found=torch.zeros(num_envs, 1))
        ),
        "left_foot_ground_contact": SimpleNamespace(
            data=SimpleNamespace(found=torch.ones(num_envs, 1))
        ),
        "right_foot_ground_contact": SimpleNamespace(
            data=SimpleNamespace(found=torch.ones(num_envs, 1))
        ),
    }
    env = SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        step_dt=0.02,
        common_step_counter=0,
        scene=_Scene(asset, sensors, num_envs),
    )
    mdp._grounded_backroll_state(env)
    env._roulade_roll_direction[:] = -1.0
    env._backroll_recovery_enabled[:] = True
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
        "backroll_launch_tuck_progress",
        "backroll_head_pivot",
        "backroll_completion_progress",
        "backroll_speed_progress",
        "backroll_feet_recontact",
        "backroll_landing_readiness",
        "backroll_success",
        "backroll_invalid",
        "backroll_overspeed",
        "backroll_sagittal",
        "backroll_lateral_velocity",
        "backroll_flatness",
        "action_rate_l2",
        "gentle_landing",
        "self_collisions",
        "backroll_contact_sequence",
    }
    assert (
        cfg.rewards["backroll_contact_sequence"].func
        is mdp.grounded_backroll_contact_sequence
    )
    assert cfg.rewards["backroll_contact_sequence"].weight == pytest.approx(0.5)
    assert cfg.rewards["backroll_contact_sequence"].params == {
        "trunk_value": 1.0,
        "head_value": 2.0,
    }
    assert cfg.rewards["backroll_progress"].weight == pytest.approx(4.0)
    assert (
        cfg.rewards["backroll_launch_tuck_progress"].func
        is mdp.grounded_backroll_launch_tuck_progress
    )
    assert cfg.rewards["backroll_launch_tuck_progress"].weight == pytest.approx(2.0)
    assert cfg.rewards["backroll_completion_progress"].weight == pytest.approx(12.0)
    assert cfg.rewards["backroll_completion_progress"].params["start_angle"] == pytest.approx(
        math.radians(110.0)
    )
    assert cfg.rewards["backroll_speed_progress"].weight == pytest.approx(3.0)
    assert cfg.rewards["backroll_speed_progress"].params == {
        "minimum_rate": 2.0,
        "target_rate": 4.5,
        "target_angle": 2.0 * math.pi,
    }
    assert cfg.rewards["backroll_feet_recontact"].weight == pytest.approx(5.0)
    assert (
        cfg.rewards["backroll_feet_recontact"].func
        is mdp.grounded_backroll_feet_recontact_rate
    )
    assert cfg.rewards["backroll_landing_readiness"].weight == pytest.approx(8.0)
    assert (
        cfg.rewards["backroll_landing_readiness"].func
        is mdp.grounded_backroll_landing_readiness_progress
    )
    assert cfg.rewards["backroll_success"].weight == pytest.approx(20.0)
    assert cfg.rewards["backroll_invalid"].weight == pytest.approx(-5.0)
    assert cfg.rewards["backroll_sagittal"].weight == pytest.approx(-0.10)
    assert cfg.rewards["backroll_flatness"].weight == pytest.approx(-1.0)
    assert cfg.rewards["backroll_sagittal"].func is mdp.grounded_backroll_sagittal_penalty
    assert cfg.rewards["backroll_flatness"].func is mdp.grounded_backroll_flatness_penalty
    assert cfg.rewards["action_rate_l2"].weight == pytest.approx(0.0)
    assert cfg.curriculum["backroll_phase"].params[
        "action_rate_reward_weights"
    ] == [0.0, -0.025, -0.05, -0.075, -0.10]
    assert cfg.curriculum["backroll_phase"].params[
        "invalid_reward_weights"
    ] == [-5.0, -6.0, -8.0, -10.0, -10.0]
    assert cfg.curriculum["backroll_sagittal_weight"].func is mdp.reward_weight
    assert cfg.curriculum["backroll_sagittal_weight"].params["weight_stages"] == [
        {"step": 0, "weight": -0.10},
        {"step": 600 * 24, "weight": -0.25},
    ]
    assert "backroll_upright_weight" not in cfg.curriculum
    assert "backroll_height_weight" not in cfg.curriculum
    forbidden = ("sprint", "distance", "lane", "road", "recovery", "reposition")
    assert not any(token in name for name in cfg.rewards for token in forbidden)
    assert cfg.events["set_grounded_backroll_state"].func is mdp.reset_grounded_backroll_state
    assert cfg.curriculum["backroll_phase"].func is mdp.grounded_backroll_curriculum
    assert MicroduckBackrollRlCfg.experiment_name == "microduck_backroll"
    assert MicroduckBackrollRlCfg.max_iterations == 4000
    assert MicroduckBackrollRlCfg.save_interval == 50
    assert MicroduckBackrollRlCfg.algorithm.learning_rate == pytest.approx(2.5e-5)
    assert MicroduckBackrollRlCfg.algorithm.schedule == "fixed"
    assert MicroduckBackrollRlCfg.algorithm.entropy_coef == pytest.approx(5.0e-4)
    assert MicroduckBackrollRlCfg.algorithm.anchor_retention == pytest.approx(0.50)
    assert (
        MicroduckBackrollRlCfg.algorithm.class_name
        == "mjlab_microduck.tasks.backroll_ppo.AnchoredPPO"
    )
    assert MicroduckBackrollRlCfg.actor.distribution_cfg["init_std"] == 1.0
    assert load_runner_cls("Mjlab-Backroll-Flat-MicroDuck") is MicroduckFrozenActorNormRunner


def test_backroll_runner_restores_configured_lr_after_optimizer_resume(monkeypatch):
    configured_lr = 2.5e-5
    runner = object.__new__(MicroduckFrozenActorNormRunner)
    runner._configured_learning_rate = configured_lr
    runner.alg = SimpleNamespace(
        learning_rate=7.59375e-4,
        optimizer=SimpleNamespace(param_groups=[{"lr": 7.59375e-4}]),
    )

    monkeypatch.setattr(
        "mjlab_microduck.tasks.MicroduckOnPolicyRunner.load",
        lambda self, path, load_cfg, strict, map_location: {"loaded": path},
    )

    infos = runner.load("parent.pt")

    assert infos == {"loaded": "parent.pt"}
    assert runner.alg.learning_rate == pytest.approx(configured_lr)
    assert runner.alg.optimizer.param_groups[0]["lr"] == pytest.approx(configured_lr)


def test_backroll_actor_anchor_preserves_only_a_bounded_residual():
    algorithm = object.__new__(AnchoredPPO)
    algorithm.device = "cpu"
    algorithm.anchor_retention = 0.90
    algorithm.actor = nn.Sequential()
    algorithm.actor.mlp = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        algorithm.actor.mlp.weight.fill_(2.0)
    algorithm._capture_actor_anchor()

    with torch.no_grad():
        algorithm.actor.mlp.weight.add_(1.0)
    relative_l2 = algorithm._apply_actor_anchor()

    assert torch.allclose(
        algorithm.actor.mlp.weight,
        torch.full_like(algorithm.actor.mlp.weight, 2.90),
    )
    assert relative_l2 == pytest.approx(0.45)


def test_backroll_play_is_deterministic_standing_start():
    cfg = make_microduck_backroll_env_cfg(play=True)
    reset = cfg.events["set_grounded_backroll_state"].params

    assert reset["standing_prob"] == 1.0
    assert reset["midroll_prob"] == 0.0
    assert reset["yaw_range"] == (0.0, 0.0)
    assert reset["joint_noise_std"] == 0.0
    assert reset["sagittal_tolerance_deg"] == pytest.approx(20.0)
    assert "backroll_phase" not in cfg.curriculum


def test_backroll_launch_tuck_reverses_only_hip_pitch_from_forward_roll():
    """Backroll launch must seek the measured back-entry, not forward tuck."""
    cfg = make_microduck_backroll_env_cfg()

    assert cfg.events["set_grounded_backroll_state"].params["tuck_overrides"] == (
        BACKROLL_TUCK_OVERRIDES
    )
    assert BACKROLL_TUCK_OVERRIDES[2] == pytest.approx(-TUCK_OVERRIDES[2])
    assert BACKROLL_TUCK_OVERRIDES[11] == pytest.approx(-TUCK_OVERRIDES[11])
    for joint_id in (3, 4, 5, 6, 12, 13):
        assert BACKROLL_TUCK_OVERRIDES[joint_id] == pytest.approx(
            TUCK_OVERRIDES[joint_id]
        )


def test_launch_tuck_bootstrap_is_one_shot_and_closes_when_rolling(monkeypatch):
    env, asset = _fake_env()
    overrides = {2: -1.0, 5: 1.0}
    env._backroll_start_is_standing[:] = True
    monkeypatch.setattr(
        mdp,
        "_update_grounded_backroll_state",
        lambda _env, _asset: None,
    )

    # HOME earns nothing. Moving physically toward the configured tuck earns
    # one bounded frontier increment; holding it cannot become an annuity.
    assert (
        mdp.grounded_backroll_launch_tuck_progress(
            env,
            target_overrides=overrides,
        ).item()
        == pytest.approx(0.0)
    )
    env.common_step_counter += 1
    asset.data.joint_pos[:, 2] = -1.0
    asset.data.joint_pos[:, 5] = 1.0
    first_tuck = mdp.grounded_backroll_launch_tuck_progress(
        env,
        target_overrides=overrides,
    )
    assert first_tuck.item() > 0.0
    env.common_step_counter += 1
    assert (
        mdp.grounded_backroll_launch_tuck_progress(
            env,
            target_overrides=overrides,
        ).item()
        == pytest.approx(0.0)
    )

    # The bootstrap cannot follow a roll into the floor or compensate for the
    # real supported signed-rotation/contact requirements.
    env._roulade_accum[:] = math.radians(50.0)
    env._roulade_max[:] = math.radians(50.0)
    env._backroll_launch_tuck_paid.zero_()
    env.common_step_counter += 1
    assert (
        mdp.grounded_backroll_launch_tuck_progress(
            env,
            target_overrides=overrides,
        ).item()
        == pytest.approx(0.0)
    )


def test_sagittal_penalty_prices_small_persistent_twist_but_not_pitch(monkeypatch):
    env, asset = _fake_env()
    monkeypatch.setattr(
        mdp,
        "_update_grounded_backroll_state",
        lambda _env, _asset: None,
    )
    # The alignment ramp is fully open after a real roll has begun. Pure
    # backwards body-y rotation is the desired maneuver and remains free.
    env._roulade_max[:] = math.radians(90.0)
    asset.data.root_link_ang_vel_b[:] = torch.tensor([[0.0, -4.0, 0.0]])
    assert mdp.grounded_backroll_sagittal_penalty(env).item() == pytest.approx(0.0)

    # A 0.25 rad/s transverse twist used to sit below the old 1 rad/s dead
    # zone, yet it can accumulate past the 90-degree physical gate. It must
    # now receive a continuous signal, bounded independently of body-y speed.
    asset.data.root_link_ang_vel_b[:] = torch.tensor([[0.25, -4.0, 0.0]])
    # Entry and tuck exploration remain free until trunk contact commits this
    # sample to an actual grounded backroll.
    assert mdp.grounded_backroll_sagittal_penalty(env).item() == pytest.approx(0.0)
    env._backroll_trunk_latch[:] = True
    assert mdp.grounded_backroll_sagittal_penalty(env).item() == pytest.approx(0.25)

    asset.data.root_link_ang_vel_b[:] = torch.tensor([[5.0, -4.0, 0.0]])
    assert mdp.grounded_backroll_sagittal_penalty(env).item() == pytest.approx(3.0)


def test_repeated_backroll_rearms_without_adding_course_objectives():
    cfg = make_microduck_repeated_backroll_env_cfg()
    play_cfg = make_microduck_repeated_backroll_env_cfg(play=True)

    assert cfg.episode_length_s == REPEATED_EPISODE_LENGTH_S == 12.0
    assert "backroll_success" not in cfg.terminations
    # Training starts with the audited one-shot champion bridge; repeated
    # rearm/recovery is enabled only after standing mastery advances stage 0.
    assert cfg.events["set_grounded_backroll_state"].params["repeat_mode"] is False
    assert cfg.rewards["backroll_speed_progress"].func is mdp.grounded_backroll_speed_progress
    assert cfg.rewards["backroll_rise_velocity"].func is mdp.grounded_backroll_rise_velocity
    assert (
        cfg.rewards["backroll_contact_sequence"].func
        is mdp.grounded_backroll_contact_sequence
    )
    assert cfg.rewards["backroll_contact_sequence"].weight == pytest.approx(2.0)
    assert (
        cfg.rewards["backroll_non_top_head_dwell"].func
        is mdp.grounded_backroll_non_top_head_dwell_penalty
    )
    assert cfg.rewards["backroll_non_top_head_dwell"].weight == pytest.approx(0.5)
    assert cfg.rewards["backroll_non_top_head_dwell"].params == {"grace_steps": 9}
    assert (
        cfg.rewards["backroll_head_alignment_progress"].func
        is mdp.grounded_backroll_head_alignment_progress
    )
    assert cfg.rewards["backroll_head_alignment_progress"].weight == pytest.approx(1.5)
    assert cfg.rewards["backroll_success"].func is mdp.grounded_backroll_repeat_success_rate
    assert cfg.metrics["backroll_cycle_count"].func is mdp.grounded_backroll_cycle_count
    assert cfg.events["set_grounded_backroll_state"].params[
        "ground_recovery_prob"
    ] == pytest.approx(0.0)
    assert cfg.events["set_grounded_backroll_state"].params[
        "ground_z_range"
    ] == (0.04, 0.05)
    for group in ("actor", "critic"):
        assert (
            cfg.observations[group].terms["command"].func
            is mdp.grounded_backroll_recovery_command
        )
    assert cfg.rewards["backroll_recovery_progress"].weight == pytest.approx(0.25)
    assert cfg.rewards["backroll_recovery_success"].weight == pytest.approx(5.0)
    assert (
        cfg.metrics["backroll_recovery_success_rate"].func
        is mdp.grounded_backroll_recovery_success_fraction
    )
    forbidden = ("sprint", "distance", "lane", "road", "reposition")
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
    ] == [0.25, 0.50, 0.50, 0.80, 0.90, 1.0]
    assert [
        stage["params"]["repeat_mode"]
        for stage in REPEATED_BACKROLL_CURRICULUM_STAGES
    ] == [False, False, True, True, True, True]
    assert [
        stage["params"]["relaxed_first_cycle"]
        for stage in REPEATED_BACKROLL_CURRICULUM_STAGES
    ] == [False, False, False, False, False, False]
    assert [
        stage["params"]["yaw_range"]
        for stage in REPEATED_BACKROLL_CURRICULUM_STAGES[:3]
    ] == [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    assert [
        stage["params"]["mastery_cycles"]
        for stage in REPEATED_BACKROLL_CURRICULUM_STAGES
    ] == [1, 1, 1, 1, 2, 3]
    assert REPEATED_BACKROLL_CURRICULUM_STAGES[0]["params"][
        "joint_noise_std"
    ] == pytest.approx(0.0)
    assert REPEATED_BACKROLL_CURRICULUM_STAGES[0]["params"][
        "synthesize_contact_latches"
    ] is True
    assert [
        stage["params"]["ground_recovery_prob"]
        for stage in REPEATED_BACKROLL_CURRICULUM_STAGES
    ] == [0.0, 0.0, 0.0, 0.05, 0.10, 0.20]
    assert [
        stage["params"]["recovery_enabled"]
        for stage in REPEATED_BACKROLL_CURRICULUM_STAGES
    ] == [False, False, True, True, True, True]
    assert [
        stage["params"]["reference_state_prob"]
        for stage in REPEATED_BACKROLL_CURRICULUM_STAGES
    ] == [1.0, 1.0, 1.0, 0.10, 0.05, 0.0]
    first_stage = REPEATED_BACKROLL_CURRICULUM_STAGES[0]["params"]
    assert first_stage["reference_phase_range_deg"] == (100.0, 180.0)
    assert first_stage["reference_source_seed"] is None
    assert first_stage["yaw_range"] == (0.0, 0.0)
    assert first_stage["midroll_pitch_min"] == pytest.approx(math.radians(100.0))
    assert first_stage["midroll_pitch_max"] == pytest.approx(math.radians(180.0))
    assert first_stage["midroll_omega_range"] == (1.0, 3.0)
    play_reset = play_cfg.events["set_grounded_backroll_state"].params
    assert play_reset["repeat_mode"] is True
    assert play_reset["relaxed_first_cycle"] is False
    assert play_reset["reference_state_prob"] == pytest.approx(0.0)
    assert play_reset["yaw_range"] == (0.0, 0.0)
    assert cfg.curriculum["backroll_phase"].params["success_threshold"] == pytest.approx(
        0.45
    )
    assert cfg.curriculum["backroll_phase"].params[
        "required_consecutive_windows"
    ] == 2
    assert cfg.curriculum["backroll_phase"].params["standing_only_mastery"] is True
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


def test_repeated_first_cycle_relaxation_cannot_open_the_sagittal_gate():
    env, _asset = _fake_env()
    env._backroll_repeat_mode[:] = True
    env._backroll_relaxed_first_cycle[:] = True
    env._backroll_cycle_max_lateral_axis_z[:] = 1.0
    env._backroll_cycle_offaxis_rotation[:] = 10.0

    assert mdp._grounded_backroll_first_cycle_relaxed(env).item() is True
    assert mdp._grounded_backroll_positive_reward_valid(env).item() is False

    env._backroll_cycle_count[:] = 1
    assert mdp._grounded_backroll_first_cycle_relaxed(env).item() is False
    assert mdp._grounded_backroll_positive_reward_valid(env).item() is False


def test_late_pose_potential_does_not_reward_a_side_basin():
    env, _asset = _fake_env()
    env._roulade_max[:] = math.radians(320.0)
    env._backroll_head_latch[:] = True
    env._backroll_cycle_max_lateral_axis_z[:] = 1.0
    env._backroll_cycle_offaxis_rotation[:] = math.radians(240.0)

    assert mdp._grounded_backroll_positive_reward_valid(env).item() is False
    assert mdp._grounded_backroll_potential_reward_valid(env).item() is False

    env._backroll_cycle_max_lateral_axis_z.zero_()
    env._backroll_cycle_offaxis_rotation.zero_()
    assert mdp._grounded_backroll_potential_reward_valid(env).item() is True

    env._roulade_max[:] = math.radians(120.0)
    assert mdp._grounded_backroll_potential_reward_valid(env).item() is False

    env._roulade_max[:] = math.radians(320.0)
    env._backroll_invalid[:] = True
    assert mdp._grounded_backroll_potential_reward_valid(env).item() is False


def test_landing_potential_starts_after_flat_head_pivot_not_before():
    env, _asset = _fake_env()
    env._backroll_head_latch[:] = True
    env._roulade_max[:] = math.radians(169.0)
    assert mdp._grounded_backroll_potential_reward_valid(env).item() is False

    env._roulade_max[:] = math.radians(170.0)
    assert mdp._grounded_backroll_potential_reward_valid(env).item() is True

    env._backroll_head_latch[:] = False
    assert mdp._grounded_backroll_potential_reward_valid(env).item() is False


def test_backroll_curriculum_matches_mastery_stages():
    assert [stage["params"]["standing_prob"] for stage in BACKROLL_CURRICULUM_STAGES] == [
        0.90,
        0.925,
        0.95,
        0.975,
        0.99,
    ]
    assert [stage["params"]["midroll_prob"] for stage in BACKROLL_CURRICULUM_STAGES] == [
        0.10,
        0.075,
        0.05,
        0.025,
        0.01,
    ]
    assert BACKROLL_CURRICULUM_STAGES[0]["params"]["midroll_pitch_min"] == pytest.approx(
        math.radians(90.0)
    )
    assert BACKROLL_CURRICULUM_STAGES[0]["params"]["midroll_pitch_max"] == pytest.approx(
        math.radians(300.0)
    )
    assert BACKROLL_CURRICULUM_STAGES[0]["params"]["midroll_omega_range"] == (
        1.5,
        4.0,
    )
    assert BACKROLL_CURRICULUM_STAGES[0]["params"]["joint_noise_std"] == pytest.approx(
        0.0
    )
    assert BACKROLL_CURRICULUM_STAGES[0]["params"]["reference_state_prob"] == pytest.approx(
        1.0
    )
    assert BACKROLL_CURRICULUM_STAGES[0]["params"]["reference_state_path"]
    assert BACKROLL_CURRICULUM_STAGES[0]["params"]["reference_phase_range_deg"] == (
        90.0,
        300.0,
    )
    assert BACKROLL_CURRICULUM_STAGES[0]["params"]["reference_phase_buckets_deg"] == (
        (90.0, 110.0),
        (130.0, 150.0),
        (170.0, 190.0),
        (210.0, 230.0),
        (250.0, 270.0),
        (280.0, 300.0),
    )
    assert all(
        stage["params"]["reference_phase_buckets_deg"]
        == BACKROLL_CURRICULUM_STAGES[0]["params"]["reference_phase_buckets_deg"]
        for stage in BACKROLL_CURRICULUM_STAGES
    )
    assert BACKROLL_CURRICULUM_STAGES[0]["params"]["reference_strict_sagittal"]
    assert not BACKROLL_CURRICULUM_STAGES[0]["params"]["synthesize_contact_latches"]
    assert [
        stage["params"]["sagittal_tolerance_deg"]
        for stage in BACKROLL_CURRICULUM_STAGES
    ] == [29.0, 27.0, 25.0, 22.0, 20.0]
    assert [stage["params"]["reference_state_prob"] for stage in BACKROLL_CURRICULUM_STAGES] == [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert BACKROLL_CURRICULUM_STAGES[0]["params"]["reference_source_seed"] is None
    assert BACKROLL_CURRICULUM_STAGES[0]["params"]["yaw_range"] == (0.0, 0.0)
    cfg = make_microduck_backroll_env_cfg()
    params = cfg.curriculum["backroll_phase"].params
    assert params["window_episodes"] == 4096
    assert params["success_threshold"] == pytest.approx(0.70)
    assert params["standing_only_mastery"] is True


def test_backroll_discovery_envelope_admits_bridge_but_rejects_side_roll():
    def quat_for_lateral_tilt(degrees: float) -> torch.Tensor:
        half = math.radians(degrees) / 2.0
        return torch.tensor([math.cos(half), math.sin(half), 0.0, 0.0])

    qpos = torch.zeros(3, 21)
    qpos[:, 3:7] = torch.stack(
        (
            quat_for_lateral_tilt(20.0),
            quat_for_lateral_tilt(28.5),
            quat_for_lateral_tilt(60.0),
        )
    )
    bank = {
        "qpos": qpos,
        "frontier": torch.full((3,), math.radians(220.0)),
        "trunk_latch": torch.ones(3, dtype=torch.bool),
        "head_latch": torch.ones(3, dtype=torch.bool),
    }

    final_mask = mdp._grounded_backroll_reference_consistent_rows(
        bank,
        max_lateral_axis_z=math.sin(math.radians(20.0)),
    )
    discovery_mask = mdp._grounded_backroll_reference_consistent_rows(
        bank,
        max_lateral_axis_z=math.sin(math.radians(29.0)),
    )

    assert final_mask.tolist() == [True, False, False]
    assert discovery_mask.tolist() == [True, True, False]


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


def test_repeated_phase_reset_can_require_physical_contact_latches(monkeypatch):
    env, _asset = _fake_env()
    env.sim = SimpleNamespace(
        data=SimpleNamespace(
            qpos=torch.zeros(1, 21),
            qvel=torch.zeros(1, 20),
        )
    )
    env.sim.data.qpos[:, 3] = 1.0
    monkeypatch.setattr(mdp, "_servo_joint_ids", lambda _env, _asset: list(range(14)))
    pitch = math.radians(170.0)

    mdp.reset_grounded_backroll_state(
        env,
        torch.tensor([0]),
        standing_prob=0.0,
        midroll_prob=1.0,
        midroll_pitch_min=pitch,
        midroll_pitch_max=pitch,
        midroll_omega_range=(0.5, 0.5),
        standing_tilt_max=0.0,
        yaw_range=(0.0, 0.0),
        joint_noise_std=0.0,
        synthesize_contact_latches=False,
    )

    assert env._roulade_max.item() == pytest.approx(pitch)
    assert not env._backroll_trunk_latch.item()
    assert not env._backroll_head_latch.item()
    assert not env._roulade_head_latch.item()


def test_repeated_ground_reset_starts_grounded_at_home_in_blocked_recovery(
    monkeypatch,
):
    env, asset = _fake_env()
    env.sim = SimpleNamespace(
        data=SimpleNamespace(qpos=torch.zeros(1, 21), qvel=torch.zeros(1, 20))
    )
    env.sim.data.qpos[:, 3] = 1.0
    env.sim.data.qpos[:, 7:] = 9.0
    env.sim.data.qvel[:, 6:] = 7.0
    asset.data.default_joint_pos[:] = torch.linspace(-0.3, 0.3, 14)
    monkeypatch.setattr(mdp, "_servo_joint_ids", lambda _env, _asset: list(range(14)))

    mdp.reset_grounded_backroll_state(
        env,
        torch.tensor([0]),
        standing_prob=1.0,
        midroll_prob=0.0,
        repeat_mode=True,
        ground_recovery_prob=1.0,
        ground_face_down_prob=1.0,
        ground_face_up_prob=0.0,
        ground_left_prob=0.0,
        ground_right_prob=0.0,
        ground_z_range=(0.045, 0.045),
        yaw_range=(0.0, 0.0),
        joint_noise_std=0.0,
    )

    assert env._backroll_recovery_active.item()
    assert env._backroll_recovery_used.item()
    assert env._backroll_invalid.item()
    assert env._backroll_recovery_attempt_count.item() == pytest.approx(1.0)
    assert env._roulade_max.item() == 0.0
    assert not env._backroll_trunk_latch.item()
    assert not env._backroll_head_latch.item()
    assert env.sim.data.qpos[0, 2].item() == pytest.approx(0.045)
    assert env.sim.data.qpos[0, 5].abs().item() > 0.5
    assert torch.allclose(env.sim.data.qpos[0, 7:], asset.data.default_joint_pos[0])
    assert torch.all(env.sim.data.qvel[0, 6:] == 0.0)


def test_repeated_reference_reset_uses_physical_state_and_actual_latches(
    monkeypatch,
    tmp_path,
):
    env, _asset = _fake_env()
    env.sim = SimpleNamespace(
        data=SimpleNamespace(qpos=torch.zeros(1, 21), qvel=torch.zeros(1, 20))
    )
    env.sim.data.qpos[:, 3] = 1.0
    monkeypatch.setattr(mdp, "_servo_joint_ids", lambda _env, _asset: list(range(14)))
    qpos = torch.linspace(-0.2, 0.2, 21)
    qpos[:7] = torch.tensor([0.2, -0.1, 0.07, 1.0, 0.0, 0.0, 0.0])
    qvel = torch.linspace(-1.0, 1.0, 20)
    reference_path = tmp_path / "reference.pt"
    torch.save(
        {
            "rows": [
                {
                    "qpos": qpos,
                    "qvel": qvel,
                    "seed": 10,
                    "accum": torch.tensor(math.radians(180.0)),
                    "frontier": torch.tensor(math.radians(190.0)),
                    "phase_center_deg": 180.0,
                    "paid": torch.tensor(math.radians(175.0)),
                    "trunk_latch": torch.tensor(True),
                    "head_latch": torch.tensor(False),
                },
                {
                    "qpos": qpos + 10.0,
                    "qvel": qvel + 10.0,
                    "seed": 10,
                    "accum": torch.tensor(math.radians(260.0)),
                    "frontier": torch.tensor(math.radians(265.0)),
                    "paid": torch.tensor(math.radians(265.0)),
                    "phase_center_deg": 260.0,
                    "trunk_latch": torch.tensor(True),
                    "head_latch": torch.tensor(True),
                },
                {
                    "qpos": qpos + 20.0,
                    "qvel": qvel + 20.0,
                    "seed": 9,
                    "accum": torch.tensor(math.radians(180.0)),
                    "frontier": torch.tensor(math.radians(190.0)),
                    "paid": torch.tensor(math.radians(190.0)),
                    "phase_center_deg": 180.0,
                    "trunk_latch": torch.tensor(True),
                    "head_latch": torch.tensor(True),
                }
            ]
        },
        reference_path,
    )

    mdp.reset_grounded_backroll_state(
        env,
        torch.tensor([0]),
        standing_prob=0.0,
        midroll_prob=1.0,
        midroll_pitch_min=math.radians(260.0),
        midroll_pitch_max=math.radians(260.0),
        repeat_mode=True,
        reference_state_prob=1.0,
        reference_state_path=str(reference_path),
        reference_phase_range_deg=(180.0, 180.0),
        reference_source_seed=10,
        yaw_range=(0.0, 0.0),
        joint_noise_std=0.0,
    )

    assert torch.allclose(env.sim.data.qpos[0], qpos)
    assert torch.allclose(env.sim.data.qvel[0], qvel)
    assert env._roulade_accum.item() == pytest.approx(math.radians(180.0))
    assert env._roulade_max.item() == pytest.approx(math.radians(190.0))
    assert env._roulade_paid.item() == pytest.approx(math.radians(190.0))
    assert env._backroll_trunk_latch.item()
    assert not env._backroll_head_latch.item()
    assert not env._backroll_start_is_standing.item()


def test_strict_reference_reset_excludes_side_biased_rows_and_balances_buckets(
    monkeypatch,
    tmp_path,
):
    env, _asset = _fake_env(24)
    env.sim = SimpleNamespace(
        data=SimpleNamespace(qpos=torch.zeros(24, 21), qvel=torch.zeros(24, 20))
    )
    env.sim.data.qpos[:, 3] = 1.0
    monkeypatch.setattr(mdp, "_servo_joint_ids", lambda _env, _asset: list(range(14)))
    qvel = torch.zeros(20)
    good_100 = torch.zeros(21)
    good_100[3] = 1.0
    good_140 = good_100.clone()
    good_260 = good_100.clone()
    side_260 = good_100.clone()
    side_260[3] = math.sqrt(3.0) / 2.0
    side_260[4] = 0.5
    rows = []
    for phase, qpos, trunk, head in (
        (100.0, good_100, False, False),
        (140.0, good_140, False, False),
        (260.0, good_260, True, True),
        (260.0, side_260, True, True),
    ):
        rows.append(
            {
                "qpos": qpos,
                "qvel": qvel,
                "seed": 1,
                "accum": torch.tensor(math.radians(phase)),
                "frontier": torch.tensor(math.radians(phase)),
                "phase_center_deg": phase,
                "trunk_latch": torch.tensor(trunk),
                "head_latch": torch.tensor(head),
            }
        )
    reference_path = tmp_path / "strict-reference.pt"
    torch.save({"rows": rows}, reference_path)

    mdp.reset_grounded_backroll_state(
        env,
        torch.arange(24),
        standing_prob=0.0,
        midroll_prob=1.0,
        reference_state_prob=1.0,
        reference_state_path=str(reference_path),
        reference_phase_range_deg=(90.0, 270.0),
        reference_phase_buckets_deg=((90.0, 110.0), (130.0, 150.0), (250.0, 270.0)),
        reference_strict_sagittal=True,
        yaw_range=(0.0, 0.0),
        joint_noise_std=0.0,
    )

    angles = torch.round(torch.rad2deg(env._roulade_max)).to(torch.long)
    assert set(angles.tolist()) == {100, 140, 260}
    assert (angles == 260).any()
    selected = env.sim.data.qpos[angles == 260, 3:7]
    assert torch.allclose(selected[:, 3], torch.zeros_like(selected[:, 3]))


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


def test_repeated_curriculum_requires_two_consecutive_clean_mastery_windows():
    env, _asset = _fake_env()
    event_cfg = SimpleNamespace(params={})
    env.event_manager = SimpleNamespace(
        get_term_cfg=lambda _name: event_cfg,
    )
    stages = [
        {"params": {"mastery_cycles": 1}},
        {"params": {"mastery_cycles": 2}},
    ]

    for expected_stage in (0, 1):
        env._backroll_window_episodes.fill_(4096)
        env._backroll_window_successes.fill_(4096)
        env._backroll_window_bad_states.zero_()
        actual = mdp.grounded_backroll_curriculum(
            env,
            torch.tensor([0]),
            event_name="set_grounded_backroll_state",
            stages=stages,
            required_consecutive_windows=2,
        )
        assert actual.item() == expected_stage
        env.common_step_counter += 50

    assert env._backroll_consecutive_mastery_windows == 0
    assert event_cfg.params["mastery_cycles"] == 2


def test_repeated_curriculum_cannot_advance_from_phase_successes_alone():
    env, _asset = _fake_env()
    event_cfg = SimpleNamespace(params={})
    env.event_manager = SimpleNamespace(
        get_term_cfg=lambda _name: event_cfg,
    )
    stages = [
        {"params": {"mastery_cycles": 1}},
        {"params": {"mastery_cycles": 2}},
    ]
    env._backroll_window_episodes.fill_(8192)
    env._backroll_window_successes.fill_(8192)
    env._backroll_window_standing_episodes.fill_(4096)
    env._backroll_window_standing_successes.zero_()

    actual = mdp.grounded_backroll_curriculum(
        env,
        torch.tensor([0]),
        event_name="set_grounded_backroll_state",
        stages=stages,
        standing_only_mastery=True,
    )

    assert actual.item() == 0.0
    assert env._backroll_curriculum_stage == 0


def test_backroll_curriculum_enables_action_smoothness_only_after_mastery():
    env, _asset = _fake_env()
    event_cfg = SimpleNamespace(params={})
    action_rate_cfg = SimpleNamespace(weight=-1.0)
    env.event_manager = SimpleNamespace(get_term_cfg=lambda _name: event_cfg)
    env.reward_manager = SimpleNamespace(
        get_term_cfg=lambda name: action_rate_cfg if name == "action_rate_l2" else None
    )
    stages = [{"params": {}}, {"params": {}}]

    mdp.grounded_backroll_curriculum(
        env,
        torch.tensor([0]),
        event_name="set_grounded_backroll_state",
        stages=stages,
        action_rate_reward_name="action_rate_l2",
        action_rate_reward_weights=[0.0, -0.025],
    )
    assert action_rate_cfg.weight == pytest.approx(0.0)

    env._backroll_window_episodes.fill_(4096)
    env._backroll_window_successes.fill_(4096)
    env.common_step_counter += 50
    mdp.grounded_backroll_curriculum(
        env,
        torch.tensor([0]),
        event_name="set_grounded_backroll_state",
        stages=stages,
        action_rate_reward_name="action_rate_l2",
        action_rate_reward_weights=[0.0, -0.025],
    )
    assert env._backroll_curriculum_stage == 1
    assert action_rate_cfg.weight == pytest.approx(-0.025)


def test_repeated_strict_roll_stage_terminates_failure_before_recovery(monkeypatch):
    env, asset = _fake_env()
    env._backroll_repeat_mode[:] = True
    env._backroll_recovery_enabled[:] = False
    env._roulade_accum[:] = math.radians(90.0)
    env._roulade_max[:] = math.radians(90.0)
    env._backroll_previous_frontier[:] = math.radians(90.0)
    env._backroll_cycle_offaxis_rotation[:] = (
        mdp._BACKROLL_REPEAT_MAX_OFFAXIS_ROTATION + math.radians(1.0)
    )
    asset.data.root_link_ang_vel_b.zero_()
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )

    mdp.grounded_backroll_progress(env)

    assert env._backroll_invalid.item()
    assert not env._backroll_recovery_active.item()
    assert env._backroll_recovery_attempt_count.item() == 0.0
    assert mdp.grounded_backroll_invalid_termination(env).item()


def test_stationary_head_wobble_cannot_accumulate_a_side_roll_rejection(monkeypatch):
    """Cross-axis jitter only counts while the backroll still advances."""
    env, asset = _fake_env()
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )
    env._roulade_accum[:] = math.radians(120.0)
    env._roulade_max[:] = math.radians(120.0)
    env._roulade_paid[:] = math.radians(120.0)
    env._backroll_previous_frontier[:] = math.radians(120.0)
    # This is an otherwise-sagittal stationary head hold with controller
    # jitter, not an active side roll. The former A222 accounting integrated
    # this x-axis noise for five seconds and tripped the 90-degree escape.
    asset.data.root_link_ang_vel_b[:] = torch.tensor([[1.0, 0.0, 0.0]])

    for step in range(100):
        env.common_step_counter = step
        mdp.grounded_backroll_progress(env)

    assert env._backroll_cycle_offaxis_rotation.item() == pytest.approx(0.0)
    assert not env._backroll_invalid.item()


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


def test_head_alignment_progress_is_active_only_and_signed(monkeypatch):
    env, asset = _fake_env()
    env._backroll_repeat_mode[:] = True
    # Head-top alignment is a transition guide for the measured head-only
    # pivot window; it must not be gated out when the trunk sensor never
    # latches, while the physical completion gate remains unchanged.
    env._backroll_trunk_latch[:] = False
    env._roulade_accum[:] = math.radians(180.0)
    env._roulade_max[:] = math.radians(180.0)
    env._backroll_previous_frontier[:] = math.radians(180.0)
    env._backroll_cycle_max_lateral_axis_z.zero_()
    env._backroll_cycle_offaxis_rotation.zero_()
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )
    alignment = torch.tensor([0.2])
    monkeypatch.setattr(
        mdp, "_head_top_alignment", lambda _env, _asset: alignment.clone()
    )
    env.scene.sensors["head_ground_contact"].data.found[:] = 1.0
    asset.data.root_link_ang_vel_b[:, 1] = -1.0

    first = mdp.grounded_backroll_head_alignment_progress(env)
    env.common_step_counter += 1
    alignment[:] = 0.4
    second = mdp.grounded_backroll_head_alignment_progress(env)
    env.common_step_counter += 1
    alignment[:] = 0.1
    backward = mdp.grounded_backroll_head_alignment_progress(env)
    env.common_step_counter += 1
    alignment[:] = 0.8
    asset.data.root_link_ang_vel_b[:, 1] = 0.0
    static = mdp.grounded_backroll_head_alignment_progress(env)

    assert first.item() == 0.0
    assert second.item() == pytest.approx(0.03)
    assert backward.item() == pytest.approx(-0.03)
    assert static.item() == 0.0


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


def test_repeated_non_top_head_dwell_allows_tuck_grace_then_penalizes(monkeypatch):
    env, _asset = _fake_env()
    env._backroll_repeat_mode[:] = True
    env._roulade_accum[:] = math.radians(177.0)
    env._roulade_max[:] = math.radians(177.0)
    env.scene.sensors["left_foot_ground_contact"].data.found[:] = 0.0
    env.scene.sensors["right_foot_ground_contact"].data.found[:] = 0.0
    env.scene.sensors["head_ground_contact"].data.found[:] = 1.0
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.zeros(1, dtype=torch.bool),
    )

    values = []
    for step in range(10):
        env.common_step_counter = step
        values.append(
            mdp.grounded_backroll_non_top_head_dwell_penalty(env).item()
        )

    assert values[:9] == [0.0] * 9
    assert values[9] == -1.0
    assert mdp.grounded_backroll_max_non_top_head_dwell_s(env).item() == pytest.approx(
        0.20
    )

    # Breaking the bad contact resets the consecutive timer, so separate
    # touches cannot accumulate into a penalty.
    env.scene.sensors["head_ground_contact"].data.found[:] = 0.0
    env.common_step_counter = 10
    assert mdp.grounded_backroll_non_top_head_dwell_penalty(env).item() == 0.0
    assert env._backroll_non_top_head_steps.item() == 0

    env.scene.sensors["head_ground_contact"].data.found[:] = 1.0
    env.common_step_counter = 11
    assert mdp.grounded_backroll_non_top_head_dwell_penalty(env).item() == 0.0


def test_repeated_non_top_head_dwell_ignores_early_tuck_contact(monkeypatch):
    """Early tuck contact is not confused with the measured head pivot window."""
    env, _asset = _fake_env()
    env._backroll_repeat_mode[:] = True
    env._roulade_accum[:] = math.radians(45.0)
    env._roulade_max[:] = math.radians(45.0)
    env.scene.sensors["left_foot_ground_contact"].data.found[:] = 0.0
    env.scene.sensors["right_foot_ground_contact"].data.found[:] = 0.0
    env.scene.sensors["head_ground_contact"].data.found[:] = 1.0
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.zeros(1, dtype=torch.bool),
    )

    for step in range(20):
        env.common_step_counter = step
        assert mdp.grounded_backroll_non_top_head_dwell_penalty(env).item() == 0.0


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
    assert env._backroll_recovery_active.item()
    assert env._backroll_invalid.item()
    assert not mdp.grounded_backroll_invalid_termination(env).item()
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


def test_alignment_penalties_start_after_entry_and_are_full_by_head_window():
    env, asset = _fake_env()
    env._backroll_trunk_latch[:] = True
    asset.data.root_link_ang_vel_b[:] = torch.tensor([[1.0, -6.0, 2.0]])
    half = math.radians(45.0) * 0.5
    asset.data.root_link_quat_w[:] = torch.tensor(
        [[math.cos(half), math.sin(half), 0.0, 0.0]]
    )
    frontiers = [0.0, math.radians(10.0), math.radians(45.0), math.radians(90.0)]
    sagittal = []
    flatness = []
    for frontier in frontiers:
        env._roulade_max[:] = frontier
        sagittal.append(mdp.grounded_backroll_sagittal_penalty(env).item())
        flatness.append(mdp.grounded_backroll_flatness_penalty(env).item())

    assert sagittal[0] == pytest.approx(0.0)
    assert flatness[0] == pytest.approx(0.0)
    assert sagittal[1] == pytest.approx(0.0)
    assert flatness[1] == pytest.approx(0.0)
    assert 0.0 < sagittal[2] < sagittal[3]
    assert 0.0 < flatness[2] < flatness[3]


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


def test_one_shot_side_escape_is_terminal_and_cannot_collect_skill_rewards(monkeypatch):
    env, asset = _fake_env()
    monkeypatch.setattr(
        mdp,
        "_lateral_axis_z",
        lambda _quat: torch.full((1,), math.sin(math.radians(31.0))),
    )
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )
    env._roulade_accum[:] = math.radians(31.0)
    env._roulade_max[:] = math.radians(31.0)
    env._roulade_paid[:] = math.radians(31.0)
    env._backroll_previous_frontier[:] = math.radians(31.0)
    asset.data.root_link_ang_vel_b[:, 1] = -2.0

    assert mdp.grounded_backroll_progress(env).item() == 0.0
    assert env._backroll_invalid.item() is True
    assert mdp.grounded_backroll_head_pivot(env).item() == 0.0
    assert mdp.grounded_backroll_contact_sequence(env).item() == 0.0


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


def test_feet_recontact_bridge_requires_head_release_and_pays_once(monkeypatch):
    env, _asset = _fake_env()
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )
    env._roulade_accum[:] = math.radians(305.0)
    env._roulade_max[:] = math.radians(305.0)
    env._roulade_paid[:] = math.radians(305.0)
    env._backroll_previous_frontier[:] = math.radians(305.0)
    env._backroll_trunk_latch[:] = True
    env._backroll_head_latch[:] = True

    # A head/feet tripod is not a landing bridge.
    env.scene.sensors["head_ground_contact"].data.found[:] = 1.0
    assert mdp.grounded_backroll_feet_recontact_rate(env).item() == 0.0
    env.common_step_counter += 1

    # Releasing the head while upright enough and supported on both feet emits
    # exactly one pulse. Holding the same posture cannot farm it.
    env.scene.sensors["head_ground_contact"].data.found[:] = 0.0
    first = mdp.grounded_backroll_feet_recontact_rate(env)
    env.common_step_counter += 1
    second = mdp.grounded_backroll_feet_recontact_rate(env)
    assert first.item() == pytest.approx(1.0 / env.step_dt)
    assert second.item() == 0.0


def test_landing_readiness_pays_only_new_post_recontact_progress(monkeypatch):
    env, _asset = _fake_env()
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )
    env._roulade_accum[:] = math.radians(305.0)
    env._roulade_max[:] = math.radians(305.0)
    env._roulade_paid[:] = math.radians(305.0)
    env._backroll_previous_frontier[:] = math.radians(305.0)
    env._backroll_trunk_latch[:] = True
    env._backroll_head_latch[:] = True

    # The physical feet-recontact step is snapshotted, not double-paid.
    assert mdp.grounded_backroll_landing_readiness_progress(env).item() == 0.0
    assert env._backroll_feet_recontact_latch.item()
    initial = env._backroll_landing_readiness_paid.item()
    assert initial > 0.0

    env.common_step_counter += 1
    assert mdp.grounded_backroll_landing_readiness_progress(env).item() == 0.0

    # More valid backward phase while remaining upright, slow, and on both
    # feet extends the max-so-far readiness frontier exactly once.
    env._roulade_max[:] = math.radians(340.0)
    env.common_step_counter += 1
    assert mdp.grounded_backroll_landing_readiness_progress(env).item() > 0.0
    reached = env._backroll_landing_readiness_paid.item()
    assert reached > initial

    env.common_step_counter += 1
    assert mdp.grounded_backroll_landing_readiness_progress(env).item() == 0.0
    assert env._backroll_landing_readiness_paid.item() == pytest.approx(reached)

    # A head contact cannot extend readiness. Releasing it later may pay only
    # the genuinely new portion, never the missed interval twice.
    env.scene.sensors["head_ground_contact"].data.found[:] = 1.0
    env._roulade_max[:] = math.radians(345.0)
    env.common_step_counter += 1
    assert mdp.grounded_backroll_landing_readiness_progress(env).item() == 0.0
    env.scene.sensors["head_ground_contact"].data.found[:] = 0.0
    env.common_step_counter += 1
    assert mdp.grounded_backroll_landing_readiness_progress(env).item() > 0.0


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


def test_one_shot_side_landing_cannot_count_as_curriculum_success(monkeypatch):
    env, _asset = _fake_env()
    env._backroll_repeat_mode[:] = False
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

    hold_steps = math.ceil(mdp._BACKROLL_LANDING_HOLD_SECONDS / env.step_dt)
    for _ in range(hold_steps + 2):
        assert mdp.grounded_backroll_success_rate(env).item() == 0.0
        env.common_step_counter += 1

    assert not env._backroll_success.item()


def test_repeated_backroll_cannot_park_on_trunk_mid_cycle(monkeypatch):
    env, asset = _fake_env()
    env._backroll_repeat_mode[:] = True
    env.scene.sensors["left_foot_ground_contact"].data.found[:] = 0.0
    env.scene.sensors["right_foot_ground_contact"].data.found[:] = 0.0
    env.scene.sensors["trunk_ground_contact"].data.found[:] = 1.0
    env._roulade_accum[:] = math.radians(180.0)
    env._roulade_max[:] = math.radians(180.0)
    env._backroll_previous_frontier[:] = math.radians(180.0)
    s = 2.0**-0.5
    asset.data.root_link_quat_w[:] = torch.tensor([[s, 0.0, s, 0.0]])
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

    assert env._backroll_recovery_active.item()
    assert env._backroll_invalid.item()
    assert not mdp.grounded_backroll_invalid_termination(env).item()


def test_repeated_backroll_does_not_treat_an_upright_launch_pause_as_a_fall(
    monkeypatch,
):
    env, asset = _fake_env()
    env._backroll_repeat_mode[:] = True
    env._roulade_accum[:] = math.radians(46.0)
    env._roulade_max[:] = math.radians(46.0)
    env._backroll_previous_frontier[:] = math.radians(46.0)
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )

    stall_steps = math.ceil(
        mdp._BACKROLL_REPEAT_PRE_EXIT_STALL_SECONDS / env.step_dt
    )
    for _ in range(stall_steps + 5):
        asset.data.root_link_ang_vel_b.zero_()
        mdp.grounded_backroll_progress(env)
        env.common_step_counter += 1

    assert not env._backroll_recovery_active.item()
    assert not env._backroll_invalid.item()


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
    assert env._backroll_recovery_active.item()
    assert env._backroll_invalid.item()
    assert not mdp.grounded_backroll_invalid_termination(env).item()


def test_repeated_recovery_rearms_once_then_credits_the_retry(monkeypatch):
    env, asset = _fake_env()
    env._backroll_repeat_mode[:] = True
    env._backroll_invalid[:] = True
    env._backroll_recovery_active[:] = True
    env._backroll_recovery_used[:] = True
    env._backroll_recovery_attempt_count[:] = 1.0
    env._roulade_accum[:] = math.radians(220.0)
    env._roulade_max[:] = math.radians(220.0)
    env._backroll_trunk_latch[:] = True
    env._backroll_head_latch[:] = True
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )

    recovery_hold_steps = math.ceil(
        mdp._BACKROLL_RECOVERY_HOLD_SECONDS / env.step_dt
    )
    recovery_pulses = []
    env.scene.sensors["left_foot_ground_contact"].data.found[:] = 0.0
    for _ in range(2):
        recovery_pulses.append(mdp.grounded_backroll_recovery_success(env).item())
        env.common_step_counter += 1
    assert env._backroll_recovery_hold_steps.item() == 0
    env.scene.sensors["left_foot_ground_contact"].data.found[:] = 1.0
    asset.data.root_link_lin_vel_w[:, 0] = (
        mdp._BACKROLL_RECOVERY_MAX_LIN_SPEED + 0.01
    )
    for _ in range(recovery_hold_steps + 1):
        recovery_pulses.append(mdp.grounded_backroll_recovery_success(env).item())
        env.common_step_counter += 1
    assert env._backroll_recovery_active.item()
    assert env._backroll_recovery_hold_steps.item() == 0
    asset.data.root_link_lin_vel_w.zero_()
    for _ in range(recovery_hold_steps + 1):
        recovery_pulses.append(mdp.grounded_backroll_recovery_success(env).item())
        env.common_step_counter += 1

    assert sum(value > 0.0 for value in recovery_pulses) == 1
    assert not env._backroll_recovery_active.item()
    assert not env._backroll_invalid.item()
    assert env._backroll_recovery_rearm_count.item() == 1
    assert env._backroll_cycle_count.item() == 0
    assert env._roulade_max.item() == 0.0
    assert not env._backroll_trunk_latch.item()
    assert not env._backroll_head_latch.item()
    assert env._backroll_recovered_cycle_armed.item()

    env._roulade_accum[:] = math.radians(355.0)
    env._roulade_max[:] = math.radians(355.0)
    env._roulade_paid[:] = math.radians(355.0)
    env._backroll_previous_frontier[:] = math.radians(355.0)
    env._backroll_trunk_latch[:] = True
    env._backroll_head_latch[:] = True
    cycle_hold_steps = math.ceil(
        mdp._BACKROLL_REPEAT_LANDING_HOLD_SECONDS / env.step_dt
    )
    for _ in range(cycle_hold_steps + 1):
        mdp.grounded_backroll_repeat_success_rate(env)
        env.common_step_counter += 1

    assert env._backroll_cycle_count.item() == 1
    assert env._backroll_recovered_and_rerolled.item() == pytest.approx(1.0)
    assert not env._backroll_recovered_cycle_armed.item()
    assert env._backroll_recovery_rearm_count.item() == 1


def test_one_shot_offaxis_rotation_cannot_earn_positive_progress(monkeypatch):
    env, asset = _fake_env()
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )
    monkeypatch.setattr(
        mdp,
        "_grounded_backroll_cycle_is_sagittal",
        lambda _env: torch.zeros(1, dtype=torch.bool),
    )
    env._backroll_repeat_mode[:] = False
    env._backroll_cycle_offaxis_rotation[:] = (
        mdp._BACKROLL_REPEAT_MAX_OFFAXIS_ROTATION + math.radians(1.0)
    )
    asset.data.root_link_ang_vel_b[:, 1] = -3.0
    assert mdp.grounded_backroll_progress(env).item() == 0.0
    assert mdp.grounded_backroll_completion_progress(env).item() == 0.0


def test_repeated_recovery_timeout_terminates_and_cannot_open_twice(monkeypatch):
    env, asset = _fake_env()
    env._backroll_repeat_mode[:] = True
    env._backroll_invalid[:] = True
    env._backroll_recovery_active[:] = True
    env._backroll_recovery_used[:] = True
    env._backroll_recovery_attempt_count[:] = 1.0
    env.scene.sensors["left_foot_ground_contact"].data.found[:] = 0.0
    env.scene.sensors["right_foot_ground_contact"].data.found[:] = 0.0
    env.scene.sensors["head_ground_contact"].data.found[:] = 1.0
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.zeros(1, dtype=torch.bool),
    )

    timeout_steps = math.ceil(
        mdp._BACKROLL_RECOVERY_TIMEOUT_SECONDS / env.step_dt
    )
    for _ in range(timeout_steps):
        mdp.grounded_backroll_recovery_progress(env)
        env.common_step_counter += 1

    assert not env._backroll_recovery_active.item()
    assert env._backroll_recovery_failed.item()
    assert mdp.grounded_backroll_invalid_termination(env).item()

    env._backroll_invalid[:] = False
    env._backroll_recovery_failed[:] = False
    env._roulade_accum[:] = math.radians(200.0)
    env._roulade_max[:] = math.radians(200.0)
    env._backroll_previous_frontier[:] = math.radians(200.0)
    env._backroll_cycle_offaxis_rotation[:] = (
        mdp._BACKROLL_REPEAT_MAX_OFFAXIS_ROTATION + math.radians(1.0)
    )
    asset.data.root_link_ang_vel_b.zero_()
    env.common_step_counter += 1
    mdp.grounded_backroll_progress(env)
    assert not env._backroll_recovery_active.item()
    assert env._backroll_recovery_attempt_count.item() == pytest.approx(1.0)
    assert mdp.grounded_backroll_invalid_termination(env).item()


def test_repeated_recovery_potential_is_delta_only(monkeypatch):
    env, asset = _fake_env()
    env._backroll_repeat_mode[:] = True
    env._backroll_invalid[:] = True
    env._backroll_recovery_active[:] = True
    env._backroll_recovery_used[:] = True
    env.scene.sensors["left_foot_ground_contact"].data.found[:] = 0.0
    env.scene.sensors["right_foot_ground_contact"].data.found[:] = 0.0
    env.scene.sensors["head_ground_contact"].data.found[:] = 1.0
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    s = 2.0**-0.5
    asset.data.root_link_quat_w[:] = torch.tensor([[s, 0.0, s, 0.0]])
    asset.data.root_link_pos_w[:, 2] = 0.06

    assert mdp.grounded_backroll_recovery_progress(env).item() == 0.0
    env.common_step_counter += 1
    asset.data.root_link_quat_w[:] = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    asset.data.root_link_pos_w[:, 2] = 0.10
    positive = mdp.grounded_backroll_recovery_progress(env).item()
    assert positive > 0.0
    env.common_step_counter += 1
    assert mdp.grounded_backroll_recovery_progress(env).item() == 0.0
    env.common_step_counter += 1
    asset.data.root_link_quat_w[:] = torch.tensor([[s, 0.0, s, 0.0]])
    asset.data.root_link_pos_w[:, 2] = 0.06
    negative = mdp.grounded_backroll_recovery_progress(env).item()
    assert negative < 0.0
    assert (positive + negative) * env.step_dt == pytest.approx(0.0, abs=1.0e-6)


def test_repeated_recovery_command_preserves_contract_and_flags_mode(monkeypatch):
    env, _asset = _fake_env()
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )
    base_command = torch.tensor([[0.2, -0.1, 0.05]])
    env.command_manager = SimpleNamespace(
        get_command=lambda name: base_command if name == "twist" else None
    )

    normal = mdp.grounded_backroll_recovery_command(env)
    assert normal.shape == (1, 3)
    assert torch.equal(normal, base_command)

    env.common_step_counter += 1
    env._backroll_repeat_mode[:] = True
    env._backroll_invalid[:] = True
    env._backroll_recovery_active[:] = True
    recovery = mdp.grounded_backroll_recovery_command(env)
    assert recovery[0, 0].item() == pytest.approx(1.0)
    assert recovery[0, 1:].tolist() == pytest.approx(base_command[0, 1:].tolist())


def test_repeated_recovery_blocks_all_roll_credit(monkeypatch):
    env, asset = _fake_env()
    env._backroll_repeat_mode[:] = True
    env._backroll_invalid[:] = True
    env._backroll_recovery_active[:] = True
    env._backroll_recovery_used[:] = True
    env._roulade_accum[:] = math.radians(355.0)
    env._roulade_max[:] = math.radians(355.0)
    env._roulade_paid[:] = math.radians(340.0)
    env._backroll_previous_frontier[:] = math.radians(340.0)
    env._backroll_completion_paid[:] = math.radians(300.0)
    env._backroll_trunk_latch[:] = True
    env._backroll_head_latch[:] = True
    asset.data.root_link_ang_vel_b[:, 1] = -6.0
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )

    assert mdp.grounded_backroll_progress(env).item() == 0.0
    assert mdp.grounded_backroll_speed_progress(env).item() == 0.0
    assert mdp.grounded_backroll_completion_progress(env).item() == 0.0
    assert mdp.grounded_backroll_contact_sequence(env).item() == 0.0
    assert mdp.grounded_backroll_repeat_success_rate(env).item() == 0.0
    assert env._backroll_cycle_count.item() == 0


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
