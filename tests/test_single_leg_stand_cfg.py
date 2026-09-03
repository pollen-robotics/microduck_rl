import pytest
from mjlab.tasks.registry import list_tasks

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_single_leg_stand_env_cfg import (
    COMMAND_NAME,
    EPISODE_LENGTH_S,
    STRICT_EPISODE_LENGTH_S,
    NONFOOT_SENSOR,
    MicroduckSingleLegStandRlCfg,
    make_microduck_single_leg_stand_env_cfg,
    make_microduck_single_leg_stand_strict_env_cfg,
)


def test_env_builds_and_keeps_the_61d_command_layout():
    cfg = make_microduck_single_leg_stand_env_cfg()
    assert cfg.episode_length_s == EPISODE_LENGTH_S == 6.0
    assert isinstance(
        cfg.commands[COMMAND_NAME], microduck_mdp.SingleLegStandCommandCfg
    )
    assert (
        cfg.observations["actor"].terms["command"].params["command_name"]
        == COMMAND_NAME
    )
    assert cfg.observations["actor"].terms["head_command"].params["dim"] == 4
    assert cfg.observations["actor"].terms["body_command"].params["dim"] == 6


def test_one_policy_trains_both_support_sides_with_mirror_loss():
    cmd = make_microduck_single_leg_stand_env_cfg().commands[COMMAND_NAME]
    assert cmd.left_prob == 0.5
    assert cmd.ranges.lin_vel_y == (-1.0, 1.0)
    assert MicroduckSingleLegStandRlCfg.algorithm.symmetry_cfg is not None
    assert (
        MicroduckSingleLegStandRlCfg.algorithm.symmetry_cfg["use_mirror_loss"] is True
    )


def test_play_side_can_be_forced(monkeypatch):
    monkeypatch.setenv("SINGLE_LEG_PLAY_SIDE", "left")
    assert (
        make_microduck_single_leg_stand_env_cfg(play=True)
        .commands[COMMAND_NAME]
        .fixed_side
        == -1
    )
    monkeypatch.setenv("SINGLE_LEG_PLAY_SIDE", "right")
    assert (
        make_microduck_single_leg_stand_env_cfg(play=True)
        .commands[COMMAND_NAME]
        .fixed_side
        == 1
    )
    monkeypatch.setenv("SINGLE_LEG_PLAY_SIDE", "invalid")
    with pytest.raises(ValueError):
        make_microduck_single_leg_stand_env_cfg(play=True)


def test_play_episode_length_can_be_extended(monkeypatch):
    monkeypatch.setenv("SINGLE_LEG_PLAY_EPISODE_S", "300")
    cfg = make_microduck_single_leg_stand_env_cfg(play=True)
    assert cfg.episode_length_s == 300.0
    assert cfg.commands[COMMAND_NAME].resampling_time_range == (300.0, 300.0)


def test_reward_signs_and_discovery_curriculum():
    cfg = make_microduck_single_leg_stand_env_cfg()
    assert cfg.rewards["single_leg_com"].weight > 0.0
    assert cfg.rewards["support_contact"].weight > 0.0
    assert cfg.rewards["swing_height"].weight == 0.0
    assert cfg.rewards["swing_contact"].weight == 0.0
    assert cfg.rewards["support_slip"].weight < 0.0
    assert cfg.rewards["excess_tilt"].weight < 0.0
    assert cfg.rewards["nonfoot_contact"].weight < 0.0

    lift = cfg.curriculum["swing_height_weight"].params["weight_stages"]
    contact = cfg.curriculum["swing_contact_weight"].params["weight_stages"]
    assert lift[0]["weight"] == 0.0 and lift[-1]["weight"] > 0.0
    assert contact[0]["weight"] == 0.0 and contact[-1]["weight"] < 0.0


def test_success_metrics_are_split_by_support_side():
    cfg = make_microduck_single_leg_stand_env_cfg()
    metrics = cfg.metrics
    assert NONFOOT_SENSOR in {sensor.name for sensor in cfg.scene.sensors}
    assert metrics["single_leg_success"].func is microduck_mdp.single_leg_hold_success
    assert metrics["single_leg_success"].params["hold_s"] == 1.0
    assert metrics["single_leg_success_left"].params["support_side"] == -1
    assert metrics["single_leg_success_right"].params["support_side"] == 1


def test_strict_phase_has_one_positive_task_reward_and_no_shaping_rewards():
    cfg = make_microduck_single_leg_stand_strict_env_cfg()
    for name in (
        "single_leg_com",
        "support_contact",
        "swing_height",
        "height_stand",
        "body_ang_vel",
        "angular_momentum",
        "excess_tilt",
    ):
        assert name not in cfg.rewards
    assert cfg.episode_length_s == STRICT_EPISODE_LENGTH_S == 10.0
    assert "fell_over" not in cfg.terminations
    assert (
        cfg.rewards["strict_single_leg_hold"].func
        is microduck_mdp.single_leg_hold_progress_reward
    )
    assert cfg.rewards["strict_single_leg_hold"].weight > 0.0
    assert cfg.rewards["swing_contact"].weight < 0.0
    assert cfg.rewards["touchdown"].weight == -5.0
    assert cfg.rewards["failed_episode"].weight == -500.0
    assert cfg.rewards["strict_single_leg_hold"].params["min_clearance"] == -1.0
    assert cfg.rewards["strict_single_leg_hold"].params["max_tilt_deg"] == 180.0
    assert cfg.rewards["strict_single_leg_hold"].params["require_com_inside"] is False
    assert cfg.metrics["single_leg_success"].params["min_clearance"] == -1.0
    assert cfg.metrics["single_leg_success"].params["max_tilt_deg"] == 180.0
    assert cfg.metrics["single_leg_success"].params["require_com_inside"] is False
    assert "swing_contact_weight" not in cfg.curriculum
    assert "push_magnitude" not in cfg.curriculum


def test_task_and_backlash_twin_are_registered():
    tasks = set(list_tasks())
    assert "Mjlab-SingleLegStand-Flat-MicroDuck" in tasks
    assert "Mjlab-SingleLegStand-Strict-Flat-MicroDuck" in tasks
    assert "Mjlab-SingleLegStand-Flat-Backlash-MicroDuck" in tasks
