from mjlab_microduck.tasks.microduck_step_up_env_cfg import (
    APPROACH_DISTANCE_RANGE,
    MicroduckStepUpFullRlCfg,
    MicroduckStepUpRlCfg,
    UpStepTerrainCfg,
    make_microduck_step_up_env_cfg,
    make_microduck_step_up_full_env_cfg,
)


def test_step_up_uses_curriculum_terrain_and_starts_easy():
    cfg = make_microduck_step_up_env_cfg()
    gen = cfg.scene.terrain.terrain_generator
    assert gen is not None and gen.curriculum is True and gen.num_rows == 10
    assert cfg.scene.terrain.max_init_terrain_level == 0
    terrain = next(iter(gen.sub_terrains.values()))
    assert isinstance(terrain, UpStepTerrainCfg)
    assert terrain.approach_distance_range == APPROACH_DISTANCE_RANGE


def test_play_scene_is_fixed_at_full_25_mm_difficulty():
    cfg = make_microduck_step_up_env_cfg(play=True)
    gen = cfg.scene.terrain.terrain_generator
    assert gen is not None and gen.curriculum is False
    assert gen.difficulty_range == (1.0, 1.0)
    assert cfg.commands["twist"].ranges.lin_vel_x == (0.0, 0.30)
    assert cfg.commands["twist"].ranges.lin_vel_y == (-0.10, 0.10)
    assert cfg.commands["twist"].ranges.ang_vel_z == (-0.10, 0.10)


def test_full_finetune_scene_is_always_25_mm_without_curriculum():
    cfg = make_microduck_step_up_full_env_cfg()
    gen = cfg.scene.terrain.terrain_generator
    assert gen is not None and gen.curriculum is False and gen.num_rows == 1
    assert gen.difficulty_range == (1.0, 1.0)
    assert cfg.scene.terrain.max_init_terrain_level is None
    assert "terrain_levels" not in cfg.curriculum


def test_step_skill_is_forward_only_and_keeps_61d_command_slots():
    cfg = make_microduck_step_up_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.ranges.lin_vel_x == (0.30, 0.30)
    assert cmd.ranges.lin_vel_y == (0.0, 0.0)
    assert cmd.ranges.ang_vel_z == (0.0, 0.0)
    assert "head_command" in cfg.observations["actor"].terms
    assert "body_command" in cfg.observations["actor"].terms
    assert "standing_envs" not in cfg.curriculum
    assert "push_robot" not in cfg.events


def test_skill_discovery_does_not_start_with_a_motion_blocking_tax():
    cfg = make_microduck_step_up_env_cfg()
    assert cfg.rewards["action_rate_l2"].weight == -0.01
    assert "action_rate_weight" not in cfg.curriculum


def test_success_is_one_shot_and_requires_upper_floor_arrival():
    cfg = make_microduck_step_up_env_cfg()
    assert cfg.rewards["step_up_success"].weight == 200.0
    assert "step_complete" in cfg.terminations
    reward_params = cfg.rewards["step_up_success"].params
    termination_params = cfg.terminations["step_complete"].params
    assert reward_params == termination_params
    assert 0.10 <= reward_params["min_trunk_height"] < 0.12


def test_full_height_finetune_has_hierarchical_progress_rewards():
    cfg = make_microduck_step_up_full_env_cfg()
    assert cfg.rewards["step_forward_velocity"].weight == 4.0
    assert cfg.rewards["one_foot_over_step"].weight == 100.0
    assert cfg.rewards["two_feet_over_step"].weight == 150.0
    assert cfg.rewards["one_foot_over_step"].params["required_feet"] == 1
    assert cfg.rewards["two_feet_over_step"].params["required_feet"] == 2


def test_foot_lift_target_clears_25_mm_with_margin():
    cfg = make_microduck_step_up_env_cfg()
    assert cfg.rewards["foot_clearance"].params["target_height"] == 0.040
    assert cfg.rewards["foot_swing_height"].params["target_height"] == 0.040


def test_runner_has_dedicated_experiment_name():
    assert MicroduckStepUpRlCfg.experiment_name == "step_up"
    assert MicroduckStepUpRlCfg.max_iterations == 6_000
    assert MicroduckStepUpFullRlCfg.experiment_name == "step_up_full"
    assert MicroduckStepUpFullRlCfg.max_iterations == 1_000
