from mjlab_microduck.tasks.microduck_standard_stairs_env_cfg import (
    MicroduckStairCurriculumRsiRlCfg,
    MicroduckStairContactMantleRsiRlCfg,
    MicroduckStairContactReleaseRsiRlCfg,
    MicroduckStairLipCommitmentRsiRlCfg,
    MicroduckStairLipCheckpointRsiRlCfg,
    MicroduckStairFrontierCollocationRsiRlCfg,
    MicroduckStairMediumDynamicsRsiRlCfg,
    MicroduckStairPhaseBalancedRsiRlCfg,
    MicroduckStairSoftDynamicsRsiRlCfg,
    STAIR_BRIDGE_SOLREF_TIME_CONSTANT,
    STAIR_DISCOVERY_SOLREF_TIME_CONSTANT,
    STAIR_MECHANISM_CURRICULUM_LEVELS,
    STAIR_MECHANISM_MIN_RISER_HEIGHT,
    STANDARD_RISER_HEIGHT,
    make_microduck_stair_curriculum_rsi_env_cfg,
    make_microduck_stair_contact_mantle_rsi_env_cfg,
    make_microduck_stair_contact_release_rsi_env_cfg,
    make_microduck_stair_lip_commitment_rsi_env_cfg,
    make_microduck_stair_lip_checkpoint_rsi_env_cfg,
    make_microduck_stair_frontier_collocation_rsi_env_cfg,
    make_microduck_stair_medium_dynamics_rsi_env_cfg,
    make_microduck_stair_phase_balanced_rsi_env_cfg,
    make_microduck_stair_soft_dynamics_rsi_env_cfg,
)


def test_phase_balanced_rsi_replays_four_roll_phases():
    cfg = make_microduck_stair_phase_balanced_rsi_env_cfg()
    bank = cfg.events["walker_state_bank"].params

    assert bank["bank_path"].endswith("full170-roulade-state-bank.pt")
    assert bank["phase_balanced"] is True
    assert bank["phase_bucket_count"] == 4
    assert bank["source_episode_step_range"] == (15, 60)
    assert cfg.episode_length_s == 8.0
    assert cfg.rewards["stair_first_tread_secured"].weight == 400.0
    assert MicroduckStairPhaseBalancedRsiRlCfg.max_iterations == 400
    assert MicroduckStairPhaseBalancedRsiRlCfg.save_interval == 25


def test_curriculum_rsi_preserves_actor_contract_and_real_stair_challenge():
    baseline = make_microduck_stair_phase_balanced_rsi_env_cfg()
    cfg = make_microduck_stair_curriculum_rsi_env_cfg()
    terrain = next(iter(cfg.scene.terrain.terrain_generator.sub_terrains.values()))
    bank = cfg.events["walker_state_bank"].params

    assert cfg.scene.terrain.terrain_generator.num_rows == (
        STAIR_MECHANISM_CURRICULUM_LEVELS
    )
    assert terrain.riser_height_range == (
        STAIR_MECHANISM_MIN_RISER_HEIGHT,
        STANDARD_RISER_HEIGHT,
    )
    assert terrain.difficulty_levels == STAIR_MECHANISM_CURRICULUM_LEVELS
    assert cfg.scene.terrain.max_init_terrain_level == 2
    assert cfg.events["route_challenge_levels"].params["standard_fraction"] == 0.25
    assert cfg.curriculum["terrain_levels"].func.__name__ == "route_terrain_levels"

    assert "local_x_range" not in bank
    assert bank["phase_aligned_local_x_range"] == (0.46, 0.62)
    assert bank["phase_aligned_x_jitter"] == 0.01
    assert tuple(cfg.observations["actor"].terms) == tuple(
        baseline.observations["actor"].terms
    )
    assert "stair_privileged_state" not in cfg.observations["actor"].terms
    assert "stair_privileged_state" in cfg.observations["critic"].terms

    fixed_proxies = (
        "stair_apex_or_mantle_frontier",
        "stair_riser_face_contact",
        "stair_first_tread_contact",
        "stair_assisted_lift",
        "stair_assisted_crossing",
        "stair_tread_support_frontier",
        "stair_first_tread_settle_quality",
    )
    assert all(cfg.rewards[name].weight == 0.0 for name in fixed_proxies)
    assert cfg.rewards["stair_first_tread_secured"].weight == 600.0
    assert cfg.rewards["stair_curriculum_mantle_frontier"].weight == 12.0
    clearance = cfg.rewards["stair_first_riser_clearance"]
    assert clearance.weight == 500.0
    assert "riser_height" not in clearance.params
    assert clearance.params["min_riser_height"] == STAIR_MECHANISM_MIN_RISER_HEIGHT
    assert clearance.params["max_riser_height"] == STANDARD_RISER_HEIGHT
    assert clearance.params["num_terrain_levels"] == STAIR_MECHANISM_CURRICULUM_LEVELS
    assert MicroduckStairCurriculumRsiRlCfg.max_iterations == 600
    assert MicroduckStairCurriculumRsiRlCfg.save_interval == 25


def test_contact_mantle_rsi_focuses_only_the_armed_last_mile():
    baseline = make_microduck_stair_curriculum_rsi_env_cfg()
    cfg = make_microduck_stair_contact_mantle_rsi_env_cfg()
    reward = cfg.rewards["stair_curriculum_contact_mantle_frontier"]

    assert tuple(cfg.observations["actor"].terms) == tuple(
        baseline.observations["actor"].terms
    )
    assert cfg.rewards["stair_curriculum_mantle_frontier"].weight == 0.0
    assert reward.weight == 12.0
    assert reward.params["x_start_margin"] == 0.005
    assert reward.params["x_target_margin"] == 0.040
    assert reward.params["z_start_margin"] == 0.005
    assert reward.params["z_target_margin"] == 0.025
    assert reward.params["support_sensor_name"] == "robot_ground_contact"
    assert cfg.rewards["stair_first_tread_secured"].weight == 600.0
    assert MicroduckStairContactMantleRsiRlCfg.max_iterations == 300
    assert MicroduckStairContactMantleRsiRlCfg.save_interval == 25


def test_contact_continuation_uses_only_full_height_challenge_rows():
    baseline = make_microduck_stair_contact_mantle_rsi_env_cfg()
    soft = make_microduck_stair_soft_dynamics_rsi_env_cfg()
    medium = make_microduck_stair_medium_dynamics_rsi_env_cfg()

    for cfg in (soft, medium):
        assert tuple(cfg.observations["actor"].terms) == tuple(
            baseline.observations["actor"].terms
        )
        assert cfg.scene.terrain.max_init_terrain_level == (
            STAIR_MECHANISM_CURRICULUM_LEVELS - 1
        )
        assert cfg.events["route_challenge_levels"].params["standard_fraction"] == 1.0
        assert "terrain_levels" not in cfg.curriculum
        assert cfg.rewards["stair_first_tread_contact"].weight == 150.0
        assert cfg.rewards["stair_tread_support_frontier"].weight == 15.0
        assert cfg.rewards["stair_first_tread_secured"].weight == 600.0

    assert STAIR_DISCOVERY_SOLREF_TIME_CONSTANT == 0.10
    assert STAIR_BRIDGE_SOLREF_TIME_CONSTANT == 0.065
    assert soft.scene.spec_fn.__name__ == "_soft_stair_discovery_contacts"
    assert medium.scene.spec_fn.__name__ == "_medium_stair_bridge_contacts"
    assert MicroduckStairSoftDynamicsRsiRlCfg.max_iterations == 200
    assert MicroduckStairMediumDynamicsRsiRlCfg.max_iterations == 200


def test_contact_release_replaces_contact_jackpots_with_one_transition_gate():
    baseline = make_microduck_stair_contact_mantle_rsi_env_cfg()
    cfg = make_microduck_stair_contact_release_rsi_env_cfg()
    release = cfg.rewards["stair_contact_loaded_release"]

    assert tuple(cfg.observations["actor"].terms) == tuple(
        baseline.observations["actor"].terms
    )
    assert cfg.scene.terrain.max_init_terrain_level == (
        STAIR_MECHANISM_CURRICULUM_LEVELS - 1
    )
    assert cfg.events["route_challenge_levels"].params["standard_fraction"] == 1.0
    assert "terrain_levels" not in cfg.curriculum
    assert cfg.rewards["stair_first_tread_contact"].weight == 0.0
    assert cfg.rewards["stair_tread_support_frontier"].weight == 0.0
    assert cfg.rewards["stair_curriculum_contact_mantle_frontier"].weight == 0.0
    assert release.weight == 50.0
    assert release.params["arm_hold_steps"] == 2
    assert release.params["release_window_steps"] == 8
    assert release.params["min_forward_speed"] == 0.05
    assert release.params["min_vertical_speed"] == 0.08
    assert release.params["target_forward_delta"] == 0.040
    assert release.params["target_vertical_delta"] == 0.025
    assert cfg.rewards["stair_first_tread_secured"].weight == 600.0
    assert MicroduckStairContactReleaseRsiRlCfg.max_iterations == 75
    assert MicroduckStairContactReleaseRsiRlCfg.save_interval == 25


def test_lip_commitment_starts_near_stair_and_requires_spatial_conversion():
    baseline = make_microduck_stair_contact_release_rsi_env_cfg()
    cfg = make_microduck_stair_lip_commitment_rsi_env_cfg()
    bank = cfg.events["walker_state_bank"].params
    commitment = cfg.rewards["stair_contact_lip_commitment"]

    assert tuple(cfg.observations["actor"].terms) == tuple(
        baseline.observations["actor"].terms
    )
    assert bank["phase_balanced"] is False
    assert bank["late_fraction"] == 0.50
    assert bank["late_source_episode_step_range"] == (38, 60)
    assert bank["phase_aligned_local_x_range"] == (0.54, 0.64)
    assert bank["phase_aligned_x_jitter"] == 0.005
    assert cfg.rewards["stair_contact_loaded_release"].weight == 0.0
    assert commitment.weight == 50.0
    assert commitment.params["impulse_window_steps"] == 12
    assert commitment.params["min_forward_velocity_gain"] == 0.05
    assert commitment.params["min_vertical_velocity_gain"] == 0.08
    assert commitment.params["commitment_delay_steps"] == 4
    assert commitment.params["commitment_hold_steps"] == 2
    assert commitment.params["commitment_x"] == 0.645
    assert commitment.params["commitment_z"] == 0.175
    assert cfg.rewards["stair_first_riser_clearance"].weight == 500.0
    assert cfg.rewards["stair_first_tread_secured"].weight == 600.0
    assert MicroduckStairLipCommitmentRsiRlCfg.max_iterations == 75
    assert MicroduckStairLipCommitmentRsiRlCfg.save_interval == 25


def test_lip_checkpoint_uses_multiplicative_signed_progress_near_stair():
    baseline = make_microduck_stair_lip_commitment_rsi_env_cfg()
    cfg = make_microduck_stair_lip_checkpoint_rsi_env_cfg()
    bank = cfg.events["walker_state_bank"].params
    checkpoint = cfg.rewards["stair_contact_lip_checkpoint_potential"]

    assert tuple(cfg.observations["actor"].terms) == tuple(
        baseline.observations["actor"].terms
    )
    assert bank["late_fraction"] == 0.75
    assert bank["phase_aligned_local_x_range"] == (0.54, 0.64)
    assert cfg.rewards["stair_contact_lip_commitment"].weight == 1.0e-6
    assert cfg.rewards["stair_assisted_approach"].weight == 0.0
    assert checkpoint.weight == 10.0
    reward_names = tuple(cfg.rewards)
    assert reward_names.index("stair_contact_lip_commitment") < reward_names.index(
        "stair_contact_lip_checkpoint_potential"
    )
    assert checkpoint.params["arm_hold_steps"] == 2
    assert checkpoint.params["target_hold_steps"] == 2
    assert checkpoint.params["x_start"] == 0.540
    assert checkpoint.params["x_target"] == 0.665
    assert checkpoint.params["z_start"] == 0.100
    assert checkpoint.params["z_target"] == 0.175
    assert cfg.rewards["stair_first_riser_clearance"].weight == 500.0
    assert cfg.rewards["stair_first_tread_secured"].weight == 600.0
    assert MicroduckStairLipCheckpointRsiRlCfg.max_iterations == 75
    assert MicroduckStairLipCheckpointRsiRlCfg.save_interval == 25


def test_frontier_collocation_retains_measured_near_lip_reset_distribution():
    baseline = make_microduck_stair_lip_checkpoint_rsi_env_cfg()
    cfg = make_microduck_stair_frontier_collocation_rsi_env_cfg()
    bank = cfg.events["walker_state_bank"].params
    frontier = cfg.rewards["stair_coupled_frontier_collocation"]

    assert tuple(cfg.observations["actor"].terms) == tuple(
        baseline.observations["actor"].terms
    )
    assert cfg.episode_length_s == 2.0
    assert bank["source_episode_step_range"] == (15, 60)
    assert bank["min_vertical_speed"] == -0.25
    assert bank["phase_balanced"] is False
    assert "local_x_range" not in bank
    assert bank["phase_aligned_local_x_range"] == (0.54, 0.64)
    assert bank["phase_aligned_x_jitter"] == 0.005
    assert bank["local_y_range"] == (-0.08, 0.08)
    assert bank["late_fraction"] == 0.75
    assert bank["late_source_episode_step_range"] == (38, 60)
    assert cfg.rewards["stair_contact_lip_commitment"].weight == 1.0e-6
    assert cfg.rewards["stair_contact_lip_checkpoint_potential"].weight == 0.0
    assert frontier.weight == 30.0
    assert frontier.params["arm_after_control_steps"] == 2
    assert frontier.params["x_start"] == 0.540
    assert frontier.params["x_target"] == 0.665
    assert frontier.params["z_start"] == 0.100
    assert frontier.params["z_target"] == 0.175
    assert frontier.params["target_bonus"] == 4.0
    reward_names = tuple(cfg.rewards)
    assert reward_names.index("stair_contact_lip_commitment") < reward_names.index(
        "stair_coupled_frontier_collocation"
    )
    assert cfg.rewards["stair_first_riser_clearance"].weight == 500.0
    assert cfg.rewards["stair_first_tread_secured"].weight == 600.0
    assert MicroduckStairFrontierCollocationRsiRlCfg.max_iterations == 75
    assert MicroduckStairFrontierCollocationRsiRlCfg.save_interval == 25
