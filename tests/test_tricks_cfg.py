import math

from mjlab_microduck.tasks.microduck_backflip_env_cfg import (
    MicroduckBackflipRlCfg,
    make_microduck_backflip_env_cfg,
)
from mjlab_microduck.tasks.microduck_headstand_env_cfg import (
    MicroduckHeadstandRlCfg,
    make_microduck_headstand_env_cfg,
)
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    make_microduck_roulade_env_cfg,
)
from mjlab_microduck.tasks.microduck_stairs_env_cfg import (
    MicroduckStairsRlCfg,
    make_microduck_stairs_env_cfg,
)
from mjlab_microduck.tasks.microduck_standard_stairs_env_cfg import (
    ROUTE_CURRICULUM_LEVELS,
    ROUTE_MIN_RISER_HEIGHT,
    STANDARD_GOAL_DISTANCE,
    STANDARD_NUM_STEPS,
    STANDARD_RISER_HEIGHT,
    STANDARD_STAIR_START_DISTANCE,
    STANDARD_STANDING_ROOT_HEIGHT,
    STANDARD_TOP_ROOT_HEIGHT,
    STANDARD_TREAD_DEPTH,
    MicroduckRouteStairsRlCfg,
    MicroduckStairSpecialistRlCfg,
    make_microduck_route_stairs_env_cfg,
    make_microduck_stair_specialist_env_cfg,
    make_microduck_standard_stairs_env_cfg,
)


def test_stairs_are_dedicated_low_rise_terrain():
    cfg = make_microduck_stairs_env_cfg()
    generator = cfg.scene.terrain.terrain_generator
    assert cfg.scene.terrain.terrain_type == "generator"
    assert set(generator.sub_terrains) == {"flat", "pyramid_stairs"}
    stairs = generator.sub_terrains["pyramid_stairs"]
    assert stairs.step_height_range == (0.0, 0.015)
    assert stairs.step_width == 0.15
    assert cfg.sim.nconmax == 200
    assert MicroduckStairsRlCfg.experiment_name == "microduck_stairs"


def test_stair_play_config_disables_curriculum_for_viewing():
    cfg = make_microduck_stairs_env_cfg(play=True)
    generator = cfg.scene.terrain.terrain_generator
    assert generator.curriculum is False
    assert generator.num_rows == 5
    assert generator.num_cols == 5


def test_standard_stairs_have_fixed_goal_and_forward_defaults():
    cfg = make_microduck_standard_stairs_env_cfg()
    generator = cfg.scene.terrain.terrain_generator
    terrain = generator.sub_terrains["standard_stairs_a"]
    command = cfg.commands["twist"]

    assert terrain.riser_height == STANDARD_RISER_HEIGHT == 0.17
    assert terrain.tread_depth == STANDARD_TREAD_DEPTH == 0.28
    assert terrain.num_steps == STANDARD_NUM_STEPS == 5
    assert cfg.episode_length_s == 12.0
    assert command.ranges.lin_vel_x == (0.10, 0.30)
    assert command.rel_turn_in_place_envs == 0.0
    assert command.heading_command is False
    assert cfg.rewards["stair_goal_progress"].params["goal_distance"] == STANDARD_GOAL_DISTANCE
    assert cfg.rewards["stair_top_goal"].params["goal_height"] == STANDARD_TOP_ROOT_HEIGHT
    assert cfg.rewards["stair_top_goal"].params["x_tolerance"] == 0.40
    assert cfg.rewards["stair_top_goal"].weight > cfg.rewards["stair_top_approach"].weight
    assert cfg.rewards["track_linear_velocity"].func.__name__ == "stair_approach_linear_tracking"
    assert cfg.rewards["track_linear_velocity"].params["stair_start_distance"] == STANDARD_STAIR_START_DISTANCE

    play_cfg = make_microduck_standard_stairs_env_cfg(play=True)
    play_generator = play_cfg.scene.terrain.terrain_generator
    assert play_generator.num_rows == 2
    assert play_generator.num_cols == 2
    assert "terrain_levels" not in play_cfg.curriculum
    assert play_cfg.events["stair_viewer_grid"].params["terrain_levels"] == (0, 1, 0, 1)
    assert play_cfg.events["stair_viewer_grid"].params["terrain_types"] == (0, 0, 1, 1)


def test_route_stairs_walk_first_and_progress_to_standard_height():
    cfg = make_microduck_route_stairs_env_cfg()
    generator = cfg.scene.terrain.terrain_generator
    terrain = generator.sub_terrains["route_stairs"]
    command = cfg.commands["twist"]

    assert generator.num_rows == ROUTE_CURRICULUM_LEVELS == 16
    assert generator.num_cols == 1
    assert generator.curriculum is True
    assert terrain.riser_height_range == (ROUTE_MIN_RISER_HEIGHT, STANDARD_RISER_HEIGHT)
    assert cfg.scene.terrain.max_init_terrain_level == 0
    assert cfg.episode_length_s == 30.0
    assert command.ranges.lin_vel_x == (0.20, 0.28)
    assert command.ranges.lin_vel_y == (0.0, 0.0)
    assert command.ranges.ang_vel_z == (0.0, 0.0)
    assert "push_robot" not in cfg.events
    assert cfg.scene.entities["robot"].spec_fn.__name__ == "get_standup_spec"
    assert cfg.scene.entities["robot"].spec_fn().geom("trunk_shell_collision") is not None
    assert {sensor.name for sensor in cfg.scene.sensors} >= {
        "head_ground_contact",
        "robot_ground_contact",
        "trunk_ground_contact",
    }
    assert "fell_over" not in cfg.terminations
    assert cfg.rewards["stair_top_approach"].params["upright_power"] == 0.0
    assert cfg.rewards["stair_goal_progress"].params["corridor_half_width"] < 0.45
    assert cfg.rewards["stair_top_approach"].params["corridor_half_width"] < 0.45
    assert cfg.rewards["stair_top_goal"].params["corridor_half_width"] < 0.45
    assert cfg.rewards["foot_clearance"].weight == 0.0
    assert cfg.rewards["foot_swing_height"].weight == 0.0
    assert cfg.rewards["body_ang_vel"].weight == 0.0
    assert cfg.rewards["dof_pos_limits"].weight == 0.0
    assert cfg.rewards["self_collisions"].weight == 0.0
    assert cfg.rewards["action_rate_l2"].weight == -0.001
    assert cfg.rewards["head_pose_tracking"].weight == 0.0
    assert cfg.rewards["stair_first_riser_clearance"].weight == 150.0
    assert cfg.rewards["stair_top_goal"].weight == 600.0
    assert set(cfg.curriculum) == {"terrain_levels"}
    assert cfg.curriculum["terrain_levels"].func.__name__ == "route_terrain_levels"
    assert cfg.rewards["upright"].func.__name__ == "stair_approach_upright"
    assert cfg.rewards["upright"].weight == 2.0
    assert cfg.events["route_challenge_levels"].params["standard_fraction"] == 0.35
    assert cfg.events["route_state_curriculum"].params["near_face_fraction"] == 0.20
    assert cfg.events["route_state_curriculum"].params["partial_mantle_fraction"] == 0.20
    assert cfg.events["route_state_curriculum"].params["on_tread_fraction"] == 0.10
    assert cfg.observations["actor"].terms["body_command"].func.__name__ == "stair_route_cues"
    assert cfg.observations["critic"].terms["body_command"].func.__name__ == "stair_route_cues"
    assert MicroduckRouteStairsRlCfg.save_interval == 50

    play_cfg = make_microduck_route_stairs_env_cfg(play=True)
    assert "terrain_levels" not in play_cfg.curriculum
    assert play_cfg.events["route_challenge_levels"].params["standard_fraction"] == 1.0
    assert play_cfg.events["route_state_curriculum"].params["near_face_fraction"] == 0.0
    assert cfg.rewards["stair_top_goal"].params["goal_height_range"] == (
        STANDARD_STANDING_ROOT_HEIGHT + STANDARD_NUM_STEPS * ROUTE_MIN_RISER_HEIGHT,
        STANDARD_TOP_ROOT_HEIGHT,
    )
    assert MicroduckRouteStairsRlCfg.experiment_name == "microduck_stair_route"


def test_stair_specialist_uses_only_full_height_handoff_states():
    cfg = make_microduck_stair_specialist_env_cfg()
    generator = cfg.scene.terrain.terrain_generator
    terrain = generator.sub_terrains["route_stairs"]
    reset = cfg.events["route_state_curriculum"].params

    assert terrain.riser_height == STANDARD_RISER_HEIGHT == 0.17
    assert terrain.riser_height_range is None
    assert terrain.tread_depth == STANDARD_TREAD_DEPTH == 0.28
    assert terrain.num_steps == STANDARD_NUM_STEPS == 5
    assert "terrain_levels" not in cfg.curriculum
    assert cfg.events["route_challenge_levels"].params["standard_fraction"] == 1.0
    assert reset["near_face_fraction"] == 0.50
    assert reset["partial_mantle_fraction"] == 0.30
    assert reset["on_tread_fraction"] == 0.20
    assert reset["min_tread_step"] == reset["max_tread_step"] == 1
    assert sum(
        reset[name]
        for name in (
            "near_face_fraction",
            "partial_mantle_fraction",
            "on_tread_fraction",
        )
    ) == 1.0
    assert cfg.episode_length_s == 8.0
    assert cfg.rewards["track_linear_velocity"].weight == 0.0
    assert cfg.rewards["track_angular_velocity"].weight == 0.0
    assert cfg.rewards["action_rate_l2"].weight == 0.0
    assert cfg.rewards["stair_first_riser_frontier"].weight == 4.0
    assert cfg.rewards["stair_first_riser_clearance"].weight == 200.0
    assert MicroduckStairSpecialistRlCfg.experiment_name == (
        "microduck_stair_specialist"
    )
    assert MicroduckStairSpecialistRlCfg.max_iterations == 800
    assert MicroduckStairSpecialistRlCfg.algorithm.learning_rate == 1.0e-4
    assert MicroduckStairSpecialistRlCfg.algorithm.schedule == "fixed"


def test_headstand_has_exclusive_contact_gate_and_shared_actor_layout():
    cfg = make_microduck_headstand_env_cfg()
    roulade = make_microduck_roulade_env_cfg()
    assert list(cfg.observations["actor"].terms) == list(roulade.observations["actor"].terms)
    assert set(cfg.rewards) == {
        "headstand_contact",
        "headstand_hold",
        "headstand_height",
        "action_rate_l2",
        "self_collisions",
    }
    assert cfg.rewards["headstand_hold"].weight > cfg.rewards["headstand_contact"].weight
    assert "fell_over" not in cfg.terminations
    assert "set_headstand_state" in cfg.events
    params = cfg.events["set_headstand_state"].params
    assert params["midroll_pitch_min"] == math.radians(90.0)
    assert params["midroll_pitch_max"] == math.radians(90.0)
    assert MicroduckHeadstandRlCfg.experiment_name == "microduck_headstand"


def test_backflip_uses_progress_frontier_and_landing_gate():
    cfg = make_microduck_backflip_env_cfg()
    roulade = make_microduck_roulade_env_cfg()
    assert list(cfg.observations["actor"].terms) == list(roulade.observations["actor"].terms)
    assert set(cfg.rewards) == {
        "backflip_progress",
        "backflip_landing",
        "action_rate_l2",
        "self_collisions",
    }
    assert cfg.rewards["backflip_progress"].weight > 0.0
    assert cfg.rewards["backflip_landing"].params["gate_lo"] == math.radians(300.0)
    assert cfg.rewards["backflip_landing"].params["gate_hi"] == math.radians(355.0)
    assert "fell_over" not in cfg.terminations
    assert "set_backflip_state" in cfg.events
    assert MicroduckBackflipRlCfg.experiment_name == "microduck_backflip"
