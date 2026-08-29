import math
from types import SimpleNamespace

import torch

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
    MicroduckAssistedStairSpecialistRlCfg,
    MicroduckStairApexMantleRlCfg,
    MicroduckStairRouladeBankRlCfg,
    MicroduckStairBridgeSpecialistRlCfg,
    MicroduckStairLaunchBankRlCfg,
    MicroduckStairWalkerBankRlCfg,
    MicroduckRouteStairsRlCfg,
    MicroduckStairSpecialistRlCfg,
    make_microduck_assisted_stair_specialist_env_cfg,
    make_microduck_stair_apex_mantle_env_cfg,
    make_microduck_stair_roulade_bank_env_cfg,
    make_microduck_stair_bridge_specialist_env_cfg,
    make_microduck_stair_launch_bank_env_cfg,
    make_microduck_stair_walker_bank_env_cfg,
    make_microduck_route_stairs_env_cfg,
    make_microduck_stair_specialist_env_cfg,
    make_microduck_standard_stairs_env_cfg,
)
from mjlab_microduck.tasks.stair_action import StairHistoryJointPositionAction


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
    assert cfg.observations["actor"].terms["body_command"].params["cue_distance"] == 0.30
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
    assert cfg.actions["joint_pos"].__class__.__name__ == (
        "StairHistoryJointPositionActionCfg"
    )


def test_stair_action_restores_history_after_manager_reset():
    term = StairHistoryJointPositionAction.__new__(StairHistoryJointPositionAction)
    current = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    previous = current - 0.25
    previous_previous = current - 0.50
    manager = SimpleNamespace(
        _action=torch.zeros_like(current),
        _prev_action=torch.zeros_like(current),
        _prev_prev_action=torch.zeros_like(current),
    )
    term._env = SimpleNamespace(
        action_manager=manager,
        _stair_reset_action_history={
            "current": current,
            "previous": previous,
            "previous_previous": previous_previous,
        },
    )
    term._raw_actions = torch.zeros_like(current)
    term._processed_actions = torch.zeros_like(current)
    term._scale = 2.0
    term._offset = 0.5

    ids = torch.tensor([1])
    term.reset(ids)

    assert torch.equal(manager._action[ids], current[ids])
    assert torch.equal(manager._prev_action[ids], previous[ids])
    assert torch.equal(manager._prev_prev_action[ids], previous_previous[ids])
    assert torch.equal(term._processed_actions[ids], current[ids] * 2.0 + 0.5)


def test_assisted_stair_specialist_keeps_full_geometry_and_stage_a_gates():
    cfg = make_microduck_assisted_stair_specialist_env_cfg()
    generator = cfg.scene.terrain.terrain_generator
    reset = cfg.events["route_state_curriculum"]
    clearance = cfg.rewards["stair_first_riser_clearance"]

    assert "route_challenge_levels" not in cfg.events
    for terrain in generator.sub_terrains.values():
        assert terrain.riser_height == 0.17
        assert terrain.riser_height_range is None
        assert terrain.tread_depth == 0.28
        assert terrain.num_steps == 5
    assert reset.func.__name__ == "reset_assisted_stair_states"
    assert reset.params["lip_release_fraction"] == 0.50
    assert reset.params["shell_brace_fraction"] == 0.25
    assert reset.params["tread_recovery_fraction"] == 0.15
    assert reset.params["real_handoff_fraction"] == 0.10
    assert cfg.episode_length_s == 4.0
    assert clearance.weight == 400.0
    assert clearance.params["riser_height"] == 0.17
    assert clearance.params["hold_time_s"] == 0.08
    assert cfg.rewards["stair_assisted_approach"].weight == 0.5
    assert cfg.rewards["stair_assisted_lift"].weight == 3.0
    assert cfg.rewards["stair_assisted_crossing"].weight == 5.0
    assert cfg.rewards["stair_first_tread_stable"].weight == 200.0
    assert cfg.rewards["stair_goal_progress"].weight == 0.0
    assert cfg.rewards["stair_top_goal"].weight == 0.0
    assert MicroduckAssistedStairSpecialistRlCfg.max_iterations == 100
    assert MicroduckAssistedStairSpecialistRlCfg.save_interval == 25
    assert (
        MicroduckAssistedStairSpecialistRlCfg.actor.distribution_cfg["init_std"]
        == 0.45
    )
    assert (
        MicroduckAssistedStairSpecialistRlCfg.algorithm.learning_rate == 5.0e-5
    )
    assert MicroduckAssistedStairSpecialistRlCfg.algorithm.max_grad_norm == 0.5


def test_stair_bridge_keeps_full_height_and_prioritizes_shell_and_tread():
    cfg = make_microduck_stair_bridge_specialist_env_cfg()
    reset = cfg.events["route_state_curriculum"].params
    clearance = cfg.rewards["stair_first_riser_clearance"]

    for terrain in cfg.scene.terrain.terrain_generator.sub_terrains.values():
        assert terrain.riser_height == 0.17
        assert terrain.riser_height_range is None
        assert terrain.tread_depth == 0.28
    assert reset["lip_release_fraction"] == 0.30
    assert reset["shell_brace_fraction"] == 0.40
    assert reset["tread_recovery_fraction"] == 0.25
    assert reset["real_handoff_fraction"] == 0.05
    assert clearance.params["z_margin"] == 0.025
    assert clearance.params["max_vertical_speed"] == 0.45
    assert clearance.params["hold_time_s"] == 0.12
    assert cfg.rewards["stair_first_tread_secured"].weight == 300.0
    assert cfg.rewards["stair_first_tread_settle_quality"].weight == 2.0
    assert MicroduckStairBridgeSpecialistRlCfg.max_iterations == 200
    assert (
        MicroduckStairBridgeSpecialistRlCfg.actor.distribution_cfg["init_std"]
        == 0.35
    )
    assert MicroduckStairBridgeSpecialistRlCfg.algorithm.learning_rate == 3.0e-5


def test_walker_bank_stage_uses_real_handoff_states_on_full_home_stairs():
    cfg = make_microduck_stair_walker_bank_env_cfg()
    reset = cfg.events["route_state_curriculum"].params

    for terrain in cfg.scene.terrain.terrain_generator.sub_terrains.values():
        assert terrain.riser_height == 0.17
        assert terrain.riser_height_range is None
        assert terrain.tread_depth == 0.28
        assert terrain.num_steps == 5
    assert reset["lip_release_fraction"] == 0.20
    assert reset["shell_brace_fraction"] == 0.25
    assert reset["tread_recovery_fraction"] == 0.25
    assert reset["real_handoff_fraction"] == 0.30
    bank_event = cfg.events["walker_state_bank"]
    assert bank_event.func.__name__ == "WalkerStateBankReset"
    assert bank_event.params["bank_path"].endswith("full170-walker-state-bank.pt")
    assert MicroduckStairWalkerBankRlCfg.max_iterations == 100
    assert MicroduckStairWalkerBankRlCfg.save_interval == 25
    assert (
        MicroduckStairWalkerBankRlCfg.actor.distribution_cfg["init_std"] == 0.30
    )
    assert MicroduckStairWalkerBankRlCfg.algorithm.learning_rate == 2.0e-5


def test_launch_bank_stage_uses_early_walker_states_and_impulse_milestones():
    cfg = make_microduck_stair_launch_bank_env_cfg()
    reset = cfg.events["route_state_curriculum"].params

    for terrain in cfg.scene.terrain.terrain_generator.sub_terrains.values():
        assert terrain.riser_height == 0.17
        assert terrain.riser_height_range is None
        assert terrain.tread_depth == 0.28
        assert terrain.num_steps == 5
    assert cfg.episode_length_s == 6.0
    assert reset["lip_release_fraction"] == 0.15
    assert reset["shell_brace_fraction"] == 0.15
    assert reset["tread_recovery_fraction"] == 0.10
    assert reset["real_handoff_fraction"] == 0.60
    assert sum(
        reset[name]
        for name in (
            "lip_release_fraction",
            "shell_brace_fraction",
            "tread_recovery_fraction",
            "real_handoff_fraction",
        )
    ) == 1.0
    bank_event = cfg.events["walker_state_bank"]
    assert bank_event.params["bank_path"].endswith(
        "full170-walker-launch-state-bank.pt"
    )
    assert cfg.rewards["stair_preload_frontier"].weight == 2.0
    assert cfg.rewards["stair_launch_sequence"].weight == 50.0
    assert cfg.rewards["stair_launch_sequence"].params["min_upward_speed"] == 0.30
    assert cfg.rewards["stair_takeoff_frontier"].weight == 3.0
    assert cfg.rewards["stair_assisted_lift"].params["x_gate"] == 0.52
    assert cfg.rewards["stair_first_tread_stable"].weight == 100.0
    assert MicroduckStairLaunchBankRlCfg.max_iterations == 200
    assert MicroduckStairLaunchBankRlCfg.save_interval == 25
    assert (
        MicroduckStairLaunchBankRlCfg.actor.distribution_cfg["init_std"] == 0.34
    )
    assert MicroduckStairLaunchBankRlCfg.algorithm.learning_rate == 2.5e-5


def test_apex_mantle_stage_rewards_capability_from_real_handoff():
    cfg = make_microduck_stair_apex_mantle_env_cfg()
    reset = cfg.events["route_state_curriculum"].params

    for terrain in cfg.scene.terrain.terrain_generator.sub_terrains.values():
        assert terrain.riser_height == 0.17
        assert terrain.riser_height_range is None
        assert terrain.tread_depth == 0.28
        assert terrain.num_steps == 5
    assert reset["lip_release_fraction"] == 0.25
    assert reset["shell_brace_fraction"] == 0.15
    assert reset["tread_recovery_fraction"] == 0.0
    assert reset["real_handoff_fraction"] == 0.60
    assert reset["lip_local_x_range"] == (0.515, 0.555)
    assert reset["lip_vertical_speed_range"] == (0.45, 0.90)
    assert reset["shell_local_x_range"] == (0.540, 0.590)
    assert reset["shell_pitch_deg_range"] == (8.0, 30.0)
    bank_event = cfg.events["walker_state_bank"]
    assert bank_event.params["bank_path"].endswith(
        "full170-walker-launch-state-bank.pt"
    )
    assert cfg.rewards["stair_preload_frontier"].weight == 0.0
    assert cfg.rewards["stair_launch_sequence"].weight == 0.0
    assert cfg.rewards["stair_takeoff_frontier"].weight == 0.0
    assert cfg.rewards["stair_apex_or_mantle_frontier"].weight == 6.0
    assert cfg.rewards["stair_assisted_lift"].weight == 2.0
    assert cfg.rewards["stair_assisted_lift"].params["x_gate"] == 0.50
    assert cfg.rewards["stair_assisted_crossing"].weight == 4.0
    assert cfg.rewards["stair_first_tread_stable"].weight == 0.0
    assert cfg.rewards["stair_first_tread_secured"].weight == 100.0
    assert cfg.rewards["stair_first_tread_settle_quality"].weight == 0.0
    assert MicroduckStairApexMantleRlCfg.max_iterations == 100
    assert MicroduckStairApexMantleRlCfg.save_interval == 25
    assert (
        MicroduckStairApexMantleRlCfg.actor.distribution_cfg["init_std"] == 0.35
    )
    assert MicroduckStairApexMantleRlCfg.algorithm.learning_rate == 3.0e-5


def test_roulade_bank_stage_requires_contact_supported_tread_progress():
    cfg = make_microduck_stair_roulade_bank_env_cfg()
    reset = cfg.events["route_state_curriculum"].params

    assert reset["lip_release_fraction"] == 0.10
    assert reset["shell_brace_fraction"] == 0.20
    assert reset["tread_recovery_fraction"] == 0.0
    assert reset["real_handoff_fraction"] == 0.70
    assert cfg.events["walker_state_bank"].params["bank_path"].endswith(
        "full170-roulade-state-bank.pt"
    )
    crossing = cfg.rewards["stair_assisted_crossing"]
    assert crossing.params["hard_height_gate"] is True
    assert crossing.params["clearance_height"] == 0.17
    assert math.isclose(crossing.params["corridor_half_width"], 0.36)
    support = cfg.rewards["stair_tread_support_frontier"]
    assert support.weight == 15.0
    assert support.params["riser_height"] == 0.17
    assert cfg.rewards["stair_first_riser_clearance"].weight == 400.0
    assert cfg.rewards["stair_first_tread_secured"].weight == 300.0
    assert MicroduckStairRouladeBankRlCfg.max_iterations == 150
    assert MicroduckStairRouladeBankRlCfg.save_interval == 25
    assert (
        MicroduckStairRouladeBankRlCfg.actor.distribution_cfg["init_std"] == 0.28
    )
    assert MicroduckStairRouladeBankRlCfg.algorithm.learning_rate == 2.0e-5


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
