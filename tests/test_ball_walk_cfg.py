"""Cfg-invariant tests for the BallWalk env (CPU, no GPU needed)."""

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_ball_walk_env_cfg import (
    BALL_RADIUS,
    MIN_ROOT_Z,
    SPAWN_Z_RANGE,
    TRUNK_ON_BALL_Z,
    make_microduck_ball_walk_env_cfg,
)


def test_ball_walk_scene_has_ball_entity():
    cfg = make_microduck_ball_walk_env_cfg()
    assert list(cfg.scene.entities.keys())[0] == "robot"  # robot MUST stay first
    assert "ball" in cfg.scene.entities
    # Ball default z = radius so the empty-pose-range reset seats it on the floor.
    assert cfg.scene.entities["ball"].init_state.pos == (0.0, 0.0, BALL_RADIUS)


def test_ball_walk_spawn_geometry():
    """Spawn/termination/height constants must stay mutually consistent."""
    ball_top = 2 * BALL_RADIUS
    # Robot spawns just above the ball apex.
    assert ball_top < SPAWN_Z_RANGE[0] < SPAWN_Z_RANGE[1] < ball_top + 0.15
    # Riding trunk height sits between ball top and spawn.
    assert ball_top < TRUNK_ON_BALL_Z < SPAWN_Z_RANGE[0]
    # Termination threshold: above floor-standing (0.115), below any on-ball pose.
    assert 0.2 < MIN_ROOT_Z < ball_top

    cfg = make_microduck_ball_walk_env_cfg()
    assert cfg.events["reset_base"].params["pose_range"]["z"] == SPAWN_Z_RANGE
    assert cfg.terminations["fell_off_ball"].params["min_height"] == MIN_ROOT_Z
    assert (
        cfg.rewards["height_stand"].params["target_height"] == TRUNK_ON_BALL_Z
    )


def test_ball_walk_feet_sensor_targets_ball():
    cfg = make_microduck_ball_walk_env_cfg()
    feet = next(s for s in cfg.scene.sensors if s.name == "feet_ground_contact")
    assert feet.secondary.entity == "ball"
    assert feet.secondary.pattern == "ball_geom"
    # nan_state guards the same sensor.
    assert cfg.terminations["nan_state"].params["sensor_names"] == (
        "feet_ground_contact",
    )


def test_ball_walk_reward_signs():
    """Penalty terms must be ≤ 0-producing: mjlab cost funcs get negative
    weights; every positive term here is a Gaussian/tracking reward."""
    cfg = make_microduck_ball_walk_env_cfg()
    r = cfg.rewards
    for name in ["action_rate_l2", "body_ang_vel", "angular_momentum", "self_collisions"]:
        assert r[name].weight <= 0.0, name
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "ball_balance",
        "upright",
        "height_stand",
        "air_time",
        "pose",
        "pose_neck",
    ]:
        assert r[name].weight > 0.0, name
    # Terrain-height-sensor terms must be gone (no ray sensor in this scene).
    for name in ["foot_clearance", "foot_swing_height", "foot_slip", "soft_landing"]:
        assert name not in r, name
    assert r["ball_balance"].func is microduck_mdp.ball_balance_gaussian


def test_ball_walk_obs_layout_61d():
    """Unified actor layout: 48 proprio + twist(3) + head(4) + body(6)."""
    cfg = make_microduck_ball_walk_env_cfg()
    actor_terms = list(cfg.observations["actor"].terms.keys())
    # Command block order matters for the runtime hot-swap contract.
    assert actor_terms[-2:] == ["head_command", "body_command"]
    assert cfg.observations["actor"].terms["head_command"].params["dim"] == 4
    assert cfg.observations["actor"].terms["body_command"].params["dim"] == 6
    # Actor is ball-blind; critic sees the ball.
    assert "ball_position" not in cfg.observations["actor"].terms
    assert "ball_position" in cfg.observations["critic"].terms
    assert "ball_velocity" in cfg.observations["critic"].terms
    assert "base_lin_vel" not in cfg.observations["actor"].terms
    assert "base_lin_vel" in cfg.observations["critic"].terms
    # No terrain-height obs anywhere.
    for grp in ("actor", "critic"):
        assert "height_scan" not in cfg.observations[grp].terms
    assert "foot_height" not in cfg.observations["critic"].terms


def test_ball_walk_commands_start_small():
    cfg = make_microduck_ball_walk_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.rel_standing_envs == 0.4  # explicit zero-command bucket
    assert cmd.heading_command is False
    assert max(abs(v) for v in cmd.ranges.lin_vel_x) <= 0.05
    assert max(abs(v) for v in cmd.ranges.ang_vel_z) <= 0.1
    # Curriculum widens, never past ball-scale speeds.
    stages = cfg.curriculum["command_ranges"].params["velocity_stages"]
    assert stages[-1]["lin_vel_range"] <= 0.2


def test_ball_walk_dr_stack_present():
    cfg = make_microduck_ball_walk_env_cfg()
    for name in [
        "expand_bam_friction_fields",  # mandatory for any standalone BAM env
        "reset_action_history",
        "reset_ball",
        "randomize_com",
        "randomize_head_com",
        "randomize_joint_friction",
        "randomize_armature",
        "encoder_bias",
        "ball_friction",
        "randomize_ball_mass",
    ]:
        assert name in cfg.events, name
    # Ball resets to the env origin with zero velocity.
    assert cfg.events["reset_ball"].params["pose_range"] == {}
    assert cfg.events["reset_ball"].params["asset_cfg"].name == "ball"
    # Pushes start at zero and ramp in via curriculum.
    stage0 = cfg.curriculum["push_magnitude"].params["push_stages"][0]
    assert stage0["velocity_range"]["x"] == (0.0, 0.0)


def test_ball_walk_play_variant_builds():
    cfg = make_microduck_ball_walk_env_cfg(play=True)
    assert "ball_balance" in cfg.rewards


def test_basketball_xml_is_a_realistic_size_7():
    """FIBA size 7: circumference 749-780 mm, mass 567-650 g. A basketball is a
    thin hollow shell → I = (2/3) m r². Guards against MuJoCo inferring a SOLID
    sphere inertia (2/5 m r², 40% too low) from a geom mass."""
    import math

    import mujoco

    from mjlab_microduck.robot.microduck_constants import MICRODUCK_BASKETBALL_XML

    m = mujoco.MjModel.from_xml_path(str(MICRODUCK_BASKETBALL_XML))
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    r = float(m.geom_size[g, 0])
    assert r == BALL_RADIUS
    assert 0.749 <= 2 * math.pi * r <= 0.780
    mass = float(m.body_mass[b])
    assert 0.567 <= mass <= 0.650
    shell = (2.0 / 3.0) * mass * r * r
    for i in range(3):
        assert abs(m.body_inertia[b, i] - shell) / shell < 0.02
