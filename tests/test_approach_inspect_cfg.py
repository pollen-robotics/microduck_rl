"""Cfg invariants for Mjlab-ApproachInspect-*-MicroDuck (CPU, no env build).

Locks in, per AGENTS.md "Building a new env" step 4:
  * the target scene entity exists and the robot stays entity #1
  * the twist command is the state-driven TargetCommand
  * every DR / sim2real block is the velocity env's, term for term
  * reward sign convention: self-negating penalties POSITIVE, mjlab costs NEGATIVE
  * the 61-D command slots are all still present (head/body kept alive at weight 0)
  * gait terms are gated OUTSIDE the ring; reset_target runs after reset_base
"""
import dataclasses

import pytest

from mjlab_microduck.robot.microduck_constants import MICRODUCK_TARGET_CFG
from mjlab_microduck.tasks import approach_inspect_mdp as ai_mdp
from mjlab_microduck.tasks.microduck_approach_inspect_env_cfg import (
    GAIT_CMD_THRESHOLD,
    RING_TOL_M,
    STANDOFF_M,
    make_microduck_approach_inspect_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)

# Everything sim2real in the velocity env that must be byte-identical here.
DR_EVENTS = (
    "expand_bam_friction_fields", "reset_action_history", "foot_friction",
    "push_robot", "randomize_com", "randomize_head_com", "randomize_mass_inertia",
    "randomize_joint_friction", "randomize_armature", "encoder_bias",
)
DR_CURRICULA = ("com_range", "head_com_range")
NOISY_ACTOR_TERMS = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel")


def _plain(obj):
    """Dataclass → comparable nested structure (functions compared by identity)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_plain(v) for v in obj)
    return obj


@pytest.fixture(scope="module")
def cfg():
    return make_microduck_approach_inspect_env_cfg()


@pytest.fixture(scope="module")
def vel():
    return make_microduck_velocity_env_cfg()


def test_target_entity_and_robot_first(cfg):
    names = list(cfg.scene.entities)
    assert names[0] == "robot"
    assert cfg.scene.entities["target"] is MICRODUCK_TARGET_CFG


def test_command_is_state_driven_target(cfg):
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, ai_mdp.TargetCommandCfg)
    assert cmd.class_type is ai_mdp.TargetCommand
    assert cmd.target_name == "target"
    assert cmd.rel_standing_envs == 0.0 and cmd.rel_turn_in_place_envs == 0.0
    # The other two command slots stay (61-D layout), tiny ranges, alive.
    assert "head_pose" in cfg.commands and "body_pose" in cfg.commands
    for grp in ("actor", "critic"):
        terms = cfg.observations[grp].terms
        assert "twist" in terms or "velocity_command" in terms or any(
            getattr(t, "params", {}).get("command_name") == "twist" for t in terms.values()
        )
        assert "head_command" in terms and "body_command" in terms
    assert cfg.rewards["head_pose_tracking"].weight == 0.0
    assert "head_pose_bias" not in cfg.rewards
    assert "head_pose_range" not in cfg.curriculum


def test_dr_blocks_identical_to_velocity(cfg, vel):
    for name in DR_EVENTS:
        assert name in cfg.events, name
        assert _plain(cfg.events[name]) == _plain(vel.events[name]), name
    for name in DR_CURRICULA:
        assert _plain(cfg.curriculum[name]) == _plain(vel.curriculum[name]), name
    # Obs noise / delays / IMU misalignment / encoder bias live on the actor terms.
    for name in NOISY_ACTOR_TERMS:
        a, b = cfg.observations["actor"].terms[name], vel.observations["actor"].terms[name]
        assert _plain(a) == _plain(b), name
    assert cfg.terminations["nan_state"].func is vel.terminations["nan_state"].func
    assert cfg.actions["joint_pos"].scale == vel.actions["joint_pos"].scale


def test_reward_signs(cfg):
    r = cfg.rewards
    # velocity-tracking gone
    assert "track_linear_velocity" not in r and "track_angular_velocity" not in r
    # self-negating microduck penalties (return <= 0) take POSITIVE weights
    assert r["head_aim_bias"].func is ai_mdp.head_bearing_ema_l1_penalty
    assert r["head_aim_bias"].weight > 0
    assert r["stop_in_ring"].func is ai_mdp.planar_speed_gated_penalty
    assert r["stop_in_ring"].weight > 0
    # potentials / gated positives
    assert r["ring_progress"].weight > 0 and r["head_aim"].weight > 0
    assert r["dwell"].weight > 0 and r["turn_away"].weight > 0
    # mjlab-base costs (>= 0) take NEGATIVE weights
    for name in ("action_rate_l2", "body_ang_vel", "angular_momentum", "self_collisions", "foot_slip"):
        assert r[name].weight < 0, name
    assert r["neck_action_rate_l2"].weight <= 0
    # dwell must be computed before turn_away (turn_away reads env._ai_dwell)
    names = list(r)
    assert names.index("dwell") < names.index("turn_away")


def test_task_params_consistent(cfg):
    r = cfg.rewards
    for name in ("ring_progress", "head_aim", "dwell", "stop_in_ring"):
        assert r[name].params["standoff"] == STANDOFF_M, name
    for name in ("head_aim", "dwell", "stop_in_ring"):
        assert r[name].params["gate_tol"] == RING_TOL_M, name
    assert r["dwell"].params["speed_tol"] > 0          # orbiting accrues no dwell
    # head_cfg must be passed in params (the manager resolves only those) and
    # name the mouth_tip site + jaw_soft body that exist on the walk model.
    for name in ("head_aim", "head_aim_bias", "dwell"):
        hc = r[name].params["head_cfg"]
        assert hc.name == "robot" and list(hc.site_names) == ["mouth_tip"]
        assert list(hc.body_names) == ["jaw_soft"]
    assert r["turn_away"].params["require_dwell"] >= 0.99


def test_gait_terms_gated_outside_ring(cfg):
    # command norm = d / 1.5 ⇒ threshold must sit just past the ring's outer edge
    assert abs(GAIT_CMD_THRESHOLD - (STANDOFF_M + RING_TOL_M) / 1.5) < 1e-9
    for name in ("air_time", "foot_clearance", "foot_swing_height", "foot_slip"):
        assert cfg.rewards[name].params["command_threshold"] == GAIT_CMD_THRESHOLD, name
    assert cfg.rewards["pose"].params["walking_threshold"] == GAIT_CMD_THRESHOLD


def test_reset_target_after_reset_base(cfg):
    ev = cfg.events["reset_target"]
    assert ev.func is ai_mdp.reset_target_around_robot and ev.mode == "reset"
    assert ev.params["radius_range"] == (0.4, 1.5)
    order = list(cfg.events)
    assert order.index("reset_base") < order.index("reset_target")


def test_regularisers_ramp_from_low(cfg):
    stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert stages[0]["weight"] == cfg.rewards["action_rate_l2"].weight
    assert all(s["weight"] <= 0 for s in stages)
    assert stages[-1]["weight"] > -1.0     # < velocity's -1.0 (smaller task stack)
    nstages = cfg.curriculum["neck_action_rate_weight"].params["weight_stages"]
    assert nstages[0]["weight"] == 0.0 and nstages[-1]["weight"] < 0


def test_rough_and_play_variants_build():
    assert "ring_progress" in make_microduck_approach_inspect_env_cfg(rough=True).rewards
    assert "ring_progress" in make_microduck_approach_inspect_env_cfg(play=True).rewards
