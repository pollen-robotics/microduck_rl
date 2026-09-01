"""Microduck jump (Elon hop) — two-foot hop, land standing.

Episodic hot-swap trick (same 61D contract as roulade / playdead / kick).
Deployment: swap ONNX in, duck hops, auto-swap back after ~2 s.

The Elon jump is a stiff two-foot hop (no arms on this robot; head-up during
flight is the duck analogue of the arms-up pose). Open-loop BAM probes
(2026-08-29) never left the ground — XL330s are at the margin — so the
reward is potential-based peak height + flight, with reverse-curriculum
airborne / crouched-upward spawns so landing is on-policy even if takeoff
is rare. Sitting/falling with feet off does not count (upright × z gates).

No per-step jackpot for being high or for holding a stand: peak and landing
are both Δ-progress.
"""

from __future__ import annotations

import math
from copy import deepcopy

ENABLE_SYMMETRY = True  # bilateral hop; mirror loss fights one-foot skips

ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True
ENABLE_KP_RANDOMIZATION = False
ENABLE_KD_RANDOMIZATION = False
ENABLE_MASS_INERTIA_RANDOMIZATION = True
ENABLE_JOINT_FRICTION_RANDOMIZATION = True
ENABLE_ARMATURE_RANDOMIZATION = True
ENABLE_VELOCITY_PUSHES = False  # a shove mid-hop is incoherent
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS = True

COM_RANDOMIZATION_RANGE = 0.003
HEAD_COM_RANDOMIZATION_RANGE = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ENCODER_BIAS_RANGE = (-0.015, 0.015)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

EPISODE_LENGTH_S = 3.5
STAND_Z = 0.115
# 3 cm is an ambitious hop on this robot; progress is capped here so a
# ballistic launch does not outscore a clean small hop.
JUMP_TARGET_ABOVE = 0.03
JUMP_TARGET_AIR_S = 0.12
JUMP_MIN_AIR_S = 0.04
JUMP_MIN_PEAK_ABOVE = 0.005
JUMP_MIN_UPRIGHT = 0.7
# subtree mass of trunk_base ≈ 0.737 kg
JUMP_ROBOT_WEIGHT = 0.737 * 9.81

# Standing squat still on the feet (not the sit keyframe). Servo-index keyed.
CROUCH_OVERRIDES = {
    2: -0.90,   # left  hip_pitch
    3: 0.80,    # left  knee
    4: 0.80,    # left  ankle
    5: 0.55,    # neck_pitch (beak slightly up)
    6: 0.55,    # head_pitch
    11: 0.90,   # right hip_pitch
    12: -0.80,  # right knee
    13: -0.80,  # right ankle
}

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def _trunk() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=("trunk_base",))


def make_microduck_jump_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Microduck two-foot-jump environment configuration."""

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    cfg = make_velocity_env_cfg()
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"
    cfg.episode_length_s = EPISODE_LENGTH_S

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
        "upright",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    _jump_kw = dict(
        stand_z=STAND_Z,
        min_upright=JUMP_MIN_UPRIGHT,
        min_air_s=JUMP_MIN_AIR_S,
        min_peak_above=JUMP_MIN_PEAK_ABOVE,
    )

    # Main task signal: Δpeak height while hopping. 3 cm hop → 1.0 × weight.
    cfg.rewards["jump_progress"] = RewardTermCfg(
        func=microduck_mdp.jump_progress,
        weight=8.0,
        params={**_jump_kw, "target_above": JUMP_TARGET_ABOVE, "asset_cfg": _trunk()},
    )
    cfg.rewards["jump_flight"] = RewardTermCfg(
        func=microduck_mdp.jump_flight,
        weight=2.0,
        params={**_jump_kw, "target_air_s": JUMP_TARGET_AIR_S, "asset_cfg": _trunk()},
    )
    # Takeoff bootstrap — Δvz in contact. Weight stays below progress so a
    # polite hop still outscores a failed launch.
    cfg.rewards["jump_takeoff"] = RewardTermCfg(
        func=microduck_mdp.jump_takeoff_vz,
        weight=1.5,
        params={
            "stand_z": STAND_Z,
            "max_vz": 0.5,
            "min_z": 0.08,
            "min_upright": JUMP_MIN_UPRIGHT,
            "asset_cfg": _trunk(),
        },
    )
    cfg.rewards["jump_unloading"] = RewardTermCfg(
        func=microduck_mdp.jump_unloading,
        weight=1.0,
        params={
            "sensor_name": feet_ground_cfg.name,
            "robot_weight": JUMP_ROBOT_WEIGHT,
            "stand_z": STAND_Z,
            "min_upright": JUMP_MIN_UPRIGHT,
            "asset_cfg": _trunk(),
        },
    )
    # Landing recovery after latch. Potential-based — bounded ~1 total.
    cfg.rewards["jump_landing"] = RewardTermCfg(
        func=microduck_mdp.jump_landing,
        weight=4.0,
        params={
            **_jump_kw,
            "height_std": 0.04,
            "upright_std": 0.40,
            "asset_cfg": _trunk(),
        },
    )
    # Always-on upright (the hop is a stiff Elon pose, not a flip).
    cfg.rewards["upright_linear"] = RewardTermCfg(
        func=microduck_mdp.body_upright_linear,
        weight=1.0,
        params={"asset_cfg": _trunk()},
    )
    cfg.rewards["jump_head_up"] = RewardTermCfg(
        func=microduck_mdp.jump_head_up,
        weight=0.5,
        params={**_jump_kw, "asset_cfg": _trunk()},
    )

    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.01
    cfg.rewards["angular_momentum"].weight = -0.005
    cfg.rewards.pop("soft_landing", None)
    # Self-negating |a_z| → POSITIVE weight.
    cfg.rewards["gentle_landing"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.002,
        params={"asset_cfg": _trunk()},
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # ── Observations (61D: 48 proprio + twist3 + head4 + body6 pad) ────────
    del cfg.observations["actor"].terms["base_lin_vel"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": ("feet_ground_contact",)},
    )

    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        "left_foot_collision",
        "right_foot_collision",
    )
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)
    if "push_robot" in cfg.events:
        del cfg.events["push_robot"]

    cfg.events["set_jump_state"] = EventTermCfg(
        func=microduck_mdp.reset_jump_state,
        mode="reset",
        params={
            "standing_prob": 0.60,
            "crouch_prob": 0.25,
            "air_prob": 0.15,
            "standing_z_min": 0.11,
            "standing_z_max": 0.12,
            "standing_tilt_max": math.radians(5.0),
            "crouch_z_min": 0.090,
            "crouch_z_max": 0.105,
            "crouch_vz_range": (0.25, 0.55),
            "crouch_overrides": CROUCH_OVERRIDES,
            "crouch_factor_range": (0.5, 1.0),
            "air_z_min": 0.13,
            "air_z_max": 0.16,
            "air_vz_range": (0.0, 0.35),
            "joint_noise_std": 0.05,
            "stand_z": STAND_Z,
            "min_air_s": JUMP_MIN_AIR_S,
        },
    )

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )
    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )
    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )
    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    if "terrain_levels" in cfg.curriculum:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # Shift toward standing starts once landing is in the data. Never drop
    # reverse-curriculum to zero — takeoff is the scarce resource.
    cfg.curriculum["jump_spawn_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_jump_state",
            "param_stages": [
                {"step": 0, "params": {"standing_prob": 0.60, "crouch_prob": 0.25, "air_prob": 0.15}},
                {"step": 800 * 24, "params": {"standing_prob": 0.70, "crouch_prob": 0.20, "air_prob": 0.10}},
                {"step": 1800 * 24, "params": {"standing_prob": 0.80, "crouch_prob": 0.15, "air_prob": 0.05}},
            ],
        },
    )
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.05},
                {"step": 800 * 24, "weight": -0.15},
                {"step": 1600 * 24, "weight": -0.30},
            ],
        },
    )
    cfg.curriculum["gentle_landing_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "gentle_landing",
            "weight_stages": [
                {"step": 0, "weight": 0.002},
                {"step": 1200 * 24, "weight": 0.005},
            ],
        },
    )
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0, "range": 0.003},
                    {"step": 500 * 24, "range": 0.008},
                    {"step": 1000 * 24, "range": 0.015},
                ],
            },
        )

    if play:
        p = cfg.events["set_jump_state"].params
        p["standing_prob"] = 0.5
        p["crouch_prob"] = 0.25
        p["air_prob"] = 0.25
        cfg.curriculum.pop("jump_spawn_mix", None)
    return cfg


MicroduckJumpRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_jump",
    run_name="microduck_jump",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=3_000,
)
