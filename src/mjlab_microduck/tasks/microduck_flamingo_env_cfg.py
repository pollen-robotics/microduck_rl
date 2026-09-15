"""Microduck *flamingo* task — balance on the right foot.

Stage 1 (this file): the robot is spawned IN a validated one-foot pose (see
docs/superpowers/specs/2026-08-28-flamingo-design.md) with joint/tilt noise, an
all-zero 13-D command, and must hold single support — swing foot up, CoM over
the stance sole, trunk at the pose's lean — under training-style pushes.

Physics that shaped this env (measured 2026-08-28, notes/tools/duck_pose.py):
- With XL330 servos at firmware kP 200 the joint stiffness (~0.56 N·m/rad) is
  below gravity's (~1.07 N·m/rad about the ankle): NO pose is passively stable,
  not even STAND. Holding the flamingo pose is a control task, exactly like the
  standing policy. The pose is a geometric equilibrium (CoM 1.2 cm inside the
  right sole's contact hull) with every joint ≥ 3° from its range limit.
- Lateral CoM authority is the bottleneck (no ankle roll, hip roll ±22°), so the
  trunk leans ~24° over the stance foot; a plain "upright" reward would fight
  the pose — we reward the pose's own projected-gravity vector instead.
- Falling toward the swing (left) side is a soft failure (the free foot lands);
  falling toward the stance (right) side has no catch. The rewards are
  asymmetric accordingly (cheap swing-foot touchdown, steep stance-side tilt tax).

Plumbing (DR, obs noise/delays, IMU misalignment, encoder bias, BAM friction,
pushes) is copied from the standup env for sim2real parity.
"""

import math
from copy import deepcopy

# Symmetry: never for a one-sided task.
ENABLE_SYMMETRY = False

# ── Domain randomisation (matched to the velocity/standup envs) ───────────────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

COM_RANDOMIZATION_RANGE             = 0.003   # ramped to 0.015 (com_range curriculum)
HEAD_COM_RANDOMIZATION_RANGE        = 0.003   # ramped to 0.01
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
VELOCITY_PUSH_INTERVAL_S            = (3.0, 6.0)
VELOCITY_PUSH_RANGE                 = (-0.25, 0.25)   # final; ramped from 0 (push_magnitude curriculum)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

EPISODE_LENGTH_S = 6.0
NUM_STEPS_PER_ENV = 24

# ── The pose (notes/poses/flamingo_right.json, 3° joint-limit margin) ────────
# 14-servo order: 0-4 left leg (hip_yaw, hip_roll, hip_pitch, knee, ankle),
# 5-8 neck/head (neck_pitch, head_pitch, head_yaw, head_roll), 9-13 right leg.
FLAMINGO_POSE = [
    0.000,  0.300,  1.200, -0.800,  0.800,     # left leg (swing): lifted forward
    0.349,  0.350, -1.500,  0.000,             # neck/head: beak turned to the stance side
    0.386, -0.334,  0.258,  0.005, -0.253,     # right leg (stance): yawed, rolled, slight fwd lean
]
STANCE_SLOT = 1          # feet_ground_contact slot: 0 = left, 1 = right
SWING_SLOT  = 0
STANCE_SITE = "right_foot"
SWING_SITE  = "left_foot"
# Trunk placement with the right sole flat on the floor (measured in plain MuJoCo):
FLAMINGO_BASE_ROLL  = math.radians(22.6)
FLAMINGO_BASE_PITCH = math.radians(-8.9)
FLAMINGO_Z          = 0.120                         # trunk_base height, m
FLAMINGO_GRAVITY_B  = (-0.154, -0.379, -0.912)      # projected gravity in the trunk frame at the pose
# g_b,y at the pose is -0.38; rolling further toward the stance side (more negative)
# than -0.45 is the hard-failure direction.
STANCE_SIDE_TILT_THRESHOLD = 0.45
SWING_FOOT_TARGET_Z = 0.05   # m above the floor; the pose has it at ~0.17, 5 cm is "clearly lifted"

_LEG_JOINTS  = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_HEAD_JOINTS = [5, 6, 7, 8]

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
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    HEAD_BODY_NAMES,
    HEAD_POSE_CMD_RESAMPLE_S,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_flamingo_env_cfg(
    play: bool = False,
    standing_prob: float = 0.0,
) -> ManagerBasedRlEnvCfg:
    """Flamingo env. ``standing_prob`` > 0 adds a HOME/upright spawn bucket (stage 2)."""

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",  # LEFT first, RIGHT second
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
    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}   # all collisions: falls are physical
    cfg.scene.sensors  = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"
    cfg.episode_length_s = EPISODE_LENGTH_S

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards ───────────────────────────────────────────────────────────────
    for name in [
        "track_linear_velocity", "track_angular_velocity",
        "air_time", "foot_clearance", "foot_swing_height", "foot_slip",
        "pose", "upright", "dof_pos_limits",
        "head_pose_tracking", "body_pose_tracking", "head_pose_bias",
        "soft_landing", "hip_yaw_roll_deviation",
    ]:
        cfg.rewards.pop(name, None)

    stance_site = SceneEntityCfg("robot", site_names=[STANCE_SITE])
    swing_site  = SceneEntityCfg("robot", site_names=[SWING_SITE])

    # The balance signal: CoM over the stance sole. std 0.03 early (visible gradient
    # from the noisy spawns), tightened to 0.02 by the com_std curriculum.
    cfg.rewards["com_over_stance_foot"] = RewardTermCfg(
        func=microduck_mdp.com_over_foot, weight=3.0,
        params={"asset_cfg": stance_site, "std": 0.03},
    )
    # Pin the stance foot (anti-hop) and keep the swing foot clearly lifted.
    cfg.rewards["stance_foot_grounded"] = RewardTermCfg(
        func=microduck_mdp.foot_contact_reward, weight=1.0,
        params={"sensor_name": feet_ground_cfg.name, "slot": STANCE_SLOT},
    )
    cfg.rewards["swing_foot_clear"] = RewardTermCfg(
        func=microduck_mdp.foot_height_gaussian, weight=1.5,
        params={"asset_cfg": swing_site, "target": SWING_FOOT_TARGET_Z, "std": 0.03,
                "gate_sensor_name": feet_ground_cfg.name, "gate_slot": STANCE_SLOT},
    )
    # Soft failure: swing foot touchdown. Self-negating (≤ 0) → POSITIVE weight.
    cfg.rewards["swing_foot_touch"] = RewardTermCfg(
        func=microduck_mdp.foot_contact_penalty, weight=0.5,
        params={"sensor_name": feet_ground_cfg.name, "slot": SWING_SLOT},
    )
    # Stay near the equilibrium pose (generous std) — legs and head.
    cfg.rewards["pose_flamingo"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match, weight=1.5,
        params={"target_overrides": {i: v for i, v in enumerate(FLAMINGO_POSE)}, "std": 0.5},
    )
    # The trunk lean IS the pose: reward its projected-gravity vector, not "upright".
    cfg.rewards["gravity_flamingo"] = RewardTermCfg(
        func=microduck_mdp.projected_gravity_match, weight=2.0,
        params={"target": FLAMINGO_GRAVITY_B, "std": 0.15},
    )
    # Quiet hold (gated on stance contact).
    cfg.rewards["stillness"] = RewardTermCfg(
        func=microduck_mdp.joint_vel_gaussian, weight=1.0,
        params={"std": 2.0, "gate_sensor_name": feet_ground_cfg.name, "gate_slot": STANCE_SLOT},
    )
    # Hard-failure direction: rolling further toward the stance side than the pose.
    # Self-negating (≤ 0) → POSITIVE weight.
    cfg.rewards["stance_side_tilt"] = RewardTermCfg(
        func=microduck_mdp.lateral_tilt_penalty, weight=4.0,
        params={"threshold": STANCE_SIDE_TILT_THRESHOLD, "direction": -1.0},
    )
    # Off the joint limits (the pose is ≥ 0.05 rad from every limit; band 0.1).
    # Returns a POSITIVE cost → negative weight.
    cfg.rewards["joint_limit_proximity"] = RewardTermCfg(
        func=microduck_mdp.joint_pos_limit_proximity, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",)), "margin": 0.10},
    )

    # Regularisers (velocity/standup values). Motion-blockers stay light: balance
    # needs motion. Smoothness ramps in after discovery (curricula below).
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=0.0
    )
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost, weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # ── Observations (identical 61-D layout to every other policy) ────────────
    del cfg.observations["actor"].terms["base_lin_vel"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    for grp in ("actor", "critic"):
        cfg.observations[grp].terms.pop("foot_height", None)
        cfg.observations[grp].terms.pop("height_scan", None)
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
    for t in ("base_ang_vel", gravity_term_name):
        cfg.observations["actor"].terms[t].delay_min_lag = 0
        cfg.observations["actor"].terms[t].delay_max_lag = 1
        cfg.observations["actor"].terms[t].delay_update_period = 64
    cfg.observations["actor"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise       = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise       = Unoise(n_min=-0.25, n_max=0.25)

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

    # ── Commands: all-zero deployment command; slots kept alive with tiny ranges ─
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015)),
    )
    cfg.commands.pop("body_pose", None)
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "head_pose"},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    command.heading_command   = False
    command.ranges.heading    = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # ── Terminations: keep the base fell_over (70° trunk tilt; the pose is 24°) ─
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": ("feet_ground_contact",)},
    )

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields, mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history, mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    # Spawn IN the pose (runs after reset_base, which it overrides).
    cfg.events["set_flamingo_state"] = EventTermCfg(
        func=microduck_mdp.set_flamingo_state,
        mode="reset",
        params={
            "joint_pose": FLAMINGO_POSE,
            "base_roll": FLAMINGO_BASE_ROLL,
            "base_pitch": FLAMINGO_BASE_PITCH,
            "z_min": FLAMINGO_Z,
            "z_max": FLAMINGO_Z + 0.01,
            "tilt_noise": math.radians(3.0),
            "joint_noise_std": 0.05,
            "standing_prob": standing_prob,
            "standing_z_min": 0.11,
            "standing_z_max": 0.12,
        },
    )

    if ENABLE_VELOCITY_PUSHES:
        interval = (0.5, 1.0) if play else VELOCITY_PUSH_INTERVAL_S
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},   # ramped by push_magnitude
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )
    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )
    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia, mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )
    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    # ── Terrain: flat ─────────────────────────────────────────────────────────
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Curricula (steps = iterations × 24) ───────────────────────────────────
    cfg.curriculum.pop("terrain_levels", None)
    cfg.curriculum.pop("command_vel", None)

    if ENABLE_VELOCITY_PUSHES:
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={
                "event_name": "push_robot",
                "push_stages": [
                    {"step": 0,                          "velocity_range": {"x": (0.0, 0.0),    "y": (0.0, 0.0)}},
                    {"step": 300 * NUM_STEPS_PER_ENV,    "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
                    {"step": 600 * NUM_STEPS_PER_ENV,    "velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}},
                    {"step": 1000 * NUM_STEPS_PER_ENV,   "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}},
                ],
            },
        )
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0,                        "weight": -0.1},
                {"step": 400 * NUM_STEPS_PER_ENV,  "weight": -0.5},
                {"step": 800 * NUM_STEPS_PER_ENV,  "weight": -1.0},
            ],
        },
    )
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0,                        "weight": 0.0},
                {"step": 600 * NUM_STEPS_PER_ENV,  "weight": -1e-3},
            ],
        },
    )
    cfg.curriculum["com_std"] = CurriculumTermCfg(
        func=microduck_mdp.reward_param_curriculum,
        params={
            "reward_name": "com_over_stance_foot",
            "param_stages": [
                {"step": 0,                        "params": {"std": 0.03}},
                {"step": 500 * NUM_STEPS_PER_ENV,  "params": {"std": 0.02}},
            ],
        },
    )
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,                        "range": 0.003},
                    {"step": 500 * NUM_STEPS_PER_ENV,  "range": 0.005},
                    {"step": 1000 * NUM_STEPS_PER_ENV, "range": 0.01},
                    {"step": 1500 * NUM_STEPS_PER_ENV, "range": 0.015},
                ],
            },
        )
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,                        "range": 0.003},
                    {"step": 500 * NUM_STEPS_PER_ENV,  "range": 0.005},
                    {"step": 1000 * NUM_STEPS_PER_ENV, "range": 0.01},
                ],
            },
        )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckFlamingoRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="flamingo",
    run_name="flamingo",
    save_interval=100,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=3_000,
)
