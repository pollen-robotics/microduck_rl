"""Microduck BallWalk task — circus-style walking on top of a size-7 basketball.

The robot spawns standing on top of a free-rolling size-7 basketball (24cm /
0.62kg, basketball.xml) and must keep its balance while tracking twist commands: the
only way the base translates sustainably is by walking the ball into rolling,
treadmill-style. Zero-command envs train balance-in-place — the base skill.

Key design decisions:
  - The actor is BLIND to the ball (unified 61D obs layout, no ball terms):
    the real robot has no ball sensing — balance must come from IMU +
    proprioception, like the rest of the policy family. The CRITIC sees ball
    pos/vel (asymmetric actor-critic) to predict falls.
  - "Stay on the ball" is enforced by hard state, not reward nudges: falling
    off drops the root below MIN_ROOT_Z (ball top is at 0.24 m; standing on
    the FLOOR is 0.115 m) → termination. No on-ball gate on the positive
    stack is needed beyond that — every positive term is only collectable
    while riding.
  - ball_balance (Gaussian on the horizontal root ↔ ball-center offset) is
    the core balance shaping: on a ball, being off-center IS falling.
    Verified in sim (2026-09-08 pre-check): from HOME on the basketball the
    robot sits at root_z ≈ 0.35 and passively tips in ~0.5 s (21° by 0.5 s;
    the 60 cm ball it replaced took ~1 s) — a fast but still recoverable
    instability the policy must learn to stabilize.
  - Commands start near-zero (balance first) and widen via curriculum; pushes
    ramp in late and stay small (±0.1 — a shove on a ball is worth ~3× one on
    the ground).
  - The feet contact sensor is aimed at the BALL, not the terrain, so the
    stock gait terms (air_time) and nan_state guard bind to the surface the
    robot actually walks on. Terrain-height-sensor terms (foot_clearance /
    foot_swing_height / foot_slip, foot_height obs) are dropped — the ray
    sensor reads the floor, which is ~0.6 m below the feet here.

DR / noise / regularization: velocity-parity (same recipe as ball_kick /
standup — the stack with proven transfer), plus ball-specific DR: ball
sliding friction and ±20% ball mass+inertia.
"""

import math
from copy import deepcopy

# ── Symmetry — OFF (walking on a ball has no enforced gait symmetry) ─────────
ENABLE_SYMMETRY = False

# ── Domain randomisation (matched to velocity / standup) ─────────────────────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False
ENABLE_KD_RANDOMIZATION              = False
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True
ENABLE_BALL_DR                       = True

# ── Ranges (matched to velocity / standup) ───────────────────────────────────
COM_RANDOMIZATION_RANGE             = 0.003           # ramped to 0.015 via curriculum
HEAD_COM_RANDOMIZATION_RANGE        = 0.003           # ramped to 0.01 via curriculum
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)    # unused (kp DR off)
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)      # unused (kd DR off)
VELOCITY_PUSH_INTERVAL_S            = (3.0, 6.0)
VELOCITY_PUSH_RANGE                 = (-0.1, 0.1)     # ball-scale: ±0.3 on the ground ≈ un-recoverable up here
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# ── Ball DR ───────────────────────────────────────────────────────────────────
BALL_FRICTION_RANGE      = (0.6, 1.2)   # sliding friction (abs) on the ball geom
BALL_MASS_SCALE_RANGE    = (0.8, 1.2)   # ±20% mass+inertia together (pseudo_inertia)

# ── Task constants ────────────────────────────────────────────────────────────
BALL_RADIUS = 0.12  # size-7 basketball (basketball.xml: r=0.12 m, 0.62 kg, hollow-shell inertia)
BALL_TOP_Z = 2 * BALL_RADIUS
# Trunk height when standing on the ball, MEASURED in the 2026-09-08 physics
# pre-check (root_z from HOME held on the free basketball, tilt < 4°): flat-ground
# STAND_Z 0.115 minus ~5mm of sphere-curvature drop under the feet (a 12 cm
# sphere drops ~7mm under feet spread ±4 cm). Geometric estimate 0.348, measured 0.350.
TRUNK_ON_BALL_Z = 0.35
# Spawn root z (feet just kissing the ball top; same +5..15mm drop margin as
# velocity's flat-ground (0.12, 0.13) spawn).
SPAWN_Z_RANGE = (0.355, 0.365)
# Spawn XY noise around the ball apex — forces an immediate balance correction.
SPAWN_XY_NOISE = 0.02
# Fell-off-the-ball termination: root below this = not riding anymore. Standing
# on the ball is ~0.35, the deepest imaginable crouch ON the ball is ~0.29
# (SIT-pose knees on the apex); root just below the ball TOP (0.24) is
# unambiguous. Standing on the floor is 0.115.
MIN_ROOT_Z = 0.23

EPISODE_LENGTH_S = 20.0

_NECK_JOINTS = [5, 6, 7, 8]

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
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_BASKETBALL_CFG,
    MICRODUCK_STANDUP_ROBOT_CFG,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_ball_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the Microduck BallWalk environment configuration (flat only)."""

    # Gait-conditioned leg-pose stds — velocity's exact values.
    std_standing = {
        r".*hip_yaw.*": 0.1,
        r".*hip_roll.*": 0.05,
        r".*hip_pitch.*": 0.15,
        r".*knee.*": 0.15,
        r".*ankle.*": 0.1,
    }
    std_walking = {
        r".*hip_yaw.*": 0.3,
        r".*hip_roll.*": 0.05,
        r".*hip_pitch.*": 0.4,
        r".*knee.*": 0.4,
        r".*ankle.*": 0.25,
    }

    # Feet-BALL contact sensor. Named "feet_ground_contact" so the base
    # template's gait terms (air_time) and obs bind unchanged — the ball IS
    # the ground of this task.
    feet_ball_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="geom", pattern="ball_geom", entity="ball"),
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

    # Full-collision robot: a fall on the ball can contact any body part.
    # Robot MUST stay the first entity (base reset events write robot root
    # state at qpos[:, 0:7]).
    cfg.scene.entities = {
        "robot": MICRODUCK_STANDUP_ROBOT_CFG,
        "ball":  MICRODUCK_BASKETBALL_CFG,
    }
    cfg.scene.sensors = (feet_ball_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

    # Contact headroom: full-collision robot pile-ups on ball + floor.
    cfg.sim.nconmax = 50

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards ───────────────────────────────────────────────────────────────
    # Drop terrain-height-sensor terms (the ray sensor reads the floor 0.6m
    # below the feet) and soft_landing (velocity drops it too).
    for name in ["foot_clearance", "foot_swing_height", "foot_slip", "soft_landing"]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # Velocity tracking — the "make the ball move" objective: the base can
    # only hold a commanded velocity by rolling the ball under itself.
    cfg.rewards["track_linear_velocity"].weight = 2.0
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.1)
    cfg.rewards["track_angular_velocity"].weight = 2.0
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.5)

    # Core balance shaping: stay vertically above the ball center.
    cfg.rewards["ball_balance"] = RewardTermCfg(
        func=microduck_mdp.ball_balance_gaussian,
        weight=2.0,
        params={"std": 0.05, "asset_name": "ball"},
    )

    # Upright — velocity's exact recipe.
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)

    # Trunk at riding height: discourages crouch-perching / draping on the
    # ball. Absolute z is exact here — flat floor, constant ball radius.
    cfg.rewards["height_stand"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=1.0,
        params={
            "std":           0.04,
            "target_height": TRUNK_ON_BALL_Z,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Gait: air_time against the feet-BALL sensor (walking on the ball needs
    # real steps; gated off at zero command like velocity).
    cfg.rewards["air_time"].weight = 3.0
    cfg.rewards["air_time"].params["command_threshold"] = 0.01
    cfg.rewards["air_time"].params["threshold_min"] = 0.125
    cfg.rewards["air_time"].params["threshold_max"] = 0.300

    # Leg pose (gait-conditioned stds), legs only — velocity's recipe.
    cfg.rewards["pose"].params["std_standing"] = std_standing
    cfg.rewards["pose"].params["std_walking"] = std_walking
    cfg.rewards["pose"].params["std_running"] = std_walking
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",)
    )
    cfg.rewards["pose"].params["walking_threshold"] = 0.01
    cfg.rewards["pose"].weight = 1.0

    # Neck/head at HOME (no head command in this task — the head is 38% of the
    # robot's mass; let the policy use it freely for balance? No: an
    # uncontrolled flailing head does not transfer. Loose std keeps small
    # counterweight moves affordable while pinning the DC posture).
    cfg.rewards["pose_neck"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=1.0,
        params={
            "std": 0.3,
            "joint_indices": _NECK_JOINTS,
            "target_overrides": None,
        },
    )

    # Sim2real regularisers — velocity parity. Motion-blockers stay LOW:
    # balance on a ball is a permanently dynamic task.
    cfg.rewards["action_rate_l2"].weight = -0.1  # ramped by curriculum below
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # ── Observations (unified 61D actor layout, ball-blind) ───────────────────
    del cfg.observations["actor"].terms["base_lin_vel"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    # No terrain-height sensor in this env — drop the sensor-backed terms.
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    # NaN-safe critic sensor terms (see velocity env rationale).
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

    # IMU obs delay — velocity's 2026-07 audit values.
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    # Obs noise — matched to the velocity env.
    cfg.observations["actor"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise       = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise       = Unoise(n_min=-0.25, n_max=0.25)

    # IMU mounting-misalignment DR (obs-level, actor only).
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # 1-ctrl-step lag on joint_vel (Dynamixel moving-average, see velocity env).
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    # Deepcopy joint_pos/joint_vel per group so the encoder-bias `biased` flag
    # applies to the actor only.
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

    # Command obs slots — unified layout parity: [twist(3), head(4), body(6)],
    # head/body zero-padded (no head/body pose control in this task).
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # CRITIC-ONLY ball state (asymmetric actor-critic): the actor stays blind
    # to the ball (no ball sensing on the real robot); the critic uses it to
    # anticipate falls and the ball's momentum.
    cfg.observations["critic"].terms["ball_position"] = ObservationTermCfg(
        func=microduck_mdp.ball_pos_in_base, params={"asset_name": "ball"},
    )
    cfg.observations["critic"].terms["ball_velocity"] = ObservationTermCfg(
        func=microduck_mdp.ball_vel_in_base, params={"asset_name": "ball"},
    )

    # ── Commands ──────────────────────────────────────────────────────────────
    # Twist = desired BASE velocity (== ball translation when riding). Starts
    # near-zero (balance is the base skill); the curriculum below widens it.
    # 40% of envs get the exact-zero command — balance-in-place is both the
    # hardest slice and the deployment idle state.
    command: UniformVelocityCommandCfg = deepcopy(cfg.commands["twist"])
    command.rel_standing_envs = 0.4
    command.rel_heading_envs  = 0.0
    command.heading_command   = False
    command.ranges.heading    = None
    command.resampling_time_range = (4.0, 8.0)
    command.ranges.lin_vel_x = (-0.05, 0.05)
    command.ranges.lin_vel_y = (-0.05, 0.05)
    command.ranges.ang_vel_z = (-0.1, 0.1)
    command.viz.z_offset = 0.5
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # ── Terminations ──────────────────────────────────────────────────────────
    # fell_over (base, 70°) kept. Fell-off-the-ball: root below MIN_ROOT_Z —
    # a hard state gate, no reward can be farmed off the ball.
    cfg.terminations["fell_off_ball"] = TerminationTermCfg(
        func=microduck_mdp.root_height_below,
        params={"min_height": MIN_ROOT_Z},
    )
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": (feet_ball_cfg.name,)},
    )

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    # Foot sliding friction DR. Feet have priority=1 (FULL_COLLISION), so this
    # range governs the FOOT-BALL contact (the ball geom's friction only sets
    # ball-floor grip). Widened above velocity's (0.7, 1.3): the real PU sole on
    # a pebbled rubber basketball is rubber-on-rubber (~1.5-2.5), and the
    # 2026-09-08 rollouts showed the policy barely lifting its feet — expecting
    # slip that the real setup won't have. Low end kept so the sim default still
    # falls in-distribution.
    cfg.events["foot_friction"].params["ranges"] = (0.7, 2.0)

    # Ball reset MUST run before the robot spawn conceptually pairs with it —
    # both center on the env origin. Empty pose_range = exactly at the origin,
    # resting on the floor (default_root_state z = ball radius), zero velocity.
    cfg.events["reset_ball"] = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("ball"),
        },
    )

    # Robot spawn: on the ball apex, small XY offset (immediate balance
    # correction from step 0), random yaw. z measured in the physics pre-check.
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (-SPAWN_XY_NOISE, SPAWN_XY_NOISE),
        "y": (-SPAWN_XY_NOISE, SPAWN_XY_NOISE),
        "z": SPAWN_Z_RANGE,
        "yaw": (-math.pi, math.pi),
    }

    # Joint noise on the standing start (velocity parity).
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)

    if ENABLE_VELOCITY_PUSHES:
        interval = (0.5, 1.0) if play else VELOCITY_PUSH_INTERVAL_S
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                "velocity_range": {
                    "x": VELOCITY_PUSH_RANGE,
                    "y": VELOCITY_PUSH_RANGE,
                },
                "asset_cfg": SceneEntityCfg("robot"),
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

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
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

    if ENABLE_BALL_DR:
        # Ball sliding friction (foot grip on the ball is governed by the FOOT
        # geom — priority 1 — so this mostly randomizes ball-floor rolling).
        cfg.events["ball_friction"] = EventTermCfg(
            func=dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("ball", geom_names=("ball_geom",)),
                "operation": "abs",
                "ranges": BALL_FRICTION_RANGE,
                "shared_random": True,
            },
        )
        # Ball mass+inertia ±20% (the real prop's weight won't match the sim).
        _b_lo, _b_hi = BALL_MASS_SCALE_RANGE
        cfg.events["randomize_ball_mass"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("ball", body_names=("ball",)),
                "alpha_range": (math.log(_b_lo) / 2.0, math.log(_b_hi) / 2.0),
            },
        )

    # ── Terrain: flat only ────────────────────────────────────────────────────
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Curriculum ────────────────────────────────────────────────────────────
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # action_rate ramp — velocity's stages, stretched (skill discovery on a
    # ball is slower; an attempt-tax too early makes "stand rigid" win).
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.1},
                {"step": 1000 * 24,  "weight": -0.2},
                {"step": 1500 * 24,  "weight": -0.4},
                {"step": 2000 * 24,  "weight": -0.6},
                {"step": 2500 * 24,  "weight": -0.8},
                {"step": 3000 * 24,  "weight": -1.0},
            ],
        },
    )

    # Command ranges: balance first, then walk the ball. Stages small — this
    # robot tops out at 0.4 m/s on the GROUND; on a ball, 0.15 is plenty.
    cfg.curriculum["command_ranges"] = CurriculumTermCfg(
        func=microduck_mdp.velocity_command_ranges_curriculum,
        params={
            "command_name": "twist",
            "velocity_stages": [
                {"step": 0,          "lin_vel_range": 0.05, "ang_vel_range": 0.1},
                {"step": 1000 * 24,  "lin_vel_range": 0.10, "ang_vel_range": 0.3},
                {"step": 2000 * 24,  "lin_vel_range": 0.15, "ang_vel_range": 0.5},
            ],
        },
    )

    if ENABLE_VELOCITY_PUSHES:
        # Pushes ramp in AFTER balance exists — a shove at iter 0 while the
        # policy can't even stand on the ball taxes the discovery itself.
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={
                "event_name": "push_robot",
                "push_stages": [
                    {"step": 0,          "velocity_range": {"x": (0.0, 0.0),    "y": (0.0, 0.0)}},
                    {"step": 1500 * 24,  "velocity_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)}},
                    {"step": 2500 * 24,  "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}},
                ],
            },
        )

    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,          "range": 0.003},
                    {"step": 1000 * 24,  "range": 0.005},
                    {"step": 2000 * 24,  "range": 0.01},
                    {"step": 3000 * 24,  "range": 0.015},
                ],
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,          "range": 0.003},
                    {"step": 1000 * 24,  "range": 0.005},
                    {"step": 2000 * 24,  "range": 0.01},
                ],
            },
        )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckBallWalkRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normalizer MUST be baked into ONNX by export.py
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
    experiment_name="ball_walk",
    run_name="ball_walk",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=10_000,
)
