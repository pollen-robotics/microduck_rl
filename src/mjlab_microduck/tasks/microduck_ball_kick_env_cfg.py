"""Microduck BallKick task — kick a ball forward with one foot (KICK_FOOT flag).

Episodic policy: the robot starts STANDING (HOME pose + noise) with a 70mm /
15g ball sitting just in front of its kicking foot (KICK_FOOT below — train a
right-footed and a left-footed policy as two separate runs). The goal is to
kick the ball forward (robot's heading at reset) at BALL_TARGET_SPEED while
keeping balance and staying robust to external pushes, then settle back into
a clean stand.

Key design decisions:
  - The policy is BLIND to the ball (no ball obs in the actor): the real robot
    has no ball sensing — the operator aims the robot at the ball. Robustness
    to placement error comes from ±2cm ball-position DR at reset instead. The
    CRITIC does see ball pos/vel (asymmetric actor-critic) so the value
    function can anticipate the kick payoff.
  - No phase command: the kick reward is available from t=0 and an earlier
    kick collects more ball-rolling reward, so the policy kicks immediately.
    At deployment: hard ONNX swap to this policy (à la jump/ground-pick), it
    kicks, then auto-swap back after ~2s.
  - Right-foot kick is enforced geometrically + economically: the ball spawns
    at the right toe, and an always-on LEFT-foot-grounded reward makes the
    left leg the support leg (lifting it costs reward every step; anti-hop).
  - Kick reward is LINEAR in ball forward speed (clamped at 5 m/s), not a
    saturating tanh — "as hard as possible" needs gradient at high speeds.
  - Obs layout is the unified 61D actor layout (twist + zero-padded head/body
    command slots) so the runtime can hard-swap ONNX files with one buffer.

DR / noise / regularization: velocity-parity, copied from the standup env
(which is itself matched to velocity — the recipe with proven transfer).
Task reward mass ~10 ≈ velocity's ~11, so the shared regularizer weights act
at the same relative strength.
"""

import math
from copy import deepcopy

# ── Kicking foot: "right" or "left" ───────────────────────────────────────────
# Flips the ball spawn side and the support-foot (anti-hop) sensor. Everything
# else is left/right symmetric (HOME pose has mirrored signs). Train the two
# policies as separate runs — wandb experiment/run name follows this flag.
KICK_FOOT = "right"
assert KICK_FOOT in ("right", "left")

# Symmetry — must stay OFF: the kick task is inherently one-footed.
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
VELOCITY_PUSH_RANGE                 = (-0.3, 0.3)     # ramped in via push curriculum
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# ── Task constants ────────────────────────────────────────────────────────────
# Long enough for kick + several seconds of ball-rolling reward + settle-back.
EPISODE_LENGTH_S = 5.0

# 70mm-diameter / 15g ball (see ball.xml).
BALL_RADIUS = 0.035
# Nominal ball-center offset in the robot's yaw frame. Measured at HOME: foot
# centers at (0, ±0.042), toe tip x≈0.034. With radius 0.035 and ±0.015 noise
# the ball's rear surface is at worst x=0.040 → always ≥6mm clear of the toe.
# (0.08 ± 0.02 allowed spawn-penetration with the toe: the solver ejected the
# ball at reset — free "kick" reward with no kick.)
# The lateral sign follows the kicking foot (right = -y, left = +y).
BALL_OFFSET_X     = 0.09
BALL_OFFSET_ABS_Y = 0.042
# Uniform ± placement noise per axis. This is the DR that makes the BLIND
# policy's swing robust to real-world aiming error.
BALL_POS_NOISE_XY = 0.015

# Target kick speed (m/s). The first trained policy (linear reward capped at
# 5 m/s) kicked much harder than needed — this tames the kick to a gentle,
# controlled tap. NOTE: the kick reward weights below are scaled to keep the
# at-target payoff ≈ +3/step regardless of this value (weight ≈ 3/target for
# the capped term) — if you change the target, rescale the weights with it.
BALL_TARGET_SPEED = 1.0

# Trunk standing height (measured natural equilibrium at HOME — see standup env).
STAND_Z = 0.115

_LEG_JOINTS  = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
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
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_BALL_CFG,
    MICRODUCK_STANDUP_ROBOT_CFG,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_ball_kick_env_cfg(
    play: bool = False,
    kick_foot: str | None = None,
) -> ManagerBasedRlEnvCfg:
    """Create the Microduck BallKick environment configuration.

    ``kick_foot`` overrides the module-level KICK_FOOT flag (used by tests);
    normal training just sets the flag at the top of this file.
    """
    kick_foot = kick_foot or KICK_FOOT
    assert kick_foot in ("right", "left")
    support_foot = "left" if kick_foot == "right" else "right"

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

    # Support-foot sensor: the non-kicking foot must stay planted through the kick.
    support_foot_ground_cfg = ContactSensorCfg(
        name="support_foot_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=rf"^{support_foot}_foot_collision$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
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

    # Full-collision robot (same spec as standup/ground-pick): the ball must be
    # able to contact the whole leg, not just the foot pads of the walk model.
    # Robot MUST stay the first entity (set_random_ground_state and the base
    # reset events write robot root state at qpos[:, 0:7]).
    cfg.scene.entities = {
        "robot": MICRODUCK_STANDUP_ROBOT_CFG,
        "ball":  MICRODUCK_BALL_CFG,
    }
    cfg.scene.sensors = (feet_ground_cfg, support_foot_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

    # Extra contact headroom for the ball (ball-terrain + ball-robot contacts
    # on top of the full-collision robot's budget).
    cfg.sim.nconmax = 50

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: drop walking-specific terms ──────────────────────────────────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",           # gait-conditioned; replaced by pose_target_match below
        "soft_landing",   # velocity removes it
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Rewards: kick objective — TARGET speed, not max speed ────────────────
    # Two-sided landscape peaking at BALL_TARGET_SPEED (0.25 m/s — a gentle tap):
    #   • ball_forward_velocity, linear and CAPPED at the target: dense
    #     bootstrap gradient from the first touch. Weight 12.0 = 3.0/target so
    #     the at-target payoff stays ≈ +3/step (with the old weight 3.0 the
    #     payoff would be 0.75/step — too weak vs the ~7/step standing stack to
    #     justify the swing's transient pose/upright cost).
    #   • ball_speed_overshoot_penalty (weight -4.0): each m/s above target
    #     costs -4/step while it persists. Needed because the cap alone does
    #     NOT tame the kick — a harder kick keeps the ball at the cap for more
    #     steps, so total (per-step × rolling time) reward still grows with
    #     strike speed.
    # Slopes stay asymmetric (+12/(m/s) below, -4/(m/s) above): the optimum
    # sits at the target, but erring hard stays much cheaper than not kicking
    # (net reward only hits 0 at ~1.0 m/s, 4× the target).
    cfg.rewards["ball_forward_velocity"] = RewardTermCfg(
        func=microduck_mdp.ball_forward_velocity,
        weight=12.0,
        params={"asset_name": "ball", "max_speed": BALL_TARGET_SPEED},
    )
    cfg.rewards["ball_speed_overshoot"] = RewardTermCfg(
        func=microduck_mdp.ball_speed_overshoot_penalty,
        weight=-4.0,
        params={"asset_name": "ball", "target_speed": BALL_TARGET_SPEED},
    )

    # Support foot: binary +1 while the non-kicking foot touches the ground.
    # Always-on anti-hop — swinging the kicking leg is free, lifting the
    # support foot costs this every step. Also suppresses walking/dribbling
    # exploits (any gait loses this reward half the time).
    cfg.rewards["support_foot_grounded"] = RewardTermCfg(
        func=microduck_mdp.single_foot_grounded_reward,
        weight=2.0,
        params={"sensor_name": support_foot_ground_cfg.name},
    )

    # ── Rewards: stand cleanly before/after the kick ──────────────────────────
    # Legs at HOME. std=0.5 is deliberately loose: the kick itself is a big
    # transient leg deviation and must stay affordable.
    cfg.rewards["pose_stand_legs"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=2.0,
        params={
            "std": 0.5,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,   # HOME = standing
        },
    )

    # Neck/head at HOME (no head command in this task; tighter std — the head
    # takes no part in the kick).
    cfg.rewards["pose_stand_neck"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=1.0,
        params={
            "std": 0.3,
            "joint_indices": _NECK_JOINTS,
            "target_overrides": None,
        },
    )

    # Upright — velocity's exact recipe (weight 2.0, std²=0.05).
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)

    # Trunk at standing height — discourages crouching/squatting as a kick prep.
    cfg.rewards["height_stand"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=1.0,
        params={
            "std":           0.04,
            "target_height": STAND_Z,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # ── Sim2real regularisers — velocity parity (see standup env rationale) ──
    cfg.rewards["action_rate_l2"].weight = -0.1  # stage-0; curriculum ramps to -1.0
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
    # No terrain-height sensor in this env (flat only) — drop the base
    # template's sensor-backed terms, like standup/ground-pick do.
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    # IMU obs delay — match velocity's 2026-07 audit values.
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
    # below applies to the actor only.
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
    # to the ball (no ball sensing on the real robot), the critic uses it to
    # predict the kick payoff.
    cfg.observations["critic"].terms["ball_position"] = ObservationTermCfg(
        func=microduck_mdp.ball_pos_in_base, params={"asset_name": "ball"},
    )
    cfg.observations["critic"].terms["ball_velocity"] = ObservationTermCfg(
        func=microduck_mdp.ball_vel_in_base, params={"asset_name": "ball"},
    )

    # ── Command: tiny noise around zero (obs-shape parity only) ───────────────
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

    # ── Terminations ──────────────────────────────────────────────────────────
    # fell_over KEPT (robot starts standing and must stay up through the kick).
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
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
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)  # match velocity

    # Joint noise on the standing start: deployment hands off from the walk /
    # velstand policy, whose settled stand won't match HOME exactly.
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)

    # Standing-only start (reuses the standup env's ground-state machinery for
    # the noisy upright spawn: random yaw ± tilt noise, z near equilibrium).
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob":   0.0,
            "face_up_prob":     0.0,
            "sitting_prob":     0.0,
            "standing_prob":    1.0,
            "sitting_tilt_max": math.radians(5),  # ±5° pitch/roll on the stand
            "standing_z_min":   0.11,
            "standing_z_max":   0.12,
        },
    )

    # Ball placement — MUST come after set_ground_state (events run in dict
    # insertion order; the ball position derives from the final robot pose).
    # Also stores the per-env kick direction (robot heading at reset).
    ball_offset_y = -BALL_OFFSET_ABS_Y if kick_foot == "right" else BALL_OFFSET_ABS_Y
    cfg.events["reset_ball"] = EventTermCfg(
        func=microduck_mdp.reset_ball_in_front_of_foot,
        mode="reset",
        params={
            "offset":      (BALL_OFFSET_X, ball_offset_y),
            "noise_xy":    BALL_POS_NOISE_XY,
            "ball_radius": BALL_RADIUS,
            "asset_name":  "ball",
        },
    )

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

    # ── Terrain: flat only (a ball on rough terrain is a different task) ──────
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Curriculum ────────────────────────────────────────────────────────────
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # action_rate ramp — velocity's exact stages (-0.1 → -1.0 by iter 1500).
    # NOTE: the kick is a fast one-shot swing; if the converged kick is too
    # weak, softening the ramp end (-1.0 → -0.6) is the first knob to try
    # (motion-blocker vs dynamic-task tradeoff, see standup regularization notes).
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.1},
                {"step": 500 * 24,   "weight": -0.2},
                {"step": 750 * 24,   "weight": -0.4},
                {"step": 1000 * 24,  "weight": -0.6},
                {"step": 1250 * 24,  "weight": -0.8},
                {"step": 1500 * 24,  "weight": -1.0},
            ],
        },
    )

    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                    {"step": 1500 * 24, "range": 0.015},
                ],
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )

    if ENABLE_VELOCITY_PUSHES:
        # Ramp pushes in AFTER the kick skill starts forming: a full-strength
        # shove during the one-legged strike phase at iter 0 would tax the
        # discovery of the swing itself (same timing lesson as standup).
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={
                "event_name": "push_robot",
                "push_stages": [
                    {"step": 0,         "velocity_range": {"x": (0.0, 0.0),    "y": (0.0, 0.0)}},
                    {"step": 500 * 24,  "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
                    {"step": 1000 * 24, "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}},
                ],
            },
        )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckBallKickRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name=f"ball_kick_{KICK_FOOT}",
    run_name=f"ball_kick_{KICK_FOOT}",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=10_000,
)
