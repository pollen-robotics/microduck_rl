"""Microduck velocity environment — roller skate variant.

MIGRATED to mjlab 1.3.0 + canonical BAM (2026-07), matching the velocity env's
sim2real machinery, and updated for the NEW roller model:

  - `get_walk_rollers_spec` now loads `robot_groundcontact_rollers.xml`
    (it silently loaded the wheel-less standup model before): 14 actuated
    joints + 4 passive wheels (passive_{L,R}{F,R}wheel), two per blade,
    INTERSPERSED in the joint order (after each ankle) — everything resolves
    joints by NAME, never by index.
  - Legs run the canonical BAM actuator like every other variant (was a plain
    XML PD — an actuator-physics mismatch, and no joint-friction DR).
  - Obs migrated to the unified 61D layout (twist + zero-padded head/body
    command slots) so roller policies load through the runtime's
    --new-cmd-obs path. Symmetry OFF (SYMMETRY_CFG is hardcoded for the old
    51D layout).
  - DR/noise/delays matched to the velocity env's FIXED (non-accumulating,
    per-env-verified) versions; wheel-bearing frictionloss DR kept
    (dr.dof_frictionloss on the passive wheels + existing curriculum).

Task design (unchanged — the roller recipe):
  cmd_x semantics: 0 = coast, >0 = push to accelerate, <0 = brake.
  cmd[2] = heading error via RelativeHeadingVelocityCommand.
  Sole positive task reward is wheel_speed — the robot must actually spin its
  wheels; braking/skating_air_time/forward_lean/heading_tracking shape the
  skating style.
"""

import math
from copy import deepcopy

# Symmetry — OFF: SYMMETRY_CFG's obs permutation is hardcoded for the old 51D
# layout and breaks on the 61D obs (same situation as all other v1.5+ envs).
ENABLE_SYMMETRY = False

# ── Domain randomisation toggles (matched to the velocity env) ────────────────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True   # BAM friction budget per-env (legs)
ENABLE_ARMATURE_RANDOMIZATION        = True   # legs only — NOT the wheel bearings
ENABLE_WHEEL_FRICTION_RANDOMIZATION  = True   # bearing frictionloss on passive wheels
ENABLE_VELOCITY_PUSHES               = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True   # obs-level per-env rotation
ENABLE_ENCODER_BIAS                  = True

# ── Ranges (matched to the velocity env unless roller-specific) ───────────────
COM_RANDOMIZATION_RANGE          = 0.003   # ±3mm initial, ramped via curriculum
HEAD_COM_RANDOMIZATION_RANGE     = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE     = (0.9, 1.1)
VELOCITY_PUSH_INTERVAL_S         = (3.0, 6.0)
VELOCITY_PUSH_RANGE              = (-0.2, 0.2)  # roller-specific: gentler than walk ±0.3
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0
ENCODER_BIAS_RANGE               = (-0.015, 0.015)

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

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROLLERS_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_velocity_rollers_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck roller skate velocity tracking environment configuration."""

    # passive_.*: 999.0 → passive wheel joints are matched but effectively ignored
    std_standing = {
        r".*hip_yaw.*": 0.05,
        r".*hip_roll.*": 0.05,
        r".*hip_pitch.*": 0.05,
        r".*knee.*": 0.05,
        r".*ankle.*": 0.05,
        r".*neck.*": 0.05,
        r".*head.*": 0.05,
        r".*passive_.*": 999.0,
    }

    std_walking = {
        r".*hip_yaw.*": 0.3,
        r".*hip_roll.*": 0.6,  # loosened: skating requires wide lateral push
        r".*hip_pitch.*": 0.4,
        r".*knee.*": 0.4,
        r".*ankle.*": 0.25,
        r".*neck.*": 0.05,
        r".*head.*": 0.05,
        r".*passive_.*": 999.0,
    }

    std_running = {
        r".*hip_yaw.*": 0.5,
        r".*hip_roll.*": 0.8,  # loosened: skating requires wide lateral push
        r".*hip_pitch.*": 0.8,
        r".*knee.*": 0.8,
        r".*ankle.*": 0.5,
        r".*neck.*": 0.05,
        r".*head.*": 0.05,
        r".*passive_.*": 999.0,
    }

    # 2026-07 model: the roller_blade bodies were merged into the ankles (blade
    # mesh is now a visual geom on ankle_{l,r}_v1); the tires hang directly off
    # the ankles. Each ankle subtree's only collision geoms are its two tires,
    # so this keeps the old per-foot semantics: 2 slots, left first.
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(ankle_l_v1|ankle_r_v1)$",
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

    # Robot setup
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROLLERS_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    # Action configuration
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0
    # NOTE: an env-side action clip was tried here to bound the target, but the
    # deployment pipeline (infer_policy.py) does NOT clip → the clip would only
    # exist in sim, a train/deploy mismatch. The over-command deterrent lives
    # policy-side instead (action_over_limit reward below), baked into the network
    # so it transfers with the ONNX.

    # === REWARDS ===
    keep = {"pose", "upright", "body_ang_vel", "angular_momentum", "action_rate_l2"}
    for name in list(cfg.rewards.keys()):
        if name not in keep:
            del cfg.rewards[name]

    cfg.rewards["pose"].params["std_standing"] = std_standing
    cfg.rewards["pose"].params["std_walking"] = std_walking
    cfg.rewards["pose"].params["std_running"] = std_running
    cfg.rewards["pose"].params["walking_threshold"] = 0.01
    cfg.rewards["pose"].params["running_threshold"] = 0.5
    cfg.rewards["pose"].weight = 2.0

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 2.0

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["action_rate_l2"].weight = -1.0

    cfg.rewards["com_height_target"] = RewardTermCfg(
        func=microduck_mdp.com_height_target,
        weight=2.0,
        params={"target_height_min": 0.0935, "target_height_max": 0.1235},
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "self_collision"},
    )
    # Gated to the STANCE foot only (sensor_name) so lifting the swing foot is no
    # longer punished — the old ungated -5.0 was minimised by keeping both blades
    # flat on the ground (the swizzle) and actively fought the stride. Weight also
    # softened -5.0 -> -2.0 to leave room for a slightly angled push.
    cfg.rewards["feet_flat"] = RewardTermCfg(
        func=microduck_mdp.feet_flat_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
            "sensor_name": "feet_ground_contact",
        },
    )
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_action_rate_l2, weight=-0.5
    )
    cfg.rewards["neck_joint_pos_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_joint_pos_l2, weight=-0.5
    )
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2, weight=-1e-3
    )
    # Deter OVER-COMMANDING a joint past its hard stop (policy-side, transfers via
    # the ONNX). hip_roll's ±0.38 rad limit vs the ±10 rad ctrlrange let the low-kp
    # servo be commanded far past the stop and slam it with max torque — a fragile
    # sim-only trick. This penalises only the COMMAND beyond (limit + 0.3 overshoot),
    # so the joint keeps its full reachable range (a qpos penalty stole that range
    # and broke the gait) while the wild over-drive is discouraged.
    cfg.rewards["action_over_limit"] = RewardTermCfg(
        func=microduck_mdp.action_over_limit_penalty,
        weight=-0.5,
        params={"action_name": "joint_pos", "overshoot": 0.3},
    )
    # Pull hip_roll back toward neutral so the stance stops resting splayed on the
    # hip_roll limits. L1 = constant gradient: it gently closes the legs AT REST,
    # but the strong stride rewards (wheel_speed, single_support, air_time) easily
    # overpower it during an active push → closes the posture WITHOUT preventing
    # the lateral push stroke. Tune: raise if still splayed, lower if it flattens
    # the stride. (Physics caveat: if the soft hip_roll servo can't hold a narrow
    # stance under body weight, the policy will bend knees / lower CoM to unload
    # it — or, if no stable narrow stance exists, it stays partly splayed.)
    cfg.rewards["hip_roll_neutral"] = RewardTermCfg(
        func=microduck_mdp.joint_deviation_l1,
        weight=-2.0,  # -1.0 -> -2.0: stronger centring pull. Sim already keeps hip_roll
                      # narrow, but a stronger corrective may help the REAL robot resist
                      # whatever spreads the legs (deployment/disturbance). Lower if it
                      # flattens the push.
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*hip_roll.*",))},
    )
    # Sole positive task reward — robot must spin wheels to get anything
    # vel_scale 0.5 -> 0.3: the tanh target speed. Measured on a trained ckpt, the
    # policy only reaches ~0.33 m/s at max push, so a 0.5 target sat on the
    # un-saturated tanh slope and kept pushing it to go faster than it can (over-
    # reach -> launch instability). 0.3 saturates near the achievable speed, so it
    # is 'content' there instead of over-driving.
    cfg.rewards["wheel_speed"] = RewardTermCfg(
        func=microduck_mdp.wheel_speed_reward,
        weight=10.0,
        params={"command_name": "twist", "vel_scale": 0.3},
    )
    # Brake: reward stopping when cmd_x < 0. Silent at cmd_x >= 0 (coast/push).
    cfg.rewards["braking"] = RewardTermCfg(
        func=microduck_mdp.braking_reward,
        weight=1.0,
        params={"command_name": "twist", "vel_std": 0.3},
    )
    # Air time during push: pay the recovery-foot lift, but ONLY when the body is
    # actually moving forward (vel_gate_ref) — otherwise a fast in-place flutter
    # farmed this. threshold_min raised 0.15 → 0.25 to forbid ultra-short swings
    # (caps the frantic kick cadence); glide below rewards the slow phase.
    # air_time rewards each swing → drives swing FREQUENCY; glide rewards staying
    # on one blade → drives commitment. Balance tilted toward glide (3.0) over
    # air_time (2.0) because the cadence was still too fast. air_time kept high
    # enough (2.0) that lifting the foot stays worthwhile.
    # Calm gait: the aggressive [0.40, 1.00] window forced big long swings ->
    # violent kicks that tipped the real robot. Back to a gentle [0.15, 0.45]
    # (small swings allowed, none forced long) and weight 2.0 -> 1.5 so swinging
    # is less incentivised (lower cadence). glide (below) rewards the coast, so it
    # pushes only occasionally.
    cfg.rewards["skating_air_time"] = RewardTermCfg(
        func=microduck_mdp.skating_air_time_reward,
        weight=1.5,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "threshold_min": 0.15,
            "threshold_max": 0.45,
            "vel_gate_ref": 0.2,
        },
    )
    # Glide phase (single-support REQUIRED, unlike the earlier broken attempt):
    # reward coasting on one blade with quiet legs so the policy commits to each
    # stroke instead of kicking frantically. Weight raised 1.5 → 3.0 to actually
    # out-weigh the swing-frequency pull of air_time.
    cfg.rewards["glide"] = RewardTermCfg(
        func=microduck_mdp.glide_reward,
        weight=4.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "vel_ref": 0.2,
        },
    )
    # NOTE: a recover_pose reward (reward default leg pose + quiet + coasting during
    # the pause) was tried to get "stroke -> recover-to-neutral -> stroke", but
    # rewarding the SYMMETRIC default posture + dropping single_support's double
    # penalty re-opened the symmetric swizzle -> reverted. A proper retry must be
    # PHASE-GATED (reward the neutral only briefly right after a stroke, not
    # continuously) and keep the double-support penalty.
    # Single-support stride vs double-support swizzle. Rewards exactly-one-blade-
    # down and penalises both-down while pushing — the core anti-swizzle signal.
    # Gated on forward speed too, so stepping that doesn't propel earns nothing.
    cfg.rewards["single_support"] = RewardTermCfg(
        func=microduck_mdp.single_support_reward,
        weight=3.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "vel_gate_ref": 0.2,
        },
    )
    # Balance left/right leg usage. With symmetry augmentation OFF nothing stops a
    # lopsided stride (pushing mostly with one leg) that veers and destabilises,
    # esp. at launch. Penalises the cumulative swing-time imbalance |L-R|/(L+R);
    # the instantaneous one-foot-swinging asymmetry of a real stride is fine.
    cfg.rewards["gait_symmetry"] = RewardTermCfg(
        func=microduck_mdp.gait_symmetry_penalty,
        weight=-1.0,
        params={"sensor_name": "feet_ground_contact"},
    )
    # NOTE: a contact_frequency penalty was tried here to slow the cadence, but it
    # penalises contact CHANGES — minimised by never lifting a foot (the swizzle),
    # so it pushes toward exactly the gait we fought to leave. Reverted; the
    # widened air-time window above is the safe cadence-slower (it forbids short
    # swings without rewarding not-stepping).
    # Encourage slight forward lean when pushing to counteract backward torque.
    cfg.rewards["forward_lean"] = RewardTermCfg(
        func=microduck_mdp.forward_lean_reward,
        weight=1.5,
        params={"command_name": "twist", "target_pitch": 0.262, "std": 0.1},
    )
    # Heading command DISABLED (straight-line focus), but we hold the heading so it
    # doesn't drift: heading_hold rewards the yaw ANGLE staying near the spawn
    # heading. Corrective (allows yaw to steer back) — unlike a yaw-RATE penalty,
    # which froze the yaw and made drift WORSE (tried and reverted). Re-add real
    # heading_tracking (turning) once the stride is solid.
    cfg.rewards["heading_hold"] = RewardTermCfg(
        func=microduck_mdp.heading_hold_reward,
        weight=1.0,
        params={"std": 0.4, "asset_cfg": SceneEntityCfg("robot")},
    )

    # === TERMINATIONS ===
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # === EVENTS ===
    # BAM (mjlab_frictionloss branch) writes per-env dof_frictionloss/dof_damping
    # every step; this no-op event registers those fields for per-world expansion.
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )

    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )

    del cfg.events["foot_friction"]  # wheels roll; ground friction lives in the XML

    if ENABLE_VELOCITY_PUSHES:
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=VELOCITY_PUSH_INTERVAL_S,
            params={
                "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    cfg.events["reset_base"].params["pose_range"]["z"] = (0.1335, 0.1435)

    # Wheel-bearing friction DR: real bearings have a little drag; the XML keeps
    # frictionloss=0 for trainability and the curriculum ramps it in. mjlab 1.3.0
    # stock dr op (operation="abs" writes the value directly; non-accumulating).
    if ENABLE_WHEEL_FRICTION_RANDOMIZATION:
        cfg.events["randomize_wheel_friction"] = EventTermCfg(
            func=dr.dof_frictionloss,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^passive_.*wheel",)),
                "operation": "abs",
                "ranges": (0.000, 0.000),  # ramped up by wheel_friction_curriculum
            },
        )

    # ── DR matched to the velocity env's FIXED versions ───────────────────────
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

    if ENABLE_ARMATURE_RANDOMIZATION:
        # Legs/head only — the wheel bearings' tiny armature is excluded (its DR
        # is the frictionloss event above).
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    # === OBSERVATIONS (unified 61D layout) ===
    del cfg.observations["actor"].terms["base_lin_vel"]
    # 1.3.0 base template adds sensor-based foot_height + height_scan; the roller
    # env has no terrain-height sensor.
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,
        scale=1.0,
    )

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )
    # IMU delay 0-1 control steps (matches velocity: the real dxl IMU path is fast)
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    # Observation noise — matched to the velocity env
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)

    # IMU mounting-misalignment DR (obs-level, actor only — matches velocity)
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # 1-ctrl-step lag on joint_vel (Dynamixel present_velocity moving average)
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    # Exclude the passive wheel joints from joint_pos/vel obs (obs dim 14, matches
    # the action space). Deepcopy per group so the encoder-bias `biased` flag
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

    # Privileged wheel speeds for the critic (4 wheels in the new model).
    wheel_cfg = SceneEntityCfg("robot", joint_names=(r"^passive_.*wheel",))
    cfg.observations["critic"].terms["wheel_vel"] = ObservationTermCfg(
        func=mdp.joint_vel_rel,
        scale=1.0,
        params={"asset_cfg": wheel_cfg},
    )

    # Command obs parity with the 61D family layout: head/body slots zero-padded
    # (the roller task drives heading through the twist slot instead).
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # === COMMANDS ===
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False  # RelativeHeadingVelocityCommand handles heading internally
    command.ranges.heading = None    # must be None when heading_command=False
    # cmd_x semantics: 0=coast, >0=push to accelerate, <0=brake to stop
    command.ranges.lin_vel_x = (-0.5, 0.6)
    command.ranges.lin_vel_y = (0.0, 0.0)
    # ang_vel_z range is the clip limit for cmd[2] = heading error (rad).
    # Set to 0 → cmd[2] is always 0 → no turning demand (straight-line focus).
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.viz.z_offset = 0.5
    cfg.commands["twist"] = microduck_mdp.RelativeHeadingVelocityCommandCfg(**vars(command))

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # === CURRICULUM ===
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # action_rate penalty raised (-0.5/-0.8/-1.0 -> -1.0/-1.5/-2.0) for a CALMER
    # gait: this is the main "less movement" lever — it penalises fast/large action
    # changes, so motions become smaller, smoother AND less frequent (rapid
    # alternation = big action change = penalised). Dial back if it gets sluggish
    # / can't push enough to move.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -1.0},
                {"step": 250 * 24, "weight": -1.5},
                {"step": 500 * 24, "weight": -2.0},
            ],
        },
    )

    if ENABLE_WHEEL_FRICTION_RANDOMIZATION:
        # Delayed + softened ramp: the previous schedule started adding bearing
        # drag at iter 750 — right when wheel_speed peaked — and reached 0.003,
        # which (with the heading ramp below) pushed the policy off skating into
        # a heading-farming local optimum. Keep the wheels free until skating is
        # robust, then add gentle, realistic drag.
        cfg.curriculum["wheel_friction"] = CurriculumTermCfg(
            func=microduck_mdp.wheel_friction_curriculum,
            params={
                "event_name": "randomize_wheel_friction",
                "ranges_stages": [
                    {"step":    0 * 24,  "ranges": (0.0000, 0.0000)},
                    {"step": 2000 * 24,  "ranges": (0.0005, 0.0005)},
                    {"step": 3500 * 24,  "ranges": (0.0010, 0.0010)},
                    {"step": 5000 * 24,  "ranges": (0.0015, 0.0015)},
                ],
            },
        )

    # (heading_tracking_weight curriculum removed — heading is disabled while we
    # focus on straight-line skating. Re-add together with the reward above.)

    # CoM randomization curricula — velocity's ramp, capped lower for the
    # balance-sensitive skating task (audit lesson: ±30 mm forced a nervous
    # gait on the walker; skates are even less forgiving).
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
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

    return cfg


MicroduckRollersRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # matches the family; normalizer baked into ONNX by export.py
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
        entropy_coef=0.03,  # roller-specific: higher exploration than the walk envs
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
    experiment_name="velocity_rollers",
    run_name="velocity_rollers",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=50_000,
)
