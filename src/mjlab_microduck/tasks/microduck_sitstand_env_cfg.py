"""Microduck *sitstand* task (v1.5, mjlab 1.3.0) — commanded sit ↔ stand, GENTLY.

One policy, both directions, driven by a posture command:
    cmd (twist slot) = [sit_flag, 0, 0]   sit_flag ∈ {0 = STAND, 1 = SIT}
"Stand" is the all-zero command — the same deployment idle as every other
policy. The command flips mid-episode with a dwell time of a few seconds, so
each episode trains descents, seated rest, rises and standing rest, plus
"hold what you're already doing" (reset state × command are independent).

2026-08 rebuild from scratch (the old phase-cycle env predates the 1.3.0
migration and every sit/standup lesson). Design synthesis:
  - Posture-conditioned single-target rewards (mdp posture_*): the sit env's
    minimum-viable "organic discovery" stack, but the target (SIT keyframe +
    SIT_Z vs HOME + STAND_Z) is selected per env from the live command. No
    trajectory, no waypoints, no phase timing — the policy discovers its own
    transition path, in as many steps as it likes (knee-down first, head
    assist, etc. are all allowed: full-collision model, no head-ground
    penalty, no fall termination).
  - Gentleness both ways: descent-speed cap (sit env's proven recipe, -10
    from step 0) AND a mirrored rise-speed cap (introduced by curriculum
    AFTER the rise is discovered — the standup attempt-tax lesson), plus the
    |a_z| shock penalty throughout.
  - Rest quality: posture_stillness (velocity-Gaussian at the commanded
    height, tilt-gated) + posture_composite (multiplicative height·upright·
    pose vs the commanded target — partial-sum exploits like plank/flop/lean
    collapse to ~0).
  - Head commandable in BOTH postures (head_pose command + tracking, exactly
    like velocity/standup), body_command slot zero-padded → 61D obs parity.
  - Sim2real: velocity-parity DR / obs noise / delays / regularisers (the
    transferring recipe), sit env's contact-solver hardening (nconmax=200,
    iters 30/50 — seated contact NaN fix), delayed push ramp (pushes early
    made the sit env unlearn sitting).

Keyframes (stability-verified, keep in sync with sit/standup envs):
  SIT  = knee ±1.35, hip_pitch ∓0.4079, ankle/hip_roll 0, trunk z 0.060
         (swept 2026-07-27 — the old keyframe tipped over; verify TILT in sim
         before changing this pose).
  STAND = HOME joints, trunk z 0.115 (measured standing equilibrium).

Joint layout (14 actuated joints):
    0-4 : left  leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
    5-8 : neck/head (neck_pitch, head_pitch, head_yaw, head_roll)
    9-13: right leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
"""

import math
from copy import deepcopy

# Symmetry
ENABLE_SYMMETRY = False

# ── Domain randomisation (matched to the velocity env for sim2real parity) ────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True   # match velocity: randomize head-assembly CoM
ENABLE_KP_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_KD_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_MASS_INERTIA_RANDOMIZATION    = True   # match velocity: dr.pseudo_inertia (mass+inertia)
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True   # match velocity: FrictionDRBamActuator.friction_scale
ENABLE_ARMATURE_RANDOMIZATION        = True   # match velocity: reflected rotor inertia
ENABLE_VELOCITY_PUSHES               = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True   # match velocity: obs-level per-env misalignment
ENABLE_ENCODER_BIAS                  = True   # match velocity: per-env joint encoder offset (actor obs)

# ── Ranges (matched to the velocity env) ──────────────────────────────────────
COM_RANDOMIZATION_RANGE             = 0.003           # ramped to 0.015 via com_range curriculum
HEAD_COM_RANDOMIZATION_RANGE        = 0.003           # ramped to 0.01 via head_com_range curriculum
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)    # unused (kp DR off)
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)      # unused (kd DR off)
VELOCITY_PUSH_INTERVAL_S            = (3.0, 6.0)
# Final magnitude matches velocity's ±0.3 but the ramp is DELAYED (see the
# push_magnitude curriculum): the sit env's lesson — pushes mid-descent before
# the transition motions have consolidated make the policy unlearn them and
# converge to "just stand doing nothing".
VELOCITY_PUSH_RANGE                 = (-0.3, 0.3)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0  # match velocity (obs-level, zero-centered random axis)

# Episode length: room for 2-3 posture segments (dwell 3.5-6.5 s each), i.e.
# at least one full sit → rest → rise → rest cycle per episode.
EPISODE_LENGTH_S = 12.0
# Dwell time in each commanded posture before a resample may flip it. The
# lower bound must comfortably exceed a gentle transition (~1.5 s) plus some
# rest, so "arrive, then hold still" is always trained.
POSTURE_DWELL_S  = (3.5, 6.5)
# Probability a resample commands SIT (vs STAND). 0.5 → all four combinations
# of (reset state × command) get equal coverage, including both holds.
SIT_PROB         = 0.5

# ── SIT keyframe (joint_pos index → angle in rad). Single fixed target. ─────
# STABILITY-VERIFIED 2026-07-27 (sit env, scratchpad sweep_sit_pose2.py):
# knee ±1.35, hip_pitch = HOME ∓ 0.05 lean, ankle 0, hip_roll 0 settles at
# 3-5° tilt for 95-100% of noisy resets. The old keyframe (knee ±1.0472,
# hip_pitch HOME) is NOT statically stable — it tips to ~88° in 1 s and
# silently drove the sit env's whole hop/back-flop/plank exploit chain.
# If the robot or keyframe changes, RE-RUN THE SWEEP — verify tilt, not z.
# Keep in sync with microduck_sit_env_cfg.SITTING_TARGET_OVERRIDES and
# microduck_standup_env_cfg.SITTING_JOINT_OVERRIDES.
SITTING_TARGET_OVERRIDES = {
    1:   0.0,      # left  hip_roll   (HOME -0.0873)
    2:  -0.4079,   # left  hip_pitch  (HOME -0.4579; +0.05 = slight fwd lean)
    3:   1.35,     # left  knee       (HOME -0.0049)
    4:   0.0,      # left  ankle      (HOME +0.4530)
    # neck/head intentionally omitted → steered by the head_pose command.
    10:  0.0,      # right hip_roll   (HOME +0.0873)
    11:  0.4079,   # right hip_pitch  (HOME +0.4579)
    12: -1.35,     # right knee       (HOME +0.0049)
    13:  0.0,      # right ankle      (HOME -0.4530)
}

_LEG_JOINTS  = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]

# Trunk height targets (m) — both MEASURED in sim, never carried across robot
# or keyframe changes (sit run-1 / standup lessons).
STAND_Z = 0.115
SIT_Z   = 0.060

# Upright gating window for ``upright_while_tall``: full upright incentive
# above STAND_UPRIGHT_Z, fades to 0 at SIT_UPRIGHT_Z (committed to the sit).
# Blocks the "tip backward while still high" descent exploit; the always-on
# upright_linear floor covers the seated regime.
STAND_UPRIGHT_Z = 0.10
SIT_UPRIGHT_Z   = 0.075

# Target-ramp duration (s): the command term slews an internal target blend
# STAND↔SIT over this time, and the posture rewards track the MOVING target.
# THE anti-crash mechanism (run-1 failure: near-instant transitions). With a
# binary target, arriving early pays the full goal jackpot (~7/step) for
# every step saved, while the linear speed caps integrate to a bounded
# excess-distance cost (~50 total for an instant drop) — crashing won ~7×.
# With the ramp, being AHEAD of the setpoint zeroes the height/composite
# stack for the ramp remainder, so tracking the slow setpoint is the argmax.
# 55 mm over 2 s ≈ 0.028 m/s, comfortably under both caps below.
POSTURE_RAMP_S = 2.0

# Vertical-speed caps (m/s) — now BACKSTOPS for overshoot/bounce around the
# slewed target (see POSTURE_RAMP_S), not the primary gentleness mechanism.
# The rise cap is looser (rising against gravity needs some momentum to get
# over the heels) and is introduced by curriculum only after the rise motion
# has been discovered — see the rise_speed_weight curriculum.
MAX_DESCENT_SPEED = 0.05
MAX_RISE_SPEED    = 0.08

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
    MICRODUCK_ROUGH_TERRAINS_CFG,
    HEAD_BODY_NAMES,
    HEAD_POSE_CMD_RESAMPLE_S,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_sitstand_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck sitstand environment configuration."""

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

    # NOTE: no head-ground contact penalty here (unlike the sit env). Using the
    # head as a third support point during transitions is explicitly allowed —
    # the plank-as-terminal-rest exploit is anti-selected by posture_composite
    # + posture_stillness instead (both ≈0 at plank tilt/height).

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    # Standup robot variant: full collision meshes — the body must physically
    # rest on the ground while seated, and knees/head may touch mid-transition.
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors  = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

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
        "pose",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Rewards: posture-conditioned single-target stack ──────────────────────
    # Every task term below reads the commanded posture and selects its target
    # (SIT keyframe + SIT_Z vs HOME + STAND_Z) per env. Weights mirror the sit
    # env's proven stack (positive task mass ≈ velocity scale, so the shared
    # sim2real regularisers act at the same RELATIVE strength — the standup
    # transfer lesson).

    # Pose target — legs only (head is command-steered). Generous std keeps
    # gradient alive from either end (~1.35 rad knee delta).
    cfg.rewards["posture_pose_legs"] = RewardTermCfg(
        func=microduck_mdp.posture_pose_match,
        weight=4.0,
        params={
            "command_name":  "twist",
            "std":           0.5,
            "joint_indices": _LEG_JOINTS,
            "sit_overrides": SITTING_TARGET_OVERRIDES,
        },
    )

    # Head pose tracking (commandable head control, like velocity/standup) —
    # active in BOTH postures. Weight kept light so a transient head-assist
    # during a transition only pays a small tracking cost.
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=0.75,
        params={"command_name": "head_pose", "std": 0.5},
    )

    # L1 bootstrap — constant gradient toward the commanded pose.
    cfg.rewards["posture_pose_l1"] = RewardTermCfg(
        func=microduck_mdp.posture_pose_l1,
        weight=1.0,
        params={
            "command_name":  "twist",
            "joint_indices": _LEG_JOINTS,
            "sit_overrides": SITTING_TARGET_OVERRIDES,
        },
    )

    # Trunk height — two-layer Gaussian (standup recipe: wide layer for the
    # bootstrap pull across the 55 mm travel, sharp layer so the final cm has
    # real gradient instead of a saturated plateau) + L1 transition driver.
    cfg.rewards["posture_height"] = RewardTermCfg(
        func=microduck_mdp.posture_height_gaussian,
        weight=1.0,
        params={
            "command_name": "twist",
            "sit_z":        SIT_Z,
            "stand_z":      STAND_Z,
            "std":          0.04,
        },
    )
    cfg.rewards["posture_height_sharp"] = RewardTermCfg(
        func=microduck_mdp.posture_height_gaussian,
        weight=1.0,
        params={
            "command_name": "twist",
            "sit_z":        SIT_Z,
            "stand_z":      STAND_Z,
            "std":          0.015,
        },
    )
    # L1 weight 6.0: between sit's 5.0 and standup's 7.5 — resting in the
    # WRONG posture must be clearly net-negative in both directions (staying
    # seated under a stand command was the standup env's stall mode at low L1).
    cfg.rewards["posture_height_l1"] = RewardTermCfg(
        func=microduck_mdp.posture_height_l1,
        weight=6.0,
        params={
            "command_name": "twist",
            "sit_z":        SIT_Z,
            "stand_z":      STAND_Z,
        },
    )

    # Rise bootstrap — pays for upward motion itself when STAND is commanded
    # and the trunk is below 0.125 (just ABOVE the target so the final cm
    # still pays). Destination-only rewards have zero gradient at zero motion;
    # without this the standup env parked seated. Zero under a SIT command.
    cfg.rewards["rise_bootstrap"] = RewardTermCfg(
        func=microduck_mdp.posture_rise_bootstrap,
        weight=0.75,
        params={
            "command_name": "twist",
            "max_height":   0.125,
            "max_vz":       MAX_RISE_SPEED,  # explosive launch can't out-earn a gentle rise
        },
    )

    # ── Gentleness (the point of this env) — three complementary signals ─────
    #  - ``descent_speed``: per-step penalty on downward vz beyond 0.05 m/s.
    #    THE anti-brutality term for the sit: a fast drop pays on every step
    #    of the fall so it can't be amortised. -10 from step 0 (sit lesson:
    #    at -5 a crash-sit was net-positive), tightened to -20 by curriculum.
    #  - ``rise_speed``: the mirror cap for the stand-up (0.08 m/s). Starts at
    #    weight 0 and is introduced at iter 750 by curriculum — the standup
    #    lesson: a motion-tax active while the skill is being DISCOVERED makes
    #    exploratory attempts net-negative and the skill is never found. The
    #    sit-keyframe start is easy (no prone flips), so 750 is late enough.
    #  - ``gentle_motion``: |a_z| shock penalty, both directions, always on.
    #
    # ⚠️ POSITIVE weights, deliberately: these three functions ALREADY return
    # negative values (-clamp(...), -|a_z|), same convention as the *_l1_penalty
    # helpers (used with +1/+6 here). Run 7ev90yd9 (2026-08-12) had them at
    # negative weights — the double negative made them REWARDS for violence
    # (wandb: Episode_Reward/descent_speed +4.6, rise_speed +2.1, gentle_motion
    # +0.57, the three biggest positive terms) and trained a butt-hopping,
    # crash-sitting policy. Same bug class roller_standup found in gentle_rise.
    # After any reward change, check wandb Episode_Reward/<penalty> stays ≤ 0.
    cfg.rewards["descent_speed"] = RewardTermCfg(
        func=microduck_mdp.trunk_downward_velocity_penalty,
        weight=10.0,
        params={
            "max_down_vel": MAX_DESCENT_SPEED,
            "asset_cfg":    SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["rise_speed"] = RewardTermCfg(
        func=microduck_mdp.trunk_upward_velocity_penalty,
        weight=0.0,
        params={
            "max_up_vel": MAX_RISE_SPEED,
            "asset_cfg":  SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["gentle_motion"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # Two-layer upright pressure (sit env values — the anti-flop calibration):
    #  - always-on linear floor: holds the trunk vertical at BOTH rests; at 2.5
    #    "lie on your back" trails upright rest by ~4.5/step (sit run-2 fix).
    #  - height-gated booster: blocks the "tip backward while tall" descent
    #    exploit; during the rise it doubles as an arrival-uprightness pull.
    cfg.rewards["upright_linear"] = RewardTermCfg(
        func=microduck_mdp.body_upright_linear,
        weight=2.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    cfg.rewards["upright_while_tall"] = RewardTermCfg(
        func=microduck_mdp.upright_while_tall,
        weight=1.5,
        params={
            "height_low":  SIT_UPRIGHT_Z,
            "height_high": STAND_UPRIGHT_Z,
            "asset_cfg":   SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Stillness at the commanded posture — "arrive, then rest QUIETLY, UPRIGHT"
    # as an explicit positive peak. The z gate is a band around the commanded
    # height (inactive during transitions); the tilt gate pays nothing for a
    # tilted rest (back/face/side flops earn zero — the sit run-2 exploit).
    cfg.rewards["posture_stillness"] = RewardTermCfg(
        func=microduck_mdp.posture_stillness,
        weight=2.0,
        params={
            "command_name":  "twist",
            "sit_z":         SIT_Z,
            "stand_z":       STAND_Z,
            "band_full":     0.012,
            "band_zero":     0.03,
            "vel_std":       0.05,
            "tilt_full_deg": 25.0,
            "tilt_zero_deg": 60.0,
        },
    )

    # Multiplicative goal score vs the COMMANDED target — kills partial-sum
    # farming in both postures (plank, flop, lean, park-1cm-short). Broad stds
    # keep gradient visible far from the goal (standup's proven calibration).
    # head_std adds the neck/head-at-command factor: the first sign-fixed run
    # rested with the head DANGLING to the floor (trunk/legs/z all on target →
    # full composite, only the 0.75 tracking term lost, and the hanging head
    # adds passive stability). With the head factor, the goal state itself
    # requires the head up at its commanded pose; transient head assist
    # mid-transition stays free (composite ≈0 there anyway).
    cfg.rewards["posture_composite"] = RewardTermCfg(
        func=microduck_mdp.posture_composite,
        weight=3.0,
        params={
            "command_name":  "twist",
            "sit_overrides": SITTING_TARGET_OVERRIDES,
            "joint_indices": _LEG_JOINTS,
            "sit_z":         SIT_Z,
            "stand_z":       STAND_Z,
            "height_std":    0.03,
            "upright_std":   0.40,   # ≈ 23° effective — plank (~70°+) scores ~0
            "pose_std":      0.40,
            "head_std":      0.40,   # head fully dropped (~1.2 rad) → factor ~0.01
        },
    )

    # ── Sim2real regularisers — MATCHED to velocity ─────────────────────────
    # velocity's exact set and absolute weights:
    #   • action_rate_l2: -0.1 at stage 0, ramped -0.1 → -1.0 by iter 1500
    #   • body_ang_vel -0.05, angular_momentum -0.02
    #   • soft_landing dropped; joint_torques_l2 / neck_action_rate_l2 not added
    # Plus joint_torque_rate_l2 (anti-jitter), phased in once the transition
    # motions exist. Both caps + |a_z| already push toward slow-careful motion;
    # per the regularizer-type lesson these smoothness terms damp jitter
    # WITHOUT blocking a slow big motion, so heavier-than-velocity would also
    # be defensible — start at parity, tighten only if the real robot shakes.
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=0.0
    )

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05      # velocity value
    cfg.rewards["angular_momentum"].weight = -0.02  # velocity value
    cfg.rewards.pop("soft_landing", None)           # velocity removes it

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # Drop the base "upright" Gaussian — replaced by the two-layer upright above.
    if "upright" in cfg.rewards:
        del cfg.rewards["upright"]

    # ── Observations (identical layout to walking / sit / standup policies) ───
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    # mjlab 1.3.0 base template adds sensor-based foot_height + height_scan obs.
    # Sitstand has no terrain-height sensor (and drops the walking foot rewards),
    # so remove these terms. foot_air_time/foot_contact(_forces) use the
    # feet_ground_contact sensor, which sitstand does define, so they stay.
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

    # IMU obs delay: max_lag 1 — velocity's 2026-07 audit value (real dxl IMU
    # path is fast, ±20 ms envelope).
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    # Obs noise matched to the velocity env.
    cfg.observations["actor"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise       = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise       = Unoise(n_min=-0.25, n_max=0.25)

    # IMU mounting-misalignment DR (match velocity): per-env constant rotation of
    # the IMU-derived actor obs; critic keeps the true values.
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # 1-ctrl-step lag on joint_vel (Dynamixel present_velocity is ~1 period old).
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    # Deepcopy joint_pos/joint_vel per group (they share base-template objects) so
    # the encoder-bias `biased` flag below applies to the actor only.
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    # Encoder-bias DR (match velocity): actor sees joint_pos + per-env bias;
    # critic keeps the true joint pos.
    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # ── Head pose command (commandable head control, like velocity/standup) ──
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.05, 0.05),    # neck_pitch
            (-0.05, 0.05),    # head_pitch
            (-0.07, 0.07),    # head_yaw
            (-0.015, 0.015),  # head_roll
        ),
    )

    # Command obs slots. head_command is the real head_pose command;
    # body_command stays zero-padded (body control not used here).
    # Layout parity with velocity/standup: [twist(3), head_pose(4), body_pose(6)].
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "head_pose"},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # ── Command: sit/stand posture flag in the twist slot ────────────────────
    # cmd = [sit_flag, 0, 0]; dwell-time resampling flips the posture mid-
    # episode. "Stand" is the all-zero command (deployment idle parity). The
    # runtime drives this by writing 0/1 into the vx slot of the command
    # buffer. Internally the term slews a target blend over POSTURE_RAMP_S
    # that the posture rewards track (see the constant's comment); the OBS
    # stays the raw binary flag.
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    command.heading_command   = False
    command.ranges.heading    = None
    command.resampling_time_range = POSTURE_DWELL_S
    command.debug_vis = False
    cfg.commands["twist"] = microduck_mdp.SitStandCommandCfg(
        **{
            **vars(command),
            "sit_prob": SIT_PROB,
            "ramp_s":   POSTURE_RAMP_S,
            "sit_z":    SIT_Z,
            "stand_z":  STAND_Z,
        }
    )

    # ── Terminations ──────────────────────────────────────────────────────────
    # No fall termination: wobbles/tips during transitions must play out so the
    # policy experiences the impact/upright costs instead of a truncated episode.
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # ── Events ────────────────────────────────────────────────────────────────
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
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)  # match velocity

    # Base reset: standing, just above the measured equilibrium (STAND_Z=0.115).
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.11, 0.12)

    # Reset-state mix: 50% standing / 50% already seated (SIT keyframe with
    # joint/tilt noise). Combined with the independent 50/50 posture command
    # this trains all four cases — sit-from-stand, rise-from-sit, hold-stand,
    # hold-sit — and hands the policy both goal states' values directly (the
    # sit env's discovery-bootstrap lesson, extended to both ends).
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob":          0.0,
            "face_up_prob":            0.0,
            "sitting_prob":            0.5,
            "standing_prob":           0.5,
            "sitting_joint_overrides": SITTING_TARGET_OVERRIDES,
            "sitting_joint_noise_std": 0.10,           # ≈ 6° per joint
            "sitting_tilt_max":        math.radians(8),
            "sitting_z_min":           0.06,            # settles to the 0.060 rest
            "sitting_z_max":           0.075,
            "standing_z_min":          0.11,
            "standing_z_max":          0.12,
        },
    )

    # MuJoCo physics robustness (sit env's contact NaN fix). The standup XML
    # has full collisions on every body; the seated pose puts trunk + folded
    # legs + head all in close ground/self contact. Default nconmax=35 and
    # solver iters=10 overflow the contact solver on sit attempts → NaN →
    # nan_state terminations that punish the descent itself ("learn then
    # unlearn by iter 500" pattern).
    cfg.sim.nconmax = 200
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50

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
        # mjlab 1.3.0: stock dr.body_ipos (operation="add") reads the compile-time
        # default each reset → non-accumulating natively.
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
        # match velocity: physics-consistent mass+inertia via pseudo_inertia
        # (alpha scales both by e^(2α), CoM untouched). Startup mode.
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
        # match velocity: scale BAM's friction budget per-env via the
        # FrictionDRBamActuator hook (dof_frictionloss is zeroed under BAM).
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    # NOTE: IMU mounting-misalignment is applied at the OBSERVATION level above
    # (matching velocity) — the old event-based randomize_imu_orientation wrote
    # site_quat, which under mjlab 1.3.0 is neither per-env nor read by the obs.

    # ── Terrain ───────────────────────────────────────────────────────────────
    if not rough:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
    else:
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = MICRODUCK_ROUGH_TERRAINS_CFG
        if play:
            cfg.scene.terrain.terrain_generator.curriculum = False
            cfg.scene.terrain.terrain_generator.num_cols = 5
            cfg.scene.terrain.terrain_generator.num_rows = 5

    # ── Curriculum ────────────────────────────────────────────────────────────
    if not rough:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # Head pose command range curriculum — same per-joint widening as the
    # velocity/standup envs (5% → 100% of each joint's reachable delta).
    cfg.curriculum["head_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "head_pose",
            "range_stages": [
                {"step": 0,         "ranges": ((-0.05, 0.05),  (-0.05, 0.05),  (-0.07, 0.07),  (-0.015, 0.015))},
                {"step": 500 * 24,  "ranges": ((-0.17, 0.17),  (-0.17, 0.17),  (-0.21, 0.21),  (-0.047, 0.047))},
                {"step": 1000 * 24, "ranges": ((-0.39, 0.39),  (-0.39, 0.39),  (-0.49, 0.49),  (-0.11, 0.11))},
                {"step": 1500 * 24, "ranges": ((-0.72, 0.72),  (-0.72, 0.72),  (-0.91, 0.91),  (-0.20, 0.20))},
                {"step": 2000 * 24, "ranges": ((-1.10, 1.10),  (-1.10, 1.10),  (-1.40, 1.40),  (-0.31, 0.31))},
            ],
        },
    )

    # CoM-randomization range curricula — match velocity (trunk capped at ±15 mm,
    # head at ±10 mm, per the 2026-07 audit).
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

    # Push curriculum — delayed significantly (sit env lesson): a push
    # mid-transition tips the robot into configurations it can't recover from
    # before the motions have consolidated; early pushes made the sit policy
    # unlearn sitting and converge to "just stand doing nothing".
    if ENABLE_VELOCITY_PUSHES:
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={
                "event_name": "push_robot",
                "push_stages": [
                    {"step": 0,         "velocity_range": {"x": (0.0, 0.0),    "y": (0.0, 0.0)}},
                    {"step": 1000 * 24, "velocity_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)}},
                    {"step": 1500 * 24, "velocity_range": {"x": (-0.10, 0.10), "y": (-0.10, 0.10)}},
                    {"step": 2000 * 24, "velocity_range": {"x": (-0.20, 0.20), "y": (-0.20, 0.20)}},
                    {"step": 2500 * 24, "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}},
                ],
            },
        )

    # action_rate curriculum — velocity's exact ramp (-0.1 → -1.0 by iter 1500).
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

    # Descent-speed cap tightening: discover the sit under magnitude 10
    # (crash-sit already net-negative), then tighten to 20. POSITIVE weights —
    # the function is self-negating (see the sign-convention warning at the
    # reward definitions).
    cfg.curriculum["descent_speed_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "descent_speed",
            "weight_stages": [
                {"step": 0,          "weight": 10.0},
                {"step": 500 * 24,   "weight": 20.0},
            ],
        },
    )

    # Rise-speed cap — introduced only AFTER the rise motion exists (the
    # standup attempt-tax lesson: any motion-tax during discovery makes
    # exploratory attempts net-negative and the skill is never found).
    # Pushed 750/1250 → 1500/2500: the rise needs a brief dynamic burst to
    # rock over the heels (vz > 0.08 for a few steps), and the first
    # sign-fixed run stalled in a head-down forward fold — a half-finished
    # rise — consistent with the cap taxing the final weight shift while it
    # was still being consolidated. Sit-direction gentleness doesn't depend
    # on this cap (descent_speed covers it), so late is cheap. If the rise
    # degrades when this kicks in, soften the final stage — never earlier.
    cfg.curriculum["rise_speed_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "rise_speed",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 1500 * 24,  "weight": 5.0},
                {"step": 2500 * 24,  "weight": 10.0},
            ],
        },
    )

    # Torque-rate anti-jitter — phased in once both transition motions exist.
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 750 * 24,   "weight": -5e-4},
                {"step": 1250 * 24,  "weight": -1e-3},
            ],
        },
    )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckSitStandRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # matches velocity; normalizer MUST be baked into ONNX by export.py
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
    experiment_name="microduck_sitstand",
    run_name="microduck_sitstand",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=15_000,
)
