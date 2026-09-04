"""Microduck *stand* task (v1.5) — specialized: sitting pose → standing.

Episodic policy that gently rises from the sitting keyframe to the standing
keyframe. Companion to the sit env — together they form a clean sit↔stand
pair, each policy doing one direction.

Reset:  sitting keyframe (trunk z ≈ 0.07, knees/ankles bent, head at HOME).
Target: standing keyframe (trunk z ≈ 0.12, HOME joints).
Reward design (mirror of sit env): a single fixed target is rewarded from
t=0 to end of episode; gentleness is enforced via |a_z| only; smoothness is
enforced by the usual sim2real regularisers. No trajectory waypoints, no
episode-progress gating — the policy is free to discover its own rise path.

Body control (reintroduced 2026-07-29): once standing, the policy tracks a
commanded trunk delta [z, roll, pitch] from the nominal stand (the real
body_pose command in the previously zero-padded 6D obs slot). Kicks in at
iter 2500 via the body-control curricula at the bottom of this file, after
the ground_state_mix recovery curriculum has finished ramping.
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
COM_RANDOMIZATION_RANGE             = 0.003           # ramped to 0.015 via com_range curriculum (velocity's 2026-07 audit cap; was 0.02 here)
HEAD_COM_RANDOMIZATION_RANGE        = 0.003           # ramped to 0.01 via head_com_range curriculum
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)    # unused (kp DR off)
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)      # unused (kd DR off)
VELOCITY_PUSH_INTERVAL_S            = (3.0, 6.0)
# Match velocity's ±0.3 (velocity was itself softened from ±0.5 in the 2026-07
# audit). The push curriculum below still ramps 0 → ±0.08 → this final value so
# the sit-rise bootstrap isn't shoved around from step 0 (velocity pushes at
# full strength from step 0, but it starts standing, not seated/prone).
VELOCITY_PUSH_RANGE                 = (-0.3, 0.3)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0  # match velocity (was 2.0 — pre-audit value; real IMU has ~5° systematic pitch error + estimator drift, 2° trained too narrow a band)

# Episode length: long enough for a gentle rise + brief stabilisation.
EPISODE_LENGTH_S = 6.0

# ── Sitting source pose (asset.data.joint_pos index → angle in rad) ───────────
# Must match the *actual end-state* of the sit policy. Mirrors the sit env's
# SITTING_TARGET_OVERRIDES (microduck_sit_env_cfg.py) — the swept stable
# equilibrium pose (knee ±1.35 ≈ 77°, hip_pitch ∓0.4079 = slight fwd lean,
# ankles 0). Keep the two in sync: this reset IS the sit→stand hand-off.
# Neck/head intentionally omitted → reset stays at HOME so the standup policy
# starts from exactly where the sit policy converges.
# Articulation joint indices under mjlab 1.3.0 + canonical BAM. The passive jaw
# joints are NO LONGER part of the articulation (excluded from qpos), so the
# layout is the clean 14-joint order: 0-4 left leg, 5-8 neck/head, 9-13 right leg.
# (Previously passive_1/passive_2 sat at 9,10 and shifted the right leg to 11-15.)
SITTING_JOINT_OVERRIDES = {
    1:   0.0,      # left  hip_roll   (HOME -0.0873)
    2:  -0.4079,   # left  hip_pitch  (HOME -0.4579; +0.05 = slight fwd lean)
    3:   1.35,     # left  knee       (HOME -0.0049)
    4:   0.0,      # left  ankle      (HOME +0.4530)
    10:  0.0,      # right hip_roll   (HOME +0.0873)
    11:  0.4079,   # right hip_pitch  (HOME +0.4579)
    12: -1.35,     # right knee       (HOME +0.0049)
    13:  0.0,      # right ankle      (HOME -0.4530)
}

_LEG_JOINTS  = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]

# Trunk height targets (m).
# SIT_Z matches the sit env's measured seated equilibrium (trunk z at rest in
# the swept stable pose above). Was 0.07 (old robot); keep in sync with
# microduck_sit_env_cfg.py.
SIT_Z = 0.060
# STAND_Z = empirically-measured trunk z at the natural standing equilibrium
# (HOME joint pose, vertical trunk). Previously was 0.120 — 5 mm above
# what's mechanically reachable at HOME — which forced the policy into a
# back-lean compromise to satisfy the impossible height target. Measured
# via the velocity policy holding the robot still at zero command: 115 mm.
STAND_Z = 0.115

# ── Body pose command (reintroduced 2026-07-29) ───────────────────────────────
# Master toggle. OFF restores the previous env exactly: no body_pose command,
# zero-padded body_command obs slot (obs stays 61D either way), no tracking
# reward, no body-control curricula (including the conflict-relax stages on
# height_stand_sharp / upright_sharp / standing_composite).
ENABLE_BODY_CONTROL = True
# 6D command slot [x, y, z, roll, pitch, yaw] for obs parity with velocity/
# velstand, but only z/roll/pitch are tracked (axis_weights below) — the same
# 3 axes as the original standup body control and the runtime interface.
# x/y/yaw stay at a tiny "alive" range forever: the policy learns to ignore
# them (they're reward-uncorrelated noise) instead of leaving dead weights.
# z range is ASYMMETRIC: STAND_Z is the natural equilibrium at HOME, so there
# is plenty of crouch below it but only ~1 cm of leg extension above it.
# Angles capped at ±15°: velocity body-control run 1 showed ±20° trains
# twitchy/overdriven tilting.
BODY_CMD_MAX_Z_DOWN  = 0.04             # m, crouch below STAND_Z
BODY_CMD_MAX_Z_UP    = 0.030             # m, extend above STAND_Z
BODY_CMD_MAX_ANGLE   = math.radians(15)  # rad, trunk pitch/roll
BODY_CMD_ALIVE_XY    = 0.005             # m, permanent x/y noise range
BODY_CMD_ALIVE_ANGLE = 0.05              # rad, stage-0 / permanent-yaw range
# Exact-zero command probability at resample: keeps the deployment idle case
# ("stand at nominal, no command") trained (velocity run-1 lesson — uniform
# sampling never produces the all-zero command).
BODY_CMD_ZERO_PROB   = 0.3

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
    BODY_POSE_CMD_RESAMPLE_S,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_standup_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck stand environment configuration (sit-keyframe start)."""

    site_names = ["left_foot", "right_foot"]

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

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

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

    # ── Rewards: minimum-viable set for an organic standup policy ────────────
    # Single fixed target (STAND = HOME pose + STAND_Z), active from t=0. No
    # trajectory, no waypoints, no episode-progress gating. The policy is free
    # to discover any rise path that satisfies:
    #   (1) end-state matches the HOME pose + STAND_Z
    #   (2) rise is gentle (low |a_z| throughout)
    #   (3) trunk stays upright throughout (failure mode: tip backward while
    #       extending legs; no "low z is safe" regime as in sit)
    #   (4) joint/action motion stays smooth (sim2real regularisers)
    #
    # 2026-07 TRANSFER FIX (violent/shaky on the real robot): ALL task weights
    # below divided by 4 (8→2, 30→7.5, 15→3.75, …) so the total task mass
    # (~12) matches velocity's (~11) and the shared sim2real regularisers act
    # at the same RELATIVE strength as in the well-transferring velocity env.
    # Previously the task mass was ~49, so nominally-identical regulariser
    # weights were effectively ~4× weaker here → jitter/limit-cycle around the
    # standing point was nearly free. Internal ratios between task terms are
    # unchanged (uniform scaling), so the per-term rationale comments below
    # still hold — just read their absolute reward numbers ×4. PPO normalises
    # advantages, so the global scale itself doesn't matter; only the
    # task↔regulariser ratio does.

    # Pose target — legs+hips+knees+ankles. target_overrides=None → HOME.
    cfg.rewards["pose_stand_legs"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=2.0,
        params={
            "std": 0.5,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,   # HOME = standing
        },
    )

    # Head pose tracking (commandable head control, like the velocity env).
    # Replaces the old pose_stand_neck reward (which pinned the neck/head to HOME)
    # — the neck/head are now steered by the head_pose command instead. Removed
    # from pose_stand_l1 / standing_composite below for the same reason, so no
    # reward fights head_pose_tracking's gradient.
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=0.75,
        params={"command_name": "head_pose", "std": 0.5},
    )

    # Head DC-droop penalty (velocity's fix, standup-adapted). L1 on a 1 s EMA
    # of the head tracking error — prices only the sustained gravity sag the
    # policy can cancel by biasing the neck command up; transient motion
    # averages out. TWO standup-specific safeties, both mandatory here:
    #  - UPRIGHT GATE (same values as arrival_damping): the gate multiplies the
    #    error feeding the EMA, so the ground/rising phase accumulates NOTHING
    #    — no reward wall at the finish line, no tax on the head-pivot flip
    #    (the retired head_impact_penalty froze the policy exactly that way).
    #  - STARTS AT 0, introduced at iter 3000 by the curriculum below — same
    #    discovery-vs-refinement timing as arrival_damping/torque_rate.
    cfg.rewards["head_pose_bias"] = RewardTermCfg(
        func=microduck_mdp.head_pose_bias_penalty,
        weight=0.0,  # ramped by head_pose_bias_weight curriculum
        params={
            "command_name":       "head_pose",
            "tau_s":              1.0,
            "gate_height_low":    0.09,
            "gate_height_high":   0.11,
            "gate_tilt_full_deg": 20.0,
            "gate_tilt_zero_deg": 45.0,
        },
    )

    # L1 bootstrap — constant gradient even when far from HOME.
    # Bumped 2 → 5: at convergence the policy parks ~0.18 rad off-HOME (mostly
    # bent knees) costing only -0.35/step at weight 2 — cheap enough to ignore.
    # At weight 5 that error costs -0.9/step, forcing the policy to actually
    # close the gap on the remaining joints.
    cfg.rewards["pose_stand_l1"] = RewardTermCfg(
        func=microduck_mdp.pose_l1_penalty,
        weight=1.25,
        params={
            # Legs only — neck/head are steered by head_pose_tracking.
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,
        },
    )

    # Trunk height target — two-layer Gaussian to get both bootstrap reach
    # AND a sharp peak at STAND_Z.
    #  - ``height_stand``: wide std (0.04), for the bootstrap pull from sit.
    #  - ``height_stand_sharp``: narrow std (0.015), creates a strong gradient
    #    in the final cm. Earlier runs converged at z ≈ 0.109 because the
    #    wide-std Gaussian was already saturated (0.93/1.0) — no gradient to
    #    pull the last cm. The sharp layer adds 0.36→1.0 reward jump in that
    #    same range, ~3× the marginal pull.
    cfg.rewards["height_stand"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=1.0,
        params={
            "std":           0.04,
            "target_height": STAND_Z,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["height_stand_sharp"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=1.0,
        params={
            "std":           0.015,
            "target_height": STAND_Z,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    # L1 bumped 10 → 30: previous run plateaued sitting still because the
    # static-sit basin (-0.5 reward from L1 + everything else positive) was
    # net positive. At weight 30, sitting still costs -1.5/step — net cost
    # of "stay sitting" forces exploration.
    cfg.rewards["height_stand_l1"] = RewardTermCfg(
        func=microduck_mdp.height_l1_penalty,
        weight=7.5,
        params={
            "target_height": STAND_Z,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Reward upward CoM velocity below STAND_Z — pays for the *motion* of
    # rising, not just for the destination. Critical bootstrap: with only
    # destination rewards, "stay sitting upright collecting most-of-pose +
    # upright" was the dominant local optimum. Rewarding vz > 0 directly
    # makes any rise attempt immediately positive. Gates off above
    # max_height so the policy can't farm it by bobbing.
    # max_height set just above STAND_Z (0.12 → 0.125) so the reward stays
    # active through the final cm of rise. Earlier 0.11 caused the policy to
    # park at ~0.108 (gate-off altitude) and never finish the climb.
    # NO max_vz cap (reverted 2026-07-24, second broken run): capping the
    # rewarded rise speed — even at a generous 0.30 — shrinks the payoff of
    # noisy recovery ATTEMPTS during the discovery phase, and face-up/face-down
    # recovery never got learned. Both broken runs shared the same wandb
    # signature regardless of cap value (0.15 or 0.30) and gate tuning:
    # standing metrics drop at the ground_state_mix stages (1500/2500) instead
    # of recovering like the reference run. Smoothing is now done by the
    # LATE-phased penalty curricula below instead (see arrival_damping /
    # smoothness_polish comments).
    cfg.rewards["com_upward_velocity"] = RewardTermCfg(
        func=microduck_mdp.com_upward_velocity,
        weight=0.75,
        params={
            "asset_cfg":  SceneEntityCfg("robot", body_names=("trunk_base",)),
            "max_height": 0.125,
        },
    )

    # Gentle rise — penalty on |a_z|. Compatible with com_upward_velocity:
    # constant positive vz collects upward-velocity reward AND has a_z = 0,
    # so the two pressures together select for smooth constant-velocity rise.
    # NOTE this term is GLOBAL (not phase-gated): prone flips pay it in full
    # (impacts + push-off are |a_z| spikes). The 2026-07-24 attempt to double
    # it to -0.01 contributed to the face-up freeze; -0.005 is the ceiling
    # unless it gets a height/tilt gate like arrival_damping.
    # ⚠️ POSITIVE weight: trunk_vertical_accel_penalty ALREADY returns -|a_z|.
    # The previous -0.005 double-negated into a (small) reward for vertical
    # shocks — the same sign bug roller_standup found and fixed in its
    # gentle_rise, confirmed again on the sitstand run 7ev90yd9 (its
    # Episode_Reward/gentle_motion logged POSITIVE). Keep magnitude small:
    # |a_z| is unavoidable during prone flips, a big weight is a motion-blocker.
    cfg.rewards["gentle_rise"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.005,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # Arrival damper — trunk ω_xy², gated on height AND tilt (zero above 45°
    # tilt / below 0.09 m, full below 20° tilt / above 0.11 m). Targets the
    # real-robot failure loop: rise → overshoot vertical → tip → retry.
    #
    # STARTS AT WEIGHT 0 — introduced at iter 3000 by the arrival_damping
    # curriculum below. Two broken runs (2026-07-24) proved that ANY
    # attempt-tax active during the recovery DISCOVERY phase (ground_state_mix
    # ramps face-down/face-up until iter 2500) prevents the flip from being
    # found at all: exploration of the hard poses is noisy thrash, taxing it
    # makes attempts net-negative, "do nothing" wins. Gate refinement (tilt
    # gating, halved weight, generous vz cap) did NOT change the failure
    # signature — the fix is timing, not magnitude. From iter 3000 the skills
    # already exist and keep being exercised by prone resets, so the damping
    # fine-tunes their execution instead of blocking their discovery.
    cfg.rewards["arrival_damping"] = RewardTermCfg(
        func=microduck_mdp.body_ang_vel_at_height,
        weight=0.0,
        params={
            "height_low":    0.09,
            "height_high":   0.11,
            "tilt_full_deg": 20.0,
            "tilt_zero_deg": 45.0,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Upright — two-layer like the height reward.
    #  - ``upright_linear``: cos(tilt). Strong gradient at high tilt (e.g.,
    #    while inverted at the start of a recovery), weak near vertical.
    #    Provides bootstrap pull from any orientation.
    #  - ``upright_sharp``: exp(-tilt²/std²) with std ≈ 6°. Gradient is
    #    STRONGEST in the near-vertical regime where the linear version
    #    runs out of steam. Previous run converged at ~37° back-lean because
    #    the linear pull at small tilt becomes weak; this term punishes that
    #    exact regime.
    cfg.rewards["upright_linear"] = RewardTermCfg(
        func=microduck_mdp.body_upright_linear,
        weight=1.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    # Sharp Gaussian upright, gated by trunk z. Pays only when the robot is
    # actually at the standing height — prevents the "crouch low and vertical"
    # exploit. Broadened std 0.1 → 0.3 (≈17°): too sharp before, scored
    # near-zero at the lean basin (no gradient). With 0.3, the lean basin
    # at z=0.111 (smoothstep ~0.91) and tilt 37° (gaussian ~0.11) scores
    # ~0.1 = visible gradient that pulls toward vertical.
    cfg.rewards["upright_sharp"] = RewardTermCfg(
        func=microduck_mdp.upright_gaussian_at_height,
        weight=1.5,
        params={
            "std":         0.3,
            "height_low":  SIT_Z,
            "height_high": STAND_Z,
            "asset_cfg":   SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Smooth multiplicative goal-state score (broad stds).
    # The previous tight stds (height=0.015, upright=0.15, pose=0.20) had
    # the composite at ~5e-5 at the lean basin — invisible to the policy,
    # zero gradient. Broadening so the lean basin scores ~0.2 (visible
    # gradient) while the goal still scores ~1.0 (clear attractor).
    cfg.rewards["standing_composite"] = RewardTermCfg(
        func=microduck_mdp.standing_composite_score,
        weight=3.75,
        params={
            "target_height":    STAND_Z,
            "height_std":       0.04,    # 4cm — broad, covers the climb
            "upright_std":      0.40,    # ≈ 23° — lean basin scores ~0.3
            "pose_std":         0.40,    # joint-RMS, broad enough for partial pose
            "joint_indices":    _LEG_JOINTS,   # neck/head steered by head_pose_tracking
            "target_overrides": None,
            "asset_cfg":        SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Body pose tracking — z/roll/pitch only (axis_weights), the runtime
    # body-control axes. Locomotion variant (not body_pose_tracking_6d) so the
    # unused x/y axes wouldn't reference the spawn origin, which the robot
    # leaves during prone flips. Weight starts at 0; body_pose_tracking_weight
    # ramps it in from iter 2500 (after ground_state_mix finishes) so recovery
    # discovery is untouched. While prone/rising the reward is ≈0 on all
    # tracked axes, so before the robot stands it is just another standing
    # attractor — unlike motion penalties, it can't tax flip/rise attempts.
    # Tight stds on purpose (standup phase-2 lesson): at 1 cm z error with
    # z_std=0.01 the axis reward drops to 0.37 (real gradient); 0.02 → 0.78.
    if ENABLE_BODY_CONTROL:
        cfg.rewards["body_pose_tracking"] = RewardTermCfg(
            func=microduck_mdp.body_pose_tracking_locomotion,
            weight=0.0,
            params={
                "command_name": "body_pose",
                "nominal_height": STAND_Z,
                "z_std": 0.01,
                "angle_std": math.radians(5),
                "axis_weights": (0.0, 0.0, 1.0, 1.0, 1.0, 0.0),
                "vel_gate_command_name": None,
            },
        )

    # ── Sim2real regularisers — MATCHED to velocity (2026-07) ───────────────
    # velocity's exact set and absolute weights:
    #   • action_rate_l2: -0.1 at stage 0, ramped -0.1 → -1.0 by iter 1500
    #     (action_rate_weight curriculum below, velocity's exact stages)
    #   • body_ang_vel -0.05, angular_momentum -0.02
    #   • microduck-only extras DROPPED, like velocity drops them:
    #     neck_action_rate_l2, joint_torques_l2, joint_torque_rate_l2, soft_landing
    # Parity is made REAL by the ÷4 task-stack scaling above — previously the
    # same absolute weights were ~4× weaker relative to the ~49 task mass.
    #
    # HISTORY / RISK: at the OLD task scale, raising body_ang_vel to -0.15 and
    # the action_rate end to -1.2 killed back-recovery (both are motion-blockers
    # for the flip). At the new ÷4 scale, body_ang_vel -0.05 ≈ -0.2 old-units —
    # WATCH face-down/face-up recovery as ground_state_mix ramps them in
    # (iters 600–2500). If recovery freezes: halve body_ang_vel to -0.025
    # first, then soften the action_rate curriculum end to -0.6.
    #
    # 2026-07 smoothness polish (rise violent + overshoot-retry loop on the
    # real robot after the ÷4 rescale): joint_torque_rate_l2 (anti-jitter:
    # penalizes torque CHANGE, not magnitude/rotation) + arrival_damping
    # (rewards block above). BOTH start at weight 0 and are introduced at iter
    # 3000 by the smoothness-polish curricula below — the reward set is
    # IDENTICAL to the working 2026-07-23 run until then. See the
    # arrival_damping comment for why timing (discovery vs fine-tuning), not
    # magnitude, is what decides whether these terms break recovery.
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=0.0
    )

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05      # motion-blocker: kept LIGHT (velocity value)
    cfg.rewards["angular_momentum"].weight = -0.02  # velocity value
    cfg.rewards.pop("soft_landing", None)           # velocity removes it

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # Drop only the base "upright" Gaussian — standup uses its own
    # upright_linear/upright_sharp instead. (angular_momentum kept above to match
    # velocity; soft_landing/hip_yaw_roll_deviation dropped to match velocity.)
    if "upright" in cfg.rewards:
        del cfg.rewards["upright"]

    # ── Observations (identical layout to walking / sit policies) ─────────────
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    # mjlab 1.3.0 base template adds sensor-based foot_height + height_scan obs.
    # Standup has no terrain-height sensor (and drops the walking foot rewards),
    # so remove these terms. foot_air_time/foot_contact(_forces) use the
    # feet_ground_contact sensor, which standup does define, so they stay.
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    # The retained sensor-derived critic terms get the NaN-safe wrappers: a
    # non-finite contact force slips past robot_state_is_nan (it checks joint +
    # root state only) and a single NaN here kills the run via rsl_rl's
    # check_nan — the 2026-08-21 Velocity2-Rough-Backlash crash. Standup lands
    # and flips constantly, so degenerate contacts are MORE likely here.
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

    # IMU obs delay: max_lag 1 (was 3 = 60 ms worst case) — match velocity's
    # 2026-07 audit value; the real dxl IMU path is fast (±20 ms envelope).
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

    # Encoder-bias DR (match velocity): actor sees joint_pos + per-env bias; critic
    # keeps the true joint pos. Requires the base-template encoder_bias event.
    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # ── Head pose command (commandable head control, like the velocity env) ───
    # 4D deltas-from-HOME on neck/head joints: [neck_pitch, head_pitch, head_yaw,
    # head_roll]. Tracked by head_pose_tracking below; ranges widened by the
    # head_pose_range curriculum. Same per-joint caps as the velocity env.
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.05, 0.05),    # neck_pitch
            (-0.05, 0.05),    # head_pitch
            (-0.07, 0.07),    # head_yaw
            (-0.015, 0.015),  # head_roll
        ),
    )

    # ── Body pose command (6D delta from nominal standing) ───────────────────
    # [x, y, z, roll, pitch, yaw]. Only z/roll/pitch are tracked (see
    # body_pose_tracking below); x/y/yaw are permanent alive-range noise.
    # Ranges start tiny; the body_pose_range curriculum widens z/roll/pitch
    # once the recovery skills exist (ground_state_mix final at 2500).
    if ENABLE_BODY_CONTROL:
        cfg.commands["body_pose"] = microduck_mdp.UniformPoseCommandCfg(
            resampling_time_range=BODY_POSE_CMD_RESAMPLE_S,
            zero_command_prob=BODY_CMD_ZERO_PROB,
            ranges=(
                (-BODY_CMD_ALIVE_XY, BODY_CMD_ALIVE_XY),        # x (m)
                (-BODY_CMD_ALIVE_XY, BODY_CMD_ALIVE_XY),        # y (m)
                (-0.005, 0.005),                                # z (m)
                (-BODY_CMD_ALIVE_ANGLE, BODY_CMD_ALIVE_ANGLE),  # roll
                (-BODY_CMD_ALIVE_ANGLE, BODY_CMD_ALIVE_ANGLE),  # pitch
                (-BODY_CMD_ALIVE_ANGLE, BODY_CMD_ALIVE_ANGLE),  # yaw
            ),
        )

    # Command obs slots. head_command is the real head_pose command; the
    # body_command slot carries the real body_pose command when body control is
    # enabled, and zero padding otherwise (obs shape identical either way).
    # Layout parity with velocity/velstand: [twist(3), head_pose(4), body_pose(6)].
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "head_pose"},
        )
        if ENABLE_BODY_CONTROL:
            cfg.observations[group].terms["body_command"] = ObservationTermCfg(
                func=mdp.generated_commands, params={"command_name": "body_pose"},
            )
        else:
            cfg.observations[group].terms["body_command"] = ObservationTermCfg(
                func=microduck_mdp.zero_command_padding, params={"dim": 6},
            )

    # ── Command: tiny noise around zero (kept for obs-shape parity) ──────────
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
    # Robot starts seated — tilt-based fall termination doesn't apply here.
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": ("feet_ground_contact",)},
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

    # Start in the sitting keyframe with noise on joints + trunk tilt. Real
    # deployment hand-off from the sit policy won't reproduce the SIT
    # keyframe exactly — the standup policy must be robust to a band of
    # plausible "sit-ish" starts. Without noise the policy was overfitting
    # to the exact canonical SIT pose.
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            # Initialize from any pose, 25% each: front (face-down), back
            # (face-up), sitting keyframe, and already-standing (so the policy
            # also learns to *hold* a stand, not only to rise).
            # Initial mix = curriculum stage 0 (easy); the ground_state_mix
            # curriculum ramps these easy→hard over training. Face-up (back) starts
            # at 0 and is introduced late (hardest recovery).
            "face_down_prob":            0.20,  # belly to floor (+90° pitch)
            "face_up_prob":              0.00,  # back to floor (-90° pitch) — introduced late
            "sitting_prob":              0.40,  # sit keyframe (deployment hand-off)
            "standing_prob":             0.40,  # already upright at standing height
            # Prone reset height: trunk rests at ~0.044 m face-down (measured), so
            # spawn just above the ground rather than the 0.20–0.25 default (which
            # would free-fall ~15 cm before landing).
            "prone_z_min":               0.05,
            "prone_z_max":               0.09,
            # Partial-roll noise on face-up spawns (±90° about the body long
            # axis): back-recovery was seed-lucky (1 success / 3 failures with
            # equivalent rewards) because the reward landscape from flat
            # supine to prone is flat — no gradient until the roll completes.
            # Near-on-side spawns put starts partway along the roll → built-in
            # reverse curriculum. See set_random_ground_state in mdp.py.
            "face_up_roll_max":          math.radians(90),
            "sitting_joint_overrides":   SITTING_JOINT_OVERRIDES,
            "sitting_joint_noise_std":   0.12,           # ≈ 7° per joint
            "sitting_tilt_max":          math.radians(10),  # ±10° pitch/roll
            # Seated equilibrium is SIT_Z=0.060 — band is −1cm/+3cm around it
            # (same spread as when equilibrium was 0.07 with 0.06–0.10).
            "sitting_z_min":             0.05,
            "sitting_z_max":             0.09,
            # Standing init: trunk just above the measured equilibrium (STAND_Z=0.115).
            "standing_z_min":            0.11,
            "standing_z_max":            0.12,
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
        # Match velocity: randomize the CoM of the head-assembly bodies.
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
        # Match velocity: reflected rotor inertia (non-accumulating, affects BAM).
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
        # (alpha scales both by e^(2α), CoM untouched). Startup mode. The old
        # custom randomize_mass_and_inertia was a no-op under mjlab 1.3.0.
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

    # NOTE: IMU mounting-misalignment is applied at the OBSERVATION level below
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

    # Init-pose curriculum: ramp the set_ground_state mix from EASY → HARD instead
    # of a flat 25/25/25/25 from step 0. With the flat split the policy optimized
    # the easy majority (hold-stand + sit-rise) and left the hard poses under-
    # trained — front only partially rose and face-up (back) froze into "do
    # nothing". This introduces standing/sitting first, then face-down, then
    # face-up last, and biases toward the hard poses late so they get the most
    # practice. (event_param_curriculum shallow-merges these keys into the live
    # set_ground_state event; the z-ranges / joint overrides are left untouched.)
    cfg.curriculum["ground_state_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_ground_state",
            "param_stages": [
                # step,          standing, sitting, face_down(front), face_up(back)
                {"step": 0,          "params": {"standing_prob": 0.40, "sitting_prob": 0.40, "face_down_prob": 0.20, "face_up_prob": 0.00}},
                {"step": 600 * 24,   "params": {"standing_prob": 0.25, "sitting_prob": 0.30, "face_down_prob": 0.35, "face_up_prob": 0.10}},
                {"step": 1500 * 24,  "params": {"standing_prob": 0.20, "sitting_prob": 0.25, "face_down_prob": 0.30, "face_up_prob": 0.25}},
                {"step": 2500 * 24,  "params": {"standing_prob": 0.15, "sitting_prob": 0.20, "face_down_prob": 0.30, "face_up_prob": 0.35}},
            ],
        },
    )

    # Head pose command range curriculum — same per-joint widening as the velocity
    # env (5% → 100% of each joint's reachable delta from HOME over ~2000 iters).
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

    # NOTE: the earlier head_pose_std / head_pose_weight curricula (band-aids for
    # the head-droop) were removed — the droop was a backward-CoM balance crutch,
    # fixed at the source by the STAND2 forward-shifted standing pose. head_pose
    # tracking stays at its baseline (weight 3.0, std 0.5) + head_pose_range.

    # CoM-randomization range curricula — match velocity (ramp 0.003 → 0.015 trunk,
    # 0.003 → 0.01 head over the first ~1500 / ~1000 iters). Trunk capped at
    # ±15 mm per velocity's 2026-07 audit: beyond that the randomized CoM can
    # leave the foot support polygon entirely, which trains hyper-reactive
    # correction. (The old 0.02 final stage here exceeded that cap.)
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

    # action_rate curriculum — velocity's exact ramp (-0.1 → -1.0 by iter 1500).
    # Gentler early stages than the old -0.4/-0.8/-1.0-by-500 ramp: the rise
    # skill gets discovered under light smoothing, then damping tightens.
    # (Old note, still relevant: a -1.2 end once blocked back-recovery; -1.0 is
    # the ceiling. With the ÷4 task scale this -1.0 now actually bites.)
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

    # Smoothness-polish curricula — introduce the anti-violence terms only
    # AFTER the recovery skills exist. ground_state_mix finishes ramping the
    # hard poses at iter 2500; from 3000 on, prone resets keep exercising the
    # learned flips while these penalties fine-tune their execution (brake at
    # arrival, less jitter). Two runs proved the same weights active from
    # step 0 prevent the flips from ever being DISCOVERED (attempt-tax on
    # exploration). If recovery degrades after 3000, soften the last stage,
    # do NOT move the introduction earlier.
    cfg.curriculum["arrival_damping_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "arrival_damping",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 3000 * 24,  "weight": -0.025},
                {"step": 4000 * 24,  "weight": -0.05},
            ],
        },
    )
    # head_pose_bias: same introduction timing as arrival_damping (see its
    # comment — timing, not magnitude, is what protects recovery discovery).
    # Dosage: standup runs head_pose_tracking at 0.75 vs velocity's 2.0 (task
    # weights ÷4 rebalance), so the bias lands at 1.5 vs velocity's 3.0. At
    # 1.5 a 15° standing droop costs 0.39/step, 5° costs 0.13/step. If the
    # standing head is still down after a run, raise the last stage — do NOT
    # move the introduction earlier.
    cfg.curriculum["head_pose_bias_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "head_pose_bias",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 3000 * 24,  "weight": 0.5},
                {"step": 4000 * 24,  "weight": 1.5},
            ],
        },
    )
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 3000 * 24,  "weight": -1e-3},
            ],
        },
    )

    # ── Body-control curricula ────────────────────────────────────────────────
    # Everything below is body-control only — NOTE the early return; add any
    # unrelated cfg above this line.
    if not ENABLE_BODY_CONTROL:
        return cfg

    # Tracking weight ramps in at 2500 — exactly when ground_state_mix reaches
    # its final (hardest) mix, so the recovery-discovery phase trains without
    # any body-command pressure. Final weight 4.0: at full command the fixed-
    # stand terms oppose tracking by ~2/step AFTER the relax stages below, and
    # tracking's marginal gain is ~0.65/step per unit weight → 4.0 wins with
    # margin. (Without the relax stages the opposition is ~4.3/step and even
    # the old design's weight 5 loses — the phase-2 lesson.)
    cfg.curriculum["body_pose_tracking_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "body_pose_tracking",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 2500 * 24,  "weight": 1.5},
                {"step": 3000 * 24,  "weight": 3.0},
                {"step": 4000 * 24,  "weight": 4.0},
            ],
        },
    )

    # Command range widening, synced to the weight ramp. x/y/yaw stay at their
    # alive ranges (untracked); only z/roll/pitch widen. z asymmetric — see the
    # BODY_CMD constants block.
    _alive_xy  = (-BODY_CMD_ALIVE_XY, BODY_CMD_ALIVE_XY)
    _alive_ang = (-BODY_CMD_ALIVE_ANGLE, BODY_CMD_ALIVE_ANGLE)
    cfg.curriculum["body_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "body_pose",
            "range_stages": [
                # ranges = (x, y, z, roll, pitch, yaw)
                {"step": 0, "ranges": (
                    _alive_xy, _alive_xy, (-0.005, 0.005),
                    _alive_ang, _alive_ang, _alive_ang,
                )},
                {"step": 2500 * 24, "ranges": (
                    _alive_xy, _alive_xy, (-0.010, 0.005),
                    (-math.radians(8), math.radians(8)),
                    (-math.radians(8), math.radians(8)),
                    _alive_ang,
                )},
                {"step": 3000 * 24, "ranges": (
                    _alive_xy, _alive_xy, (-0.018, 0.008),
                    (-math.radians(12), math.radians(12)),
                    (-math.radians(12), math.radians(12)),
                    _alive_ang,
                )},
                {"step": 4000 * 24, "ranges": (
                    _alive_xy, _alive_xy,
                    (-BODY_CMD_MAX_Z_DOWN, BODY_CMD_MAX_Z_UP),
                    (-BODY_CMD_MAX_ANGLE, BODY_CMD_MAX_ANGLE),
                    (-BODY_CMD_MAX_ANGLE, BODY_CMD_MAX_ANGLE),
                    _alive_ang,
                )},
            ],
        },
    )

    # Conflict relax — the standup phase-2 lesson applied to THIS reward set:
    # the sharp fixed-stand attractors directly out-bid commanded deviations
    # (at Δz=−2cm/15° tilt: height_stand_sharp −0.83, upright_sharp −0.79,
    # standing_composite −1.9 per step). Their bootstrap/polish job is done by
    # 3000; body_pose_tracking at cmd=0 (30% of resamples) takes over the
    # "sharp peak at nominal stand" role with even tighter stds. The broad
    # bootstrap layers (height_stand, upright_linear, height_stand_l1,
    # pose_stand_*) are left untouched — they're what recovery leans on, and
    # their opposition at full command is mild (~0.9/step total). Standing-
    # attractor mass is roughly conserved: 6.25 before → 2.2 + tracking 4.0.
    cfg.curriculum["height_stand_sharp_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "height_stand_sharp",
            "weight_stages": [
                {"step": 0,          "weight": 1.0},
                {"step": 3000 * 24,  "weight": 0.5},
                {"step": 4000 * 24,  "weight": 0.2},
            ],
        },
    )
    cfg.curriculum["upright_sharp_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "upright_sharp",
            "weight_stages": [
                {"step": 0,          "weight": 1.5},
                {"step": 3000 * 24,  "weight": 1.0},
                {"step": 4000 * 24,  "weight": 0.5},
            ],
        },
    )
    cfg.curriculum["standing_composite_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "standing_composite",
            "weight_stages": [
                {"step": 0,          "weight": 3.75},
                {"step": 3000 * 24,  "weight": 2.5},
                {"step": 4000 * 24,  "weight": 1.5},
            ],
        },
    )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckStandUpRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="microduck_stand",
    run_name="microduck_stand",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=15_000,
)
