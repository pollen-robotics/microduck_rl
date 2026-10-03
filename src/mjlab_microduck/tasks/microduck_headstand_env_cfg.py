"""Microduck headstand: the fold, the kick-ups and the split switch.

Episodic tricks, triggered at deployment like standup/roulade (a policy
switch starts the entry). Three policies come from this cfg:

  fold      standing → the pike (head top and both feet on the floor, trunk
            at 76°), the state every kick-up starts from.
  kick-up   the pike → a headstand, held: legs together ("legs_together") or in a
            split with the left leg forward ("split"). The foot pushes off,
            the neck swings the trunk up, the legs balance.
  switch    the split hold → the mirrored split, on a flag in the twist vx
            slot (kick-up "split" with switch=True).

The entry is an acrobat's: head to the floor, one leg extended into a deep
split, a small push from the last foot on the ground.

The measured physics (the hold pose, the pike, the swing) is in the headstand
section of mdp.py, next to the functions that encode it.

Design, following AGENTS.md and the standup/roulade lessons:
  • ONE dense swing signal: potential-based Δ(inverted_cos), support-gated,
    paid at the frontier only (rocking is a wash, ballistic flips pay nothing).
  • The hold is a product of Gaussians (height × inverted × pose) paid only
    with the head latched on the floor and both feet in the air. A flop
    scores ~0, a foot on the floor is gated out. Broad stds + a sharp inverted layer.
  • Hard gates for what counts (state-based, not nudges): head-top latch,
    feet-off, airborne penalty (a jump is never the entry), other-body floor
    contact penalty (thigh/shin/trunk down = flop).
  • Reverse curriculum via the spawn mix: pikes and the hold from step 0,
    with the hold and swing shares shrinking as the kick-up appears.
  • Motion-blockers (body_ang_vel, angular_momentum) ≈ 0 during discovery;
    arrival damping / torque-rate polish introduced late by curriculum.
  • Symmetry OFF for the split (asymmetric, left leg forward), ON for the
    fold and the legs-together kick-up.

Robot model: allcollisions — the head, thighs, shins and trunk all collide
with the floor, and the sensors below need to know which one touched.
"""

import math
from copy import deepcopy

# Knee gate of the hold: both knees within KNEE_FULL rad count as straight,
# the gate is 0 past KNEE_ZERO.
KNEE_FULL, KNEE_ZERO = 0.3, 0.6
# A foot on the floor costs from 60° off inverted (inverted_cos 0.5) up to
# full cost inside the balance basin at 35° off (0.82).
FEET_DOWN_INVERTED_LO, FEET_DOWN_INVERTED_HI = 0.5, 0.82
# Arrival damping is full within ~20° of inverted (0.94), zero beyond ~45° (0.71).
DAMPING_INVERTED_FULL, DAMPING_INVERTED_ZERO = 0.94, 0.71
# A flop ends a kick-up episode after this many steps (0.5 s), so a partway
# spawn has time to settle and latch.
FLOP_GRACE_STEPS = 25
# Spawn noise: ±8° of trunk pitch on hold spawns, 0.05 rad on every servo.
HOLD_PITCH_NOISE_DEG = 8.0
SPAWN_JOINT_NOISE_STD = 0.05

# ── Domain randomisation (matched to standup/velocity for sim2real parity) ───
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = False  # a shove mid-entry is incoherent (roulade)
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

# ── Ranges (matched to the standup env) ───────────────────────────────────────
COM_RANDOMIZATION_RANGE             = 0.003   # ramped to 0.015 via curriculum
HEAD_COM_RANDOMIZATION_RANGE        = 0.003   # ramped to 0.01 via curriculum
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# Episode: a fold takes ~1.5 s, a kick-up ~1 s; 6 s leaves seconds of hold to pay.
EPISODE_LENGTH_S = 6.0

# Head contact force above HEAD_IMPACT_THRESH_N costs per step (velstand's
# body_impact_cost); above HEADSTAND_SLAM_N, at any time after the spawn's
# settle, it forfeits the hold reward for the episode. The static load in the
# hold is the whole 7.2 N body.
HEAD_IMPACT_THRESH_N = 10.0
HEADSTAND_SLAM_N = 12.0

# ── The hold pose (servo index → rad), measured ──────────────────────────────
# Hip yaw/roll, knees, ankles and head yaw/roll are 0, not HOME: this is where
# the settle test found the balance. Left hip 1.2 rad is inside the 0.9 soft
# limit (1.41). The neck at 1.0 rad is past its soft limit (0.94 of a 1.047
# range) and 2.7° from its hard stop: it costs dof_pos_limits ~0.06/step
# against a hold paying ~5/step, and every pose inside the soft limit landed
# half as often from noisy drops (9/30 vs 15/30).
HEADSTAND_OVERRIDES = {
    0:   0.0,   # left  hip_yaw
    1:   0.0,   # left  hip_roll
    2:   1.2,   # left  hip_pitch  (forward leg of the split)
    3:   0.0,   # left  knee
    4:   0.0,   # left  ankle
    5:   1.0,   # neck_pitch
    6:   1.25,  # head_pitch
    7:   0.0,   # head_yaw
    8:   0.0,   # head_roll
    9:   0.0,   # right hip_yaw
    10:  0.0,   # right hip_roll
    11:  0.8,   # right hip_pitch  (back leg of the split)
    12:  0.0,   # right knee
    13:  0.0,   # right ankle
}
# Legs together and straight up: the same neck and head, every leg joint at
# 0. No static balance exists for it, so the policy holds it actively.
HEADSTAND_LEGS_TOGETHER_OVERRIDES = {**{i: 0.0 for i in range(14)}, 5: HEADSTAND_OVERRIDES[5], 6: HEADSTAND_OVERRIDES[6]}
# trunk_base z at rest in the hold pose (measured, allcollisions model).
HEADSTAND_Z = 0.117
# Hold spawns start here: the rest z plus 3 mm, so nothing falls in.
HEADSTAND_SPAWN_Z = HEADSTAND_Z + 0.003

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

from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    HEAD_BODY_NAMES,
    HEAD_POSE_CMD_RESAMPLE_S,
    BODY_POSE_CMD_RESAMPLE_S,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_headstand_env_cfg(
    play: bool = False,
    style: str = "split",
    switch: bool = False,
    *,
    park_cost_w: float = -0.5,
    progress_w: float = 10.0,
    omega_max: float = 2.0,
    overspeed_w: float = -0.5,
    handover_prob: float = 0.5,
    polish_at: int = 2500,
    switch_ramp_s: float = 0.3,
) -> ManagerBasedRlEnvCfg:
    """Create one of the headstand environment configurations.

    style: "fold" (standing → the pike), "split" (kick-up to the split hold)
    or "legs_together" (kick-up to the legs-together hold). switch=True on the
    split adds the mirror flag and its command. `play` is accepted for the
    registry and changes nothing.

    The kick-ups start head-down in the pike or further along, never
    standing: the fold is another policy's job. A single end-to-end task
    with standing starts learns to park head-and-feet-down every time, which
    is why the entry is two policies.

    The keyword arguments default to the values the published policies were
    trained with:
      park_cost_w   weight of the always-on not-inverted cost.
      progress_w    weight of the frontier reward for the swing.
      omega_max     trunk rotation rate (rad/s) above which overspeed_w
                    applies; 2.0 gives a 0.4 s kick-up, 6.0 a 0.2 s snap.
      handover_prob     weight of the handover bucket (the fold policy's recorded
                    pikes) against partway 0.30, hold 0.30 and the measured
                    pike 0.20; 0.5 makes it 38% of kick-up episodes.
      polish_at     iteration around which the smoothness costs ramp to
                    full weight (the last rung is 500-1000 iterations after
                    it); 0 applies them from the start, for a run warm-
                    started from a policy that already has the skill.
      switch_ramp_s seconds the split switch takes to mirror the pose target.
    """
    if style not in ("fold", "split", "legs_together"):
        raise ValueError(f"style must be fold, split or legs_together, got {style!r}")
    if switch and style != "split":
        raise ValueError("switch=True is the split hold's flag; it needs style='split'")
    if park_cost_w > 0.0 or overspeed_w > 0.0:
        raise ValueError("park_cost_w and overspeed_w weight costs (>= 0 terms) and must be <= 0")
    kickup = style != "fold"
    overrides = HEADSTAND_LEGS_TOGETHER_OVERRIDES if style == "legs_together" else HEADSTAND_OVERRIDES

    # Whole ankle BODIES, not the sole geoms: each ankle carries three more
    # collision geoms (servo housing etc.) that sit 2–8 mm below the sole when
    # the ankle is pitched, and a sole-only sensor reads "feet off" while the
    # housing carries the weight.
    feet_ground_cfg = ContactSensorCfg(
        name=microduck_mdp._HEADSTAND_FEET_SENSOR,
        primary=ContactMatch(mode="body", pattern=r"^(ankle_left|ankle_right)$", entity="robot"),
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
    # The head on the floor — the latch. jaw_soft carries the head collision
    # geoms; the latch additionally requires the flat top to be the part down.
    head_ground_cfg = ContactSensorCfg(
        name=microduck_mdp._HEADSTAND_HEAD_SENSOR,
        primary=ContactMatch(mode="body", pattern="^jaw_soft$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),   # force feeds the head impact penalty
        reduce="netforce",
        num_slots=1,
    )
    # Everything that is neither head nor foot: thighs, shins, hips, trunk.
    # Any of these on the floor is a flop or a collapse. The head is three
    # bodies (jaw_soft, yaw_roll_motion, neck_pitch) and all three are exempt.
    other_ground_cfg = ContactSensorCfg(
        name=microduck_mdp._HEADSTAND_OTHER_SENSOR,
        primary=ContactMatch(
            mode="body",
            pattern=r"^(?!jaw_soft$|yaw_roll_motion$|neck_pitch$|ankle_left$|ankle_right$).*",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    # Anything at all on the floor — the support gate (roulade lesson).
    robot_ground_cfg = ContactSensorCfg(
        name=microduck_mdp._HEADSTAND_SUPPORT_SENSOR,
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_ALLCOLLISIONS_ROBOT_CFG}
    # 70 collision geoms and a fold that piles head, thighs, shins and
    # self-contacts up at once overflow the base template's nconmax 35 / 10
    # solver iterations into NaN resets. Same budget as velstand and sitstand.
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 200)
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, head_ground_cfg, other_ground_cfg, robot_ground_cfg)
    cfg.viewer.body_name = "trunk_base"
    cfg.episode_length_s = EPISODE_LENGTH_S

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: drop walking-specific terms ──────────────────────────────────
    for name in [
        "track_linear_velocity", "track_angular_velocity", "air_time",
        "foot_clearance", "foot_swing_height", "foot_slip", "pose", "upright",
    ]:
        cfg.rewards.pop(name, None)

    # ── Rewards: the headstand set ────────────────────────────────────────────
    # Task mass at the hold ≈ 5.5/step (composite 4 + sharp 1.5); the swing
    # pays ~2/step × 2 for ~1 s. That is about half of standup's ~12, so the
    # shared regularisers below are scaled by ~0.46 to act at the relative
    # strength that transferred (AGENTS.md: compare reward mass, not weights).
    switch_command = "twist" if switch else None
    hold_params = {
        "target_overrides": overrides,
        "knee_full":        KNEE_FULL,
        "knee_zero":        KNEE_ZERO,
        "style":            style,
        "switch_command":   switch_command,
        "slam_n":           HEADSTAND_SLAM_N,
    }
    cfg.rewards["headstand_progress"] = RewardTermCfg(
        func=microduck_mdp.headstand_progress,
        weight=progress_w,
        params={"slam_n": HEADSTAND_SLAM_N},
    )
    cfg.rewards["headstand_composite"] = RewardTermCfg(
        func=microduck_mdp.headstand_composite,
        weight=4.0,
        params={
            "target_height":    HEADSTAND_Z,
            "height_std":       0.03,   # 3 cm: a flop at z≈0.06 scores ~0.03
            "inverted_std":     0.6,    # 1-cos(35°)=0.18 → e^-0.5; the balance basin scores visibly
            "pose_std":         0.45,   # joint-RMS, broad: partial pose still scores
            **hold_params,
        },
    )
    # Flat-topped inside the measured 20° rest tilt, falling off beyond it:
    # 1-cos(30°)-0.06 = 0.074 → 0.44 at 30°, 0.06 at 40°.
    cfg.rewards["headstand_inverted_sharp"] = RewardTermCfg(
        func=microduck_mdp.headstand_inverted_sharp,
        weight=1.5,
        params={"inverted_std": 0.3, "pose_std": 0.45, **hold_params},
    )
    # Always on: standing costs 1.0/step, the pike 0.5/step, the headstand 0.
    # Same role as standup's height_stand_l1. A version gated on the head-top
    # latch left a beak-down, feet-down rest free, and the policy parked there.
    cfg.rewards["headstand_not_inverted"] = RewardTermCfg(
        func=microduck_mdp.headstand_not_inverted_cost,
        weight=park_cost_w,
        params={"slam_n": HEADSTAND_SLAM_N},
    )
    # Gates: hard, from step 0, cheap to compute, impossible to farm.
    cfg.rewards["headstand_feet_down"] = RewardTermCfg(
        func=microduck_mdp.headstand_feet_down_cost,
        weight=-1.0,
        params={"inverted_lo": FEET_DOWN_INVERTED_LO, "inverted_hi": FEET_DOWN_INVERTED_HI},
    )
    # A flop is a TERMINATION (below), not a per-step cost: charged per step, a
    # failed attempt costs more than freezing in the pike, and the policy
    # freezes. A small one-off cost keeps a flop worse than a clean miss.
    cfg.rewards["headstand_other_contact"] = RewardTermCfg(
        func=microduck_mdp.headstand_other_contact_cost,
        weight=-0.2,
    )
    cfg.rewards["headstand_airborne"] = RewardTermCfg(
        func=microduck_mdp.headstand_airborne_cost,
        weight=-2.0,
    )
    # |a_z| impact shaping from step 0 (the roulade's lesson: style is the
    # scarce resource, discovery is helped by the spawn mix). SELF-NEGATING →
    # POSITIVE weight (returns -|a_z|; a negative weight would pay for shocks).
    cfg.rewards["gentle"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.002,   # ramped to 0.005 by the curriculum below, the roulade's schedule
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    # Head impact: N above threshold on jaw_soft from the floor (velstand's
    # body_impact_cost), something the trunk accelerometer alone cannot see.
    # The structural protection is the slam gate (a slam forfeits the hold
    # for the episode); this cost prices the hard settles below it.
    cfg.rewards["head_impact"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=-1.0,
        params={"sensor_name": head_ground_cfg.name, "threshold": HEAD_IMPACT_THRESH_N},
    )
    # Rotation cost above omega_max. The natural tumble band of this robot is
    # 3.5-5.5 rad/s (the roulade's measured transit); a 0.2 s snap kick-up runs
    # at ~7 rad/s, and the cap at 2 rad/s with this weight gives the 0.4 s
    # kick-up of the published policies.
    cfg.rewards["headstand_overspeed"] = RewardTermCfg(
        func=microduck_mdp.roulade_overspeed_penalty,
        weight=overspeed_w,
        params={"omega_max": omega_max},
    )
    # Arrival damper: wobble around the balance point only. Starts at 0,
    # curriculum below (standup: timing, not magnitude, protects discovery).
    cfg.rewards["headstand_arrival_damping"] = RewardTermCfg(
        func=microduck_mdp.headstand_arrival_damping_cost,
        weight=0.0,
        params={"inverted_full": DAMPING_INVERTED_FULL, "inverted_zero": DAMPING_INVERTED_ZERO},
    )

    if style == "fold":
        # The fold policy: standing → resting pike. Drop every headstand hold
        # term and pay the pike instead; keep the gates that price slams,
        # jumps and flops.
        pike_q = microduck_mdp._HEADSTAND_PIKE_QPOS
        pike_pitch = float(microduck_mdp._HEADSTAND_PIKE_PITCH)
        pike_overrides = {i: float(pike_q[7 + i]) for i in range(14)}
        for name in ("headstand_progress", "headstand_composite", "headstand_inverted_sharp",
                     "headstand_not_inverted", "headstand_feet_down", "headstand_arrival_damping"):
            cfg.rewards.pop(name, None)
        # With the always-on cost and a flop termination, an early backward
        # fall was the cheapest episode. So: standing costs nothing, a flop
        # does not end the episode (it costs lightly, below), and the way down
        # pays ~13 once (the frontier to 76°) then 4/step in the pike. Forward
        # is the only thing that pays.
        cfg.rewards["fold_progress"] = RewardTermCfg(
            func=microduck_mdp.fold_progress,
            weight=10.0, params={"target_pitch": pike_pitch, "slam_n": HEADSTAND_SLAM_N},
        )
        cfg.rewards["fold_composite"] = RewardTermCfg(
            func=microduck_mdp.fold_composite, weight=4.0,
            params={
                "target_pitch": pike_pitch, "pitch_std": 0.25,            # ≈ 14°
                "target_height": float(pike_q[2]), "height_std": 0.03,
                "pose_std": 0.45, "target_overrides": pike_overrides,
                "slam_n": HEADSTAND_SLAM_N,
            },
        )
        cfg.rewards["fold_overshoot"] = RewardTermCfg(
            func=microduck_mdp.fold_overshoot_cost, weight=-1.0, params={"target_pitch": pike_pitch},
        )
        cfg.rewards["headstand_other_contact"].weight = -0.3   # a flop is billed, not terminated (below)

    # ── Sim2real regularisers (velocity's set; motion-blockers kept ≈ 0) ─────
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=0.0
    )
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.002   # the swing IS ω
    cfg.rewards["angular_momentum"].weight = -0.001
    cfg.rewards.pop("soft_landing", None)
    # Light: the fold brings the head near the chest and the legs near the trunk.
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_name": self_collision_cfg.name},
    )

    # ── Observations (identical layout to walking / standup policies) ─────────
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

    # ── Commands: every slot alive at a tiny range, none tracked ─────────────
    # The headstand takes no command; the 13D command block stays in the obs
    # (61D contract) with small non-zero sampling so no input neuron is dead.
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015)),
    )
    cfg.commands["body_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=BODY_POSE_CMD_RESAMPLE_S,
        ranges=((-0.005, 0.005),) * 3 + ((-0.05, 0.05),) * 3,
    )
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "head_pose"},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "body_pose"},
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
    if switch:
        # The split-switch flag rides in the twist vx slot (the posture-flag
        # convention of the sit-stand env): 0 = left leg forward, 1 = mirrored.
        # Resampled every 2-4 s, each resample sets it to 1 with probability
        # 0.6; every episode starts at 0 to match the hold spawn.
        command.resampling_time_range = (2.0, 4.0)
        cfg.commands["twist"] = microduck_mdp.SplitSwitchCommandCfg(
            **{**vars(command), "flip_prob": 0.6, "ramp_s": switch_ramp_s}
        )
    else:
        cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # ── Terminations ──────────────────────────────────────────────────────────
    # Inverted is the goal, so the tilt-based fall termination does not apply.
    cfg.terminations.pop("fell_over", None)
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": (feet_ground_cfg.name,)},
    )
    if kickup:
        cfg.terminations["flopped"] = TerminationTermCfg(
            func=microduck_mdp.headstand_flopped,
            time_out=False,
            params={"grace_steps": FLOP_GRACE_STEPS},
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

    # Spawn mix, stage 0 of the reverse curriculum below. The five bucket
    # weights are normalised by their sum. Kick-up: partway 0.30, hold 0.30,
    # the measured pike 0.20, the handover set's weight (0.5 → 23% / 23% / 15% / 38%
    # of episodes). Fold: standing 0.7, the pike 0.3. Partway spawns start AT
    # the pike angle (100°): with none between 95° and 125° the first half of
    # the swing has no on-policy data (AGENTS.md's reverse-curriculum rule).
    cfg.events["set_headstand_spawn"] = EventTermCfg(
        func=microduck_mdp.reset_headstand_spawn,
        mode="reset",
        params={
            "standing_prob":       0.0 if kickup else 0.7,
            "partway_prob":        0.30 if kickup else 0.0,
            "hold_prob":           0.30 if kickup else 0.0,
            "pike_prob":           0.20 if kickup else 0.30,
            "handover_prob":           handover_prob if kickup else 0.0,
            "standing_z_min":      0.11,
            "standing_z_max":      0.12,
            "standing_tilt_max":   math.radians(3.0),
            "partway_pitch_min":   math.radians(100.0),
            "partway_pitch_max":   math.radians(165.0),
            "partway_lerp_range":  (0.0, 1.0),
            "hold_pitch_noise":    math.radians(HOLD_PITCH_NOISE_DEG),
            "hold_z":              HEADSTAND_SPAWN_Z,
            "hold_overrides":      overrides,
            "joint_noise_std":     SPAWN_JOINT_NOISE_STD,
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
    cfg.events.pop("push_robot", None)

    # ── Terrain: flat only ────────────────────────────────────────────────────
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Curricula ─────────────────────────────────────────────────────────────
    cfg.curriculum.pop("terrain_levels", None)
    cfg.curriculum.pop("command_vel", None)

    # Reverse curriculum on the spawn mix: the hold and the swing first, then
    # more of the whole kick-up from the pike (measured pike weight 0.20 →
    # 0.275 → 0.325; the handover weight stays). Pacing copied from the roulade's
    # stages (1500 / 3000, not 300 / 600): shifting away from the hold before
    # the swing is mastered loses the skill. If a metric steps DOWN at a
    # boundary, stretch, never move earlier.
    cfg.curriculum["headstand_spawn_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_headstand_spawn",
            "param_stages": [
                {"step": 0,         "params": {"standing_prob": 0.0, "partway_prob": 0.30, "hold_prob": 0.30, "pike_prob": 0.20, "handover_prob": handover_prob}},
                {"step": 1500 * 24, "params": {"standing_prob": 0.0, "partway_prob": 0.25, "hold_prob": 0.20, "pike_prob": 0.275, "handover_prob": handover_prob}},
                {"step": 3000 * 24, "params": {"standing_prob": 0.0, "partway_prob": 0.20, "hold_prob": 0.15, "pike_prob": 0.325, "handover_prob": handover_prob}},
            ] if kickup else [
                {"step": 0,         "params": {"standing_prob": 0.7, "partway_prob": 0.0, "hold_prob": 0.0, "pike_prob": 0.3}},
                {"step": 1000 * 24, "params": {"standing_prob": 0.8, "partway_prob": 0.0, "hold_prob": 0.0, "pike_prob": 0.2}},
            ],
        },
    )
    # The not-inverted cost is on from step 0 and tightens once hold spawns balance.
    if kickup:
        cfg.curriculum["not_inverted_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "headstand_not_inverted",
                "weight_stages": [
                    {"step": 0,          "weight": park_cost_w},
                    {"step": 1000 * 24,  "weight": park_cost_w * 1.5},
                    {"step": 2000 * 24,  "weight": park_cost_w * 2.0},
                ],
            },
        )
    cfg.curriculum["gentle_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "gentle",
            "weight_stages": [
                {"step": 0,          "weight": 0.002},
                {"step": 1500 * 24,  "weight": 0.0035},
                {"step": 3000 * 24,  "weight": 0.005},
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
    # Smoothness: velocity's action_rate ramp scaled by this task's reward
    # mass (5.5/12 ≈ 0.46), stretched to the spawn-mix pacing; the polish
    # terms come in at polish_at (2500 like the roulade), after the skill
    # exists. polish_at 0 applies the final weights from step 0.
    def polish_stages(ladder: list[tuple[int, float]]) -> list[dict]:
        if polish_at <= 0:
            return [{"step": 0, "weight": ladder[-1][1]}]
        return [{"step": max(polish_at + offset, 0) * 24, "weight": w} for offset, w in ladder]

    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": polish_stages([(-2500, -0.05), (-1500, -0.1), (-1000, -0.2), (-500, -0.3), (0, -0.4), (500, -0.46)]),
        },
    )
    if kickup:
        cfg.curriculum["arrival_damping_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "headstand_arrival_damping",
                "weight_stages": polish_stages([(-2500, 0.0), (0, -0.025), (1000, -0.05)]),
            },
        )
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "joint_torque_rate_l2",
            "weight_stages": polish_stages([(-2500, 0.0), (0, -1e-3)]),
        },
    )

    return cfg


# ── RL runner configs ─────────────────────────────────────────────────────────
# One network and PPO setting for the family; each task has its own experiment
# name so checkpoints and wandb runs never mix. Symmetric tricks (the fold,
# the legs-together kick-up) train with the mirror loss; the split ones are
# asymmetric and must not. The actor and critic cfgs are shared instances;
# the registry deep-copies a runner cfg when it loads one.

_ACTOR = RslRlModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,  # normalizer MUST be baked into ONNX by export.py
    distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
    },
)
_CRITIC = RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True)


def _runner(name: str, symmetric: bool) -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=_ACTOR,
        critic=_CRITIC,
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
            symmetry_cfg=SYMMETRY_CFG if symmetric else None,
        ),
        wandb_project="mjlab_microduck",
        experiment_name=name,
        run_name=name,
        save_interval=250,
        num_steps_per_env=24,
        max_iterations=6_000,
    )


MicroduckHeadstandFoldRlCfg = _runner("microduck_headstand_fold", symmetric=True)
MicroduckHeadstandKickupLegsTogetherRlCfg = _runner("microduck_headstand_kickup_legs_together", symmetric=True)
MicroduckHeadstandKickupRlCfg = _runner("microduck_headstand_kickup", symmetric=False)
MicroduckHeadstandSplitSwitchRlCfg = _runner("microduck_headstand_splitswitch", symmetric=False)
