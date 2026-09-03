"""Microduck backflip task — launched by the operator's hands, land on the feet.

Episodic policy. The robot starts standing on a prescribed "launcher plate"
(``launcher.xml``, an 18x18x2 cm 50 kg prop whose pose and velocity are
rewritten every control step — see the BACKFLIP section of ``mdp.py``). The
plate holds still for ``hold`` seconds, then accelerates upward to ``vz`` while
pitching backward at ``w0`` over ``launch`` seconds, then teleports away. The
robot must ride that toss through a full 360 deg BACKWARD rotation and land on
its feet.

HAND-OFF PREMISE. On the real robot there is no launcher and no launcher
sensing: a human picks the duck up, holds it on two flat palms, and flicks.
The policy therefore gets NO plate observation — it feels the toss through the
IMU and the joint encoders, exactly as it will on hardware. The plate state and
the launch phase go to the CRITIC only (asymmetric actor-critic; the ball-kick
env is the precedent, where the actor is ball-blind and the critic sees the
ball). This is also why the actor obs block is byte-identical in layout to the
rest of the family (61D: 48 proprioception + [twist(3), head_pose(4),
body_pose(6)], head/body slots zero-padded): the runtime hot-swaps ONNX files
walk/stand/trick through one buffer.

MEASURED LAUNCH ENVELOPE (docs/backflip_envelope_results.md, Task 3 CPU probe;
these numbers are measured, not guessed — do not "tidy" them):
  z0 in [0.10, 0.20] m, vz in [2.00, 2.25] m/s, w0 in [24, 30] rad/s,
  tuck in [0.5, 1.0] closes a full 360 deg backward flip at all three probed
  launch heights, landing at 1.46-2.59 m/s.
  * vz has a CLIFF at the bottom of the box: 1.80 does not close the flip
    (347 deg), 1.85 barely does (361 deg, 0.8 deg of margin), 2.00 closes with
    39.5 deg of margin. NEVER widen vz downward.
  * w0 above ~36-39 rad/s REVERSES the measured rotation direction (the flick
    overdrives the sole contact and the flip comes out forward, face-down).
    30 is the ceiling with a verified backward direction, so the box stops
    there.
HARDWARE CONSTRAINT (the user's): landings above roughly 2.6 m/s — about a
34 cm free fall — risk damaging the real duck. The box's LOW-vz corner lands
around 1.5 m/s, so the defaults sit at the low-vz end (2.00-2.25, not the
2.5-3.0 that the first sweep's best-rotation cells wanted). The same concern is
why the |a_z| impact penalty starts at 2x the roulade weight and ramps higher,
and why ``backflip_landing`` prices SETTLING rather than merely passing
through a good pose between bounces.

WHY THE ROTATION GATE IS AIRBORNE, NOT SUPPORTED. Roulade (a floor roll) only
counts rotation while the robot TOUCHES the ground, because a roulade that
leaves the floor is a ballistic whip, not a roll. The backflip is the exact
mirror: rotation only counts while NOTHING touches the terrain. Without that
gate the cheapest 2*pi is to flop onto the back and log-roll along the floor,
which is not a backflip. The gate reads the whole-body ``robot_ground_contact``
sensor by name (``mdp._BACKFLIP_GROUND_SENSOR``) — and ``_sensor_any_contact``
fails OPEN, so a renamed sensor silently disables the gate with no error. That
name is asserted in ``tests/test_backflip_cfg.py``; keep it that way.
  Known, bounded exposure: the plate is a separate entity, so standing on it is
  "airborne" by this sensor. A policy could in principle rock backward off the
  parked plate to bank rotation without flipping. It is bounded — the fall from
  z0 buys well under 180 deg before terrain contact freezes the frontier, the
  progress term is potential-based (the frontier only pays once), and the
  landing annuity needs 300 deg — so a real flip strictly dominates it. If a
  run shows pre-launch rocking, the fix is to add the plate body to the
  ``robot_ground_contact`` sensor's secondary match, not to tax rotation.

DESIGN CHOICES AND WHERE THEY CAME FROM
  * ONE dense signal: ``flip_progress`` pays increments of the max-so-far
    airborne backward rotation, potential-based and normalized to 1.0 per full
    turn. With ``scale_rewards_by_dt`` (mjlab's default) a complete flip
    therefore pays exactly its WEIGHT, 8.0, in episode-summed reward, however
    it is flown; camping anywhere pays 0/step. The pay-rate cap is 25 rad/s,
    from the measurement: the envelope closes 377-455 deg in a ~0.5-0.6 s
    airborne window (13-16 rad/s average, higher while tucked), so the plan's
    pre-measurement 14 rad/s placeholder would have forfeited rotation during
    a perfectly good flip and blunted the only dense signal in the task.
  * NO overspeed penalty (roulade has one). The spin here is imparted BY the
    plate: taxing |omega| would price the launcher's action, not the policy's,
    and it is a pure motion-blocker on the one thing the maneuver is made of.
    Anti-violence pressure lives on |a_z|, action_rate and the landing's
    settle factor instead.
  * The landing annuity (weight 4.0) is gated on a near-complete flip (300-345
    deg): "stand still and never flip" satisfies feet/upright/height/calm
    trivially, and without the gate it is the argmax. Reward MASS (episode
    sums, dt-scaled): flip 8.0, landing up to 4.0 x ~2 s of post-landing
    annuity = ~8.0, ready_stance ~0.4. So a flip-and-crash earns ~8 and a
    flip-and-land earns ~16 — the landing is worth a second flip, and the
    stance is worth 5% of one.
  * ``ready_stance`` (weight 1.0) pays only during HOLD and dies at launch, so
    it can never oppose the flip. It exists so the robot waits on the hands
    instead of squirming off before the flick. Its ``stand_z`` is
    STAND_Z + PLATE_HALF_THICKNESS because the robot stands on the plate's TOP
    surface while the term measures against the plate's centre height ``z0``.
  * Motion-blockers (body_ang_vel, angular_momentum) stay at roulade's
    near-zero weights. Arithmetic, since this is the term most likely to eat
    the task: at a typical 14 rad/s flip, body_ang_vel costs
    0.002 x 14^2 x 0.5 s ~ 0.2, i.e. ~2.5% of the flip's 8.0. Smoothness terms
    are introduced by curriculum only after the skill exists — an attempt-tax
    during discovery makes "do nothing" win (proven twice on standup).
  * Symmetry mirror-loss ON: a backflip is sagittal / left-right symmetric,
    and the mirror loss directly fights the sideways-collapse failure mode
    (the accumulator's flatness gate already refuses to count a side tumble).
  * NO mid-flight reverse-curriculum spawn bucket. Deliberately deferred by
    the plan's self-review: it is the fix for "learns the launch, never the
    landing", which is a training-time finding, and this branch stops before
    training.

SPAWN. ``reset_backflip_robot_on_plate`` derives the robot's spawn height from
the SAME ``z0`` the plate uses, and runs after ``backflip_launch_params``
(events fire in dict insertion order). The base template's +-0.5 m x/y scatter
and random yaw are narrowed to a small on-plate jitter with near-zero yaw: the
plate is only 18 cm across, and the flick axis is world +y, so a random heading
would turn the backflip into a side flip.

DR / obs / noise / NaN guard mirror the roulade env (which mirrors standup,
which mirrors velocity — the recipe with proven transfer). Velocity pushes are
OFF: a shove mid-flip is incoherent.
"""

import math
from copy import deepcopy

# Symmetry — the flip is sagittal / left-right symmetric.
ENABLE_SYMMETRY = True

# ── Domain randomisation (matched to roulade/standup/velocity) ────────────────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_KD_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = False  # a shove mid-flip is incoherent
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

# ── Ranges (matched to the roulade / standup envs) ────────────────────────────
COM_RANDOMIZATION_RANGE             = 0.003   # ramped to 0.015 via curriculum
HEAD_COM_RANDOMIZATION_RANGE        = 0.003   # ramped to 0.01 via curriculum
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)  # unused (kp DR off)
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)    # unused (kd DR off)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# Episode budget: hold (<= 1.0 s once the curriculum has widened it) + launch
# ramp (<= 0.15 s) + a ~0.6 s airborne window + ~2 s to settle on the feet.
EPISODE_LENGTH_S = 4.0

# Empirically-measured standing trunk height above the sole contact plane
# (standup lesson: measure it on the actual model, never carry it across
# revisions — a 5 mm error once made the goal unreachable for days).
STAND_Z = 0.115

# launcher.xml: box half-extents 0.09 / 0.09 / 0.01. The plate BODY sits at z0;
# its top surface — where the feet are — is one half-thickness above that.
PLATE_HALF_THICKNESS = 0.01

# ── Launch envelope (MEASURED — see the module docstring) ─────────────────────
HOLD_RANGE    = (0.1, 0.4)     # widened to (0.1, 1.0) by curriculum
LAUNCH_RANGE  = (0.08, 0.15)
Z0_RANGE      = (0.10, 0.20)   # DR tail extended to 0.30 by curriculum
VZ_RANGE      = (2.00, 2.25)   # measured box; the low corner is the gentle landing
W0_RANGE      = (24.0, 30.0)   # 30 is the direction-verified ceiling
MAX_PAID_RATE = 25.0           # rad/s; the envelope needs 13-16 average

# Spawn scatter on the plate. x/y: the plate is 18 cm across and the robot's
# feet span ~8 cm, so 1 cm of jitter is what fits. yaw: the flick axis is world
# +y — a random heading would make it a SIDE flip, so heading noise is small and
# is DR for the operator's aim, not a task variation.
SPAWN_XY_NOISE   = 0.01
SPAWN_YAW_NOISE  = 0.05
SPAWN_TILT_NOISE = 0.02   # rad of roll/pitch — the operator's hands are not level

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
    MICRODUCK_LAUNCHER_CFG,
    MICRODUCK_STANDUP_ROBOT_CFG,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_backflip_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the Microduck backflip environment configuration.

    ``play`` is accepted for registry parity with the other task factories;
    nothing in this env branches on it (roulade is the same).
    """
    del play

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

    # Whole-robot ground contact — the AIRBORNE GATE. The rotation accumulator
    # only integrates while NOTHING here touches the terrain, so a log-roll on
    # the floor earns no progress and never completes the flip.
    # NAME IS LOAD-BEARING: _update_backflip_accum reads it by this exact string
    # (mdp._BACKFLIP_GROUND_SENSOR) and _sensor_any_contact fails OPEN on a
    # miss — a typo here disables the gate silently. Pinned by a cfg test.
    robot_ground_cfg = ContactSensorCfg(
        name="robot_ground_contact",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    # Full-collision robot (same model as standup/roulade): during the flip
    # anything can hit the floor, so foot-pad-only collisions are not enough.
    # Robot MUST stay the FIRST entity — the base reset events and
    # reset_backflip_robot_on_plate write robot root state at qpos[:, 0:7].
    cfg.scene.entities = {
        "robot": MICRODUCK_STANDUP_ROBOT_CFG,
        "plate": MICRODUCK_LAUNCHER_CFG,
    }
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, robot_ground_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

    # Extra contact headroom for the plate (two feet plus whatever else the
    # robot puts on it, on top of the full-collision robot's budget) — the same
    # allowance the ball-kick env makes for its ball.
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
        "pose",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Rewards: backflip task set ────────────────────────────────────────────
    # THE dense signal. Potential-based, normalized to 1.0 per full flip, so
    # (with dt-scaled rewards) a complete flip pays exactly 8.0 in episode-sum
    # however it is flown, and camping anywhere pays 0/step. max_paid_rate is
    # MEASURED (see docstring):
    # 25 rad/s prices genuinely violent spin without forfeiting the 13-16 rad/s
    # the real envelope needs.
    cfg.rewards["flip_progress"] = RewardTermCfg(
        func=microduck_mdp.backflip_progress,
        weight=8.0,
        params={"target_angle": 2 * math.pi, "max_paid_rate": MAX_PAID_RATE},
    )

    # Completion-gated landing annuity — the dominant attractor and the whole
    # point of the task. Multiplicative (gate x feet x upright x height x calm
    # x settle): any single deficient factor collapses it, so there is no
    # compromise basin. Stds stay wide enough that a mediocre first landing
    # still scores visibly (standup composite lesson). Worth up to ~8.0 in
    # episode-sum over the ~2 s that remain after a flip — a second flip's
    # worth, which is what makes landing, not merely rotating, the argmax.
    cfg.rewards["landing"] = RewardTermCfg(
        func=microduck_mdp.backflip_landing,
        weight=4.0,
        params={
            "stand_z":     STAND_Z,       # measured against the GROUND
            "height_std":  0.04,
            "omega_std":   3.0,
            "lin_vel_std": 0.5,
            "sensor_name": feet_ground_cfg.name,
        },
    )

    # Stand still on the operator's hands. HOLD-phase only — it dies at launch,
    # so it can never oppose the flip, and it is not gated on a bad state
    # (standing on the plate IS the good state here). stand_z carries the plate
    # half-thickness: the term measures trunk height against the plate CENTRE
    # (z0) but the feet rest on its top surface.
    cfg.rewards["ready_stance"] = RewardTermCfg(
        func=microduck_mdp.backflip_ready_stance,
        weight=1.0,
        params={
            "stand_z":    STAND_Z + PLATE_HALF_THICKNESS,
            "height_std": 0.03,
        },
    )

    # ── Sim2real regularisers ─────────────────────────────────────────────────
    # Motion-blockers stay near zero: the flip IS a large angular-velocity
    # event, and (unlike roulade) most of that omega is imparted BY the plate,
    # so taxing it prices the launcher rather than the policy. No overspeed
    # penalty here for the same reason.
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=0.0
    )

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.002   # must stay ~0: the flip is omega
    cfg.rewards["angular_momentum"].weight = -0.001
    cfg.rewards.pop("soft_landing", None)

    # Arrival damper — trunk omega_xy^2 gated on standing height AND low tilt,
    # so the flip itself is never taxed. Introduced at 0, ramped by curriculum.
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

    # |a_z| impact shaping — 2x roulade's starting weight, and ramped higher:
    # the landing is THE hardware risk in this task (>2.6 m/s touchdown damages
    # the real duck), so impact is priced from step 0 rather than introduced
    # late. NOTE: trunk_vertical_accel_penalty is SELF-NEGATING (returns
    # -|a_z|) → POSITIVE weight. A negative weight here would pay for violence.
    cfg.rewards["gentle_landing"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.004,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # Self-collision — LIGHT: a tucked flip needs body-on-body contact
    # (knees against the trunk); standup's -1.0 would fight the tuck.
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_name": self_collision_cfg.name},
    )

    # An always-on upright term would oppose the flip (the roulade lesson);
    # landing uprightness is handled by the completion-gated annuity.
    if "upright" in cfg.rewards:
        del cfg.rewards["upright"]

    # ── Observations (identical layout to walking / standup / roulade) ────────
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
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

    # Command obs slots: zero padding for BOTH head (4) and body (6). Neither is
    # commanded here, but the 61D layout parity with velocity/standup is what
    # lets the runtime hot-swap this ONNX into the same buffer (send zeros).
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # CRITIC-ONLY launcher state (asymmetric actor-critic). The actor stays
    # plate-blind — there is no launcher sensor on the real robot — while the
    # critic gets the plate's relative pose/velocity and the phase, which is
    # what makes the value function's job tractable across a randomized toss.
    # (ball_pos_in_base / ball_vel_in_base are generic in the asset name; they
    # are the ball-kick env's critic-only ball terms, reused verbatim.)
    cfg.observations["critic"].terms["plate_position"] = ObservationTermCfg(
        func=microduck_mdp.ball_pos_in_base, params={"asset_name": "plate"},
    )
    cfg.observations["critic"].terms["plate_velocity"] = ObservationTermCfg(
        func=microduck_mdp.ball_vel_in_base, params={"asset_name": "plate"},
    )
    cfg.observations["critic"].terms["plate_phase"] = ObservationTermCfg(
        func=microduck_mdp.backflip_phase_obs,
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
    # Being upside down is the task — keep only the NaN guard + timeout.
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # ── Events ────────────────────────────────────────────────────────────────
    # Base template's reset_base scatters the robot by +-0.5 m and gives it a
    # random heading. The plate is 18 cm across and its flick axis is world +y,
    # so both must be narrowed: off-plate spawns get no launch at all, and a
    # random yaw turns the backflip into a side flip. Height is left at the
    # default here and overwritten below from the sampled z0.
    cfg.events["reset_base"].params["pose_range"] = {
        "x":     (-SPAWN_XY_NOISE, SPAWN_XY_NOISE),
        "y":     (-SPAWN_XY_NOISE, SPAWN_XY_NOISE),
        "z":     (0.0, 0.0),
        "roll":  (-SPAWN_TILT_NOISE, SPAWN_TILT_NOISE),
        "pitch": (-SPAWN_TILT_NOISE, SPAWN_TILT_NOISE),
        "yaw":   (-SPAWN_YAW_NOISE, SPAWN_YAW_NOISE),
    }
    # Deployment hands off from the walk/stand policy, whose settled pose won't
    # match HOME exactly.
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)

    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    # This episode's toss. Registered AFTER the base reset events (dict
    # insertion order) and BEFORE the spawn, which reads the sampled z0.
    cfg.events["backflip_launch_params"] = EventTermCfg(
        func=microduck_mdp.reset_backflip_launch_params,
        mode="reset",
        params={
            "hold_range":   HOLD_RANGE,
            "launch_range": LAUNCH_RANGE,
            "z0_range":     Z0_RANGE,
            "vz_range":     VZ_RANGE,
            "w0_range":     W0_RANGE,
        },
    )

    # Stand the robot on the plate top at THIS episode's z0. Must follow
    # backflip_launch_params; see the function's docstring for why an
    # independent spawn height is not an option.
    cfg.events["backflip_spawn"] = EventTermCfg(
        func=microduck_mdp.reset_backflip_robot_on_plate,
        mode="reset",
        params={
            "stand_z":              STAND_Z,
            "plate_half_thickness": PLATE_HALF_THICKNESS,
        },
    )

    # Park the plate at THIS episode's z0 during the reset itself. The step
    # event below already re-places it before every physics step of an
    # auto-reset episode, but ManagerBasedRlEnv.reset() does NOT run step-mode
    # events: without this, the first physics step after construction runs with
    # the plate at its XML default (0.15) while the robot stands at the sampled
    # z0 — up to 5 cm of interpenetration with a 50 kg prop on step one. Same
    # function, reset mode; it ignores env_ids and rewrites every env's plate
    # from the prescription, which is exactly what the step event does anyway.
    cfg.events["backflip_plate_reset"] = EventTermCfg(
        func=microduck_mdp.backflip_plate_step,
        mode="reset",
        params={"asset_name": "plate"},
    )

    # The plate is a PROP: its pose and velocity are rewritten every control
    # step, so its own dynamics never matter (a stiff hand that neither sags
    # under the duck nor recoils when it pushes off).
    cfg.events["backflip_plate"] = EventTermCfg(
        func=microduck_mdp.backflip_plate_step,
        mode="step",
        params={"asset_name": "plate"},
    )

    if "push_robot" in cfg.events:
        del cfg.events["push_robot"]

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

    # ── Terrain ───────────────────────────────────────────────────────────────
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Curriculum ────────────────────────────────────────────────────────────
    if "terrain_levels" in cfg.curriculum:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # Hold-window widening. A narrow hold early keeps every episode's launch at
    # a similar time so the policy can learn what the flick FEELS like; widening
    # it later (to a full second) stops it from learning the clock instead of
    # the cue. Phase-aligned with the flip existing first — hardening the
    # timing DR before the skill consolidates is the pacing failure that has
    # bitten this repo before.
    cfg.curriculum["backflip_hold_range"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "backflip_launch_params",
            "param_stages": [
                {"step": 0,         "params": {"hold_range": (0.1, 0.4)}},
                {"step": 1500 * 24, "params": {"hold_range": (0.1, 0.7)}},
                {"step": 3000 * 24, "params": {"hold_range": (0.1, 1.0)}},
            ],
        },
    )

    # Launch-height DR tail. [0.10, 0.20] is the MEASURED box; 0.20-0.30 is the
    # spec's operator tail and is EXTRAPOLATION beyond the probe — a higher
    # launch means a higher apex and a harder landing, so it is introduced late
    # and only as a tail. If landings get violent, cut this back to 0.25 (or to
    # the measured box) before touching the reward weights.
    cfg.curriculum["backflip_z0_range"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "backflip_launch_params",
            "param_stages": [
                {"step": 0,         "params": {"z0_range": (0.10, 0.20)}},
                {"step": 2000 * 24, "params": {"z0_range": (0.10, 0.25)}},
                {"step": 4000 * 24, "params": {"z0_range": (0.10, 0.30)}},
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

    # action_rate: softer than roulade's ramp (-0.05 rather than -0.1 at stage
    # 0, ceiling -0.2 rather than -0.4). The tuck-and-open is a FAST, one-shot
    # motion inside a ~0.6 s window; roulade's ceiling was already measured to
    # be squeezing its (slower) rise. Kept non-zero from step 0 so the policy
    # never breeds jitter it later has to unlearn.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "action_rate_l2",
            "weight_stages": [
                {"step": 0,         "weight": -0.05},
                {"step": 1500 * 24, "weight": -0.1},
                {"step": 3000 * 24, "weight": -0.2},
            ],
        },
    )

    # Smoothness polish and settle damping — introduced only AFTER the flip
    # exists. Any attempt-tax active during discovery makes "do nothing" win
    # (proven twice on standup); the fix is timing, not magnitude.
    cfg.curriculum["arrival_damping_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "arrival_damping",
            "weight_stages": [
                {"step": 0,         "weight": 0.0},
                {"step": 2000 * 24, "weight": -0.025},
                {"step": 3000 * 24, "weight": -0.05},
            ],
        },
    )
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0,         "weight": 0.0},
                {"step": 2000 * 24, "weight": -5e-4},
                {"step": 3000 * 24, "weight": -1e-3},
            ],
        },
    )
    # POSITIVE weights: trunk_vertical_accel_penalty is self-negating. This is
    # the hardware-safety knob — it ramps to 2.5x roulade's ceiling because a
    # >2.6 m/s touchdown damages the real robot.
    cfg.curriculum["gentle_landing_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "gentle_landing",
            "weight_stages": [
                {"step": 0,         "weight": 0.004},
                {"step": 1500 * 24, "weight": 0.008},
                {"step": 3000 * 24, "weight": 0.0125},
            ],
        },
    )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckBackflipRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="backflip",
    run_name="backflip",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=10_000,
)
