"""Microduck backflip task — launched by the operator's hands, land on the feet.

STATUS. The first training run (4096 envs, 279 iterations) worked as a proof
that the env produces real backflips: flip_progress +1.97, landing +1.72, every
penalty negative, and the 300 deg landing gate opening. Three things it also
showed, and what answers each:
  * It COLLAPSED into the plate before the impulse — reported three times, and
    two rounds of re-weighting and re-shaping ``ready_stance`` never touched
    it, because it was never a tuning problem. The plate fires on its own
    schedule whatever the robot is doing, so a collapsed robot still collected
    flip_progress and the landing annuity (13.6 of 14.7) and forfeited only the
    stance term. And a lower, more compact body rotates MORE at the same flick
    (measured), so collapsing was slightly PROFITABLE. Answered structurally by
    the LAUNCH-ATTITUDE GATE (``mdp._backflip_launch_gate``): a latched,
    smooth, floored factor on BOTH task terms, so a flip that launches from a
    collapse does not count. See the mass table below.
  * The hold was invisible in play, and priced only at the last curriculum
    stage. The hold is now plain DR, 1-5 s uniform from step 0 (no curriculum
    at all), the episode is 7.6 s to fit it, and the landing annuity pays over
    a FIXED window latched at touchdown so its payout does not depend on the
    hold draw.
  * The ejection was too strong, and then the user described what it actually
    looked like: the robot "takes something like a force that makes it
    rotate". It did. THE PLATE WAS SWEEPING 72-110 DEG UNDER THE FEET
    (0.5 * w0 * t_launch — see ``mdp.backflip_plate_sweep_deg``), which is a
    catapult paddle levering the robot over, not a hand imparting an impulse.
    No probe had ever reported that quantity. It is now a first-class measured
    column and a HARD acceptance gate at 45 deg, next to direction, and the
    box was re-measured under it: w0 18-24 / t_launch 0.14-0.16 became
    w0 5-6 / t_launch 0.22-0.26 (sweep 32-45 deg), a LONGER and much gentler
    flick. The honest cost is stated at the ranges: a hand-like sweep cannot
    close 360 deg open-loop at any landing speed the user would accept, so
    open-loop closure is now 0% and the last 50-160 deg has to come from the
    policy's tuck.
  * And then the user restated the actual ask: "je veux juste qu'il reste
    immobile droit sur une plateforme immobile" — stand still and straight on a
    static platform. Three things were wrong with how that was paid for.
    (a) ``ready_stance`` was capped at 1.10 by an INVALID argument (that a
    stance worth more than the landing annuity would make "stand still and
    never flip" the argmax — impossible, since the plate fires on a prescribed
    schedule and the robot cannot decline to flip). Retracted; the weight is
    now 3.0, i.e. 3-15 points across the hold draw, a co-equal major term.
    (b) The term said nothing about the JOINTS, so a robot sagging into a
    different configuration at the same trunk height scored the same. It now
    carries a pose factor against the spawn pose — as a FACTOR, never an
    additive term, because measured open-loop the joint error against HOME is
    LOWER once the robot has toppled. (c) The platform was not static: at any
    ``z0`` above PLATE_HALF_THICKNESS the plate free-falls 2.5 mm per control
    step between the step event's rewrites and hammers the soles at 50 Hz
    (measured, ``--plate-jitter``), so ``Z0_RANGE`` is now the single value
    that rests the slab on the floor.

Everything above is measured on CPU MuJoCo with BAM actuators and recorded in
docs/backflip_envelope_results.md, which is append-only: it contains the
superseded boxes too, and its header names the one that is current.

ACTUATORS, AND A PROBE DEFAULT THAT MISLED THIS BRANCH TWICE. Training uses
BAM (voltage-controlled XL330). The scene XML's own position servos are
kp 0.386-0.55 N.m/rad, which at a realistic 0.05-0.20 rad of tracking error
deliver 9-30x LESS torque than BAM does, and a robot with legs that soft
absorbs the flick in its knees instead of transmitting it. Measured open-loop
drift from the standing spawn, same command: BAM tilts 4.8 / 8.8 / 13.6 /
26.0 deg at 0.3 / 0.5 / 0.7 / 1.0 s; the XML PD tilts 6.5 / 15.6 / 32.5 and has
toppled by 1.0 s. ``scripts/backflip_envelope.py`` therefore defaults to BAM
now (``--xml-pd`` opts out, ``--bam`` is accepted and ignored), and every table
in the results doc above its "Standing-spawn re-measurement" section — which is
all of them measured on the XML PD — is marked there as not representative of
the trained dynamics.

WHY THE PLATE SITS ON THE GROUND NOW. An earlier revision floored Z0_RANGE at
0.07 m because the then-current KNEELING tuck rested on its shins with its feet
hanging ~8 cm below the slab, so a lower plate put them through the floor. That
posture is gone. Standing's LOWEST GEOMS ARE THE FEET, so the plate can rest on
the ground (z0 = PLATE_HALF_THICKNESS puts the slab exactly on the floor) and
the spawn still puts the soles on its top surface by construction. The honest
cost is altitude: a floor-level launch has less airtime, so a given ``vz``
closes less rotation than it did from a 0.10 m plate. That is a real trade the
user accepted, and it is why the open-loop closure fraction is now REPORTED
rather than required (see below).

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

MEASURED LAUNCH ENVELOPE — the ranges below are MEASURED, from the env's ACTUAL
standing spawn with the plate on the ground, with BAM actuators
(docs/backflip_envelope_results.md, "Gentler ejection"; probe mode
``scripts/backflip_envelope.py --box-check --bam``). Do not "tidy" them:
  z0 in [0.01, 0.03] m, vz in [2.20, 2.60] m/s, w0 in [5, 6] rad/s,
  t_launch in [0.22, 0.26] s, hold 1-5 s. The probe imports these from THIS
  module, so there is one source of truth.

  TWO HARD ACCEPTANCE GATES, both per-cell:
    1. DIRECTION — every cell must rotate BACKWARD. A cell that comes out
       forward is not a gentler backflip, it is a different maneuver, and the
       accumulator's sign convention would refuse to pay it anyway. It binds
       from BOTH ends: w0 above ~24-27 overdrives the sole contact and comes
       out forward, and so does a SHORT flick (t_launch <= 0.08 s, i.e. a
       linear acceleration of 28-56 m/s^2 under a head-heavy body whose CoM is
       ahead of the sole contact: EVERY such cell measured forward, -88 to
       -269 deg).
    2. PLATE SWEEP <= 45 deg — ``sweep = 0.5 * w0 * t_launch``
       (``mdp.backflip_plate_sweep_deg``; the launch ramps the pitch rate 0 ->
       w0 over t_launch, so the swept angle is the area under that ramp). A
       hand sweeps 30-40 deg and lets go; a surface that sweeps 90 deg is a
       paddle levering the robot over, and the feet cannot stay on it. This
       box sweeps 32-45 deg. The withdrawn box swept 72-110 deg in all 243 of
       its cells, and no probe mode reported the number until it was made a
       column.

  OPEN-LOOP CLOSURE IS INFORMATION, NOT A BAR — and it is now 0%. The box
  rotates 198-305 deg with a full tuck held from the flick, landing at
  2.37-3.50 m/s (81 cells, z0 being a single value now). That is arithmetic,
  not tuning: the sweep cap forces
  w0 <= 1.571 / t_launch, the forward-tip limit forces t_launch >= ~0.16 s, so
  w0 <= ~9.8 rad/s; the robot leaves at roughly the plate's final rate, so
  closing 2*pi needs >= 0.64 s of airtime, i.e. vz >= ~3.1 m/s, i.e. a
  3.7-4.9 m/s landing. A HAND-LIKE SWEEP AND OPEN-LOOP CLOSURE ARE
  INCOMPATIBLE at an acceptable landing speed. The last 50-160 deg is the
  policy's tuck — which is the skill the task exists to train, but it is a
  bet, and the measured alternative (vz 3.0-3.6: 29% closure at 3.73-4.88 m/s)
  is one row down the ladder at the ranges if the user prefers to pay for it.

HARDWARE CONSTRAINT (the user's): landings above roughly 2.6 m/s — about a
34 cm free fall — risk damaging the real duck. The current box lands at
2.37-3.50 m/s, with 23 of its 81 sampled cells under the threshold. The apex
is 0.45-0.65 m. Both are set by ``vz``, which is also the only remaining lever
on rotation now that the sweep is capped — the trade is written out at
VZ_RANGE. The same
concern is why the |a_z| impact penalty starts at 2x the roulade weight and
ramps higher,
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
  parked plate to bank rotation without flipping. It is bounded because the
  accumulator measures ONE CONTINUOUS AIRBORNE ARC: terrain contact resets it
  to zero (``mdp._update_backflip_accum``, clamp 2), so the fall from z0 buys
  well under 180 deg and the next rock starts from zero rather than adding to
  it. The frontier is the best single arc, the progress term is
  potential-based (the frontier only pays once), and the landing annuity needs
  300 deg — so a real flip strictly dominates. That reset is load-bearing, not
  tidiness: the earlier version only zeroed the per-step delta on contact,
  which left the frontier RATCHETABLE (bank 20-30 deg airborne, land, unwind
  on the ground for free, repeat ~11 times, collect the annuity with no flip).
  The accumulator is also floored at zero so a launch that comes out FORWARD
  cannot dig a hole the policy must climb out of before the frontier moves.
  Both clamps have regression tests. If a run shows pre-launch rocking anyway,
  the next fix is to add the plate body to the ``robot_ground_contact``
  sensor's secondary match, not to tax rotation.

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
  * REWARD MASS (episode sums, dt-scaled — AGENTS.md: compare mass, not
    weight). The hold H is drawn uniformly 1-5 s, the landing annuity pays over
    a fixed 1.4 s window, and BOTH flip_progress and landing are multiplied by
    the launch-attitude gate latched at the flick
    (``mdp._backflip_launch_gate``), 1.0 for an upright launch and
    ``LAUNCH_GATE_FLOOR`` for a collapsed one:

      launch posture      flip   landing   ready_stance (H=1 / H=5)   TOTAL
      upright              8.0     5.6      3.0 / 15.0             16.6 / 28.6
      collapsed (0.3)      2.4     1.68     ~0                       4.1
      collapsed (0.05)     0.4     0.28     ~0                       0.7

    READY_STANCE IS 3.0, NOT 1.10, AND THE OLD CEILING IS RETRACTED. For three
    revisions this weight was capped by "a stance worth more than the landing
    annuity makes 'stand still and never flip' the argmax". That argument is
    INVALID: ``backflip_plate_step`` is a ``mode="step"`` event, so the plate
    fires on its prescribed schedule whatever the policy does — the robot
    cannot decline to flip, and there is no such strategy to farm. The only way
    to dodge the flick is to walk off the plate, which pays nothing (the
    stance's height factor collapses on the floor beside it, the attitude gate
    never latches a good posture, and the annuity is gated on a flip that never
    happens). The invalid ceiling is what kept the hold at 7-28% of the stack
    for three waves while the user reported the same collapse three times.

    So the hold is now a co-equal major term: 3.0-15.0 against the flip's 8.0
    and the annuity's 5.6, median 9.0. Standing upright through the hold is
    worth 12.5 (H=1) to 24.5 (H=5) more than collapsing into the plate — the
    stance mass plus the (1 - floor) of both gated terms.

    THE ACCEPTANCE STATEMENT, at both ends of the hold draw:
      1. Collapsing is heavily unprofitable: it forfeits the stance mass AND
         9.5 points of gated flip+landing at floor 0.3 (12.9 at 0.05). The
         extra rotation a compact body buys cannot pay for that, and
         flip_progress caps at one turn anyway.
      2. Flipping still beats not flipping at every draw — an upright launch
         adds 13.6 on top of the stance, and a collapsed one still adds 4.1
         (the floor is what keeps flip discovery alive).
      3. Nothing rewards being in a bad state. `pose` is a FACTOR of
         ready_stance, never an additive term: measured open loop, the joint
         error against HOME is LOWER after the robot has toppled than while it
         is still upright and loaded (rms 0.056-0.076 vs 0.086 rad), and every
         flop basin scores pose 0.90-0.99 — inside the product, height and
         upright zero all of them (measured: held pose 0.988, best flop 0.000,
         `--flop-audit`).
  * The landing annuity (weight 4.0) is gated on a near-complete flip (300-345
    deg): "stand still and never flip" satisfies feet/upright/height/calm
    trivially, and without the gate it is the argmax. It pays over a FIXED
    LANDING_WINDOW_S window latched at touchdown, so the same landing is worth
    the same 5.6 whenever it happens — before that it paid "all the time
    remaining", and an identical backflip earned 21.8 at a 1 s hold against
    5.84 at a 5 s hold, a 4x swing in the main attractor decided by a draw the
    policy neither controls nor observes.
  * ``ready_stance`` (weight 1.10) pays only during HOLD and dies at launch, so
    it can never oppose the flip. It pays ``height x upright`` for STANDING ON
    THE PLATE: the height Gaussian (std 0.03) says "be at full standing height
    on the plate" — the floor beside the plate is ~3 sigma out and a 5 cm
    crouch scores under a fifth of upright — and a tilt smoothstep (full below
    10 deg, zero above 45) says "and be upright". Its ``stand_z`` carries
    PLATE_HALF_THICKNESS because the robot rests on the plate's TOP surface
    while the term measures against the plate's centre height ``z0``. The
    upright factor is load-bearing: height alone cannot tell a standing robot
    from a stable side-lying one, and the measured flop basins sit at 80-126
    deg of tilt, so without it a flop could out-earn the stance (it did, on the
    tucked hold: 0.991 against upright's 0.950). It costs nothing at the pose's
    own measured drift (7.3 deg by 0.5 s).
    It is SHAPING for the launch-attitude gate, not the thing pricing the hold:
    the gate is what makes a collapse worthless, and this term is what gives
    the policy a dense gradient toward the posture the gate scores.
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

SPAWN. ``reset_backflip_robot_on_plate`` stands the robot on the plate's TOP
surface: it derives its spawn height from the SAME ``z0`` the plate uses, plus
PLATE_HALF_THICKNESS, plus the MEASURED standing trunk height STAND_Z. It sets
height only and leaves the joints to ``reset_robot_joints`` (whose +-0.05 rad of
scatter is a standing pose plus noise), and it runs after
``backflip_launch_params``, which samples ``z0`` — events fire in dict insertion
order. The TUCKED spawn is withdrawn (AMENDMENT 3): the tucked rest put the feet
2.2 cm INSIDE the plate slab, and the whole envelope measured from it was
measured from a state the physics would never produce.
The base template's +-0.5 m x/y scatter
and random yaw are narrowed to a small on-plate jitter with near-zero yaw: the
plate is only 18 cm across, and the flick axis is world +y, so a random heading
would turn the backflip into a side flip.

FIRST EPISODE UNDER ``play``. mjlab never calls ``env.reset()`` before the
viewer's first episode (``ManagerBasedRlEnv.__init__`` does not reset, and
``mjlab/viewer/base.py`` calls it only on the RESET action), so
``uv run play`` runs a whole 4 s episode on the COMPILED default state plus
``_backflip_state``'s lazy defaults. That state used to be nonsense — robot
standing at its compiled 0.12 m trunk height, plate hovering motionless at
0.15 m, i.e. through its body, then teleporting away without launching — and
it is what a user reported as "the plate is stuck in the middle of the robot,
not under its feet". The entity init states and the lazy defaults are now
coherent (tucked on the plate at the middle of Z0_RANGE, with a mid-box toss),
so the first episode looks like an episode. Press the viewer's reset key for a
properly sampled one.

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

# Episode budget, MEASURED rather than guessed: hold (up to 5.0 s, sampled
# uniformly from step 0) + launch ramp (<= 0.26 s, the flick is LONG now — see
# LAUNCH_RANGE and the plate-sweep gate) + the airborne window (0.88 s is kept
# as a CONSERVATIVE bound; it was measured when vz reached 4.0, and the current
# apex of 0.45-0.65 m flies for ~0.55-0.75 s) + settle. 7.5 -> 7.6 s when the
# flick lengthened to 0.26 s: the worst case must leave at least the full
# LANDING_WINDOW_S to settle, or a long hold would TRUNCATE the annuity and
# make the landing hold-dependent again — the exact defect the fixed window
# exists to remove. At 7.6 s it leaves 7.6 - 5.0 - 0.26 - 0.88 = 1.46 s
# against a 1.4 s window, and a short hold leaves over 6 s. A cfg test derives
# this from LAUNCH_RANGE so lengthening the flick again cannot slip through.
EPISODE_LENGTH_S = 7.6

# How long after the first terrain contact `backflip_landing` keeps paying.
# 1.4 s is what the WORST case can always afford (7.5 - 5.0 - 0.16 - 0.88 =
# 1.46 s), so an identical backflip earns the same annuity whatever the hold
# draw was. Before this the annuity paid for "all the time remaining", i.e.
# 21.8 in episode-sum at a 1 s hold against 5.84 at a 5 s hold — a ~4x swing
# in the MAIN attractor decided by a draw the policy cannot observe, and it
# made a long-hold episode worth less than a short one for the same skill.
LANDING_WINDOW_S = 1.4

# Floor of the LAUNCH-ATTITUDE GATE, which multiplies BOTH flip_progress and
# landing (see mdp._backflip_launch_gate). A collapsed launch earns
# floor x the task reward instead of all of it.
#
# 0.3 at step 0, tightened toward 0 by curriculum once the stance consolidates.
# It is floored rather than binary on purpose: a hard zero starves flip
# discovery, because the robot cannot balance yet and a stance it cannot yet
# hold would pay nothing for the flip either. AGENTS.md's "introduce the tax
# after the skill exists" applies to the FLOOR, not to the gate.
LAUNCH_GATE_FLOOR = 0.3
LAUNCH_GATE_FLOOR_FINAL = 0.05

# Empirically-measured standing trunk height above the sole contact plane
# (standup lesson: measure it on the actual model, never carry it across
# revisions — a 5 mm error once made the goal unreachable for days). Used by
# the LANDING target only: the duck lands on its feet, standing.
STAND_Z = 0.115

# ── The HOLD posture: TUCKED ─────────────────────────────────────────────────
# The robot waits on the operator's hands FOLDED, not standing. That is not a
# style choice, it is the measurement: from a standing spawn no launch setting
# in vz [2, 4] x w0 [3, 36] closes 360 deg below the ~2.6 m/s hardware landing
# limit, and the configured box rotates a standing robot FORWARD. From the
# tucked spawn the same class of box closes 393-484 deg at 1.68-2.51 m/s. See
# docs/backflip_envelope_results.md, "Tucked hold".
#
# Servo index -> full-tuck angle. Same values as the roulade env's
# TUCK_OVERRIDES (legs folded, chin tucked); duplicated rather than imported so
# this env can retune its own tuck without moving roulade's, and pinned equal
# to roulade's by a cfg test so the duplication cannot drift silently.
TUCK_OVERRIDES = {
    2:  -1.15,  # left  hip_pitch
    3:   1.25,  # left  knee
    4:   1.05,  # left  ankle
    5:  -1.0,   # neck_pitch  (chin tuck)
    6:   1.0,   # head_pitch  (chin tuck)
    11:  1.15,  # right hip_pitch
    12: -1.25,  # right knee
    13: -1.05,  # right ankle
}

# Same tuck, keyed by joint NAME instead of servo index — for the entity's
# init_state (mjlab matches joint_pos by regex, not by index). Kept adjacent to
# TUCK_OVERRIDES and pinned equal to it by a cfg test.
_TUCK_BY_NAME = {
    r"^left_hip_pitch$":  -1.15,
    r"^left_knee$":        1.25,
    r"^left_ankle$":       1.05,
    r"^neck_pitch$":      -1.0,
    r"^head_pitch$":       1.0,
    r"^right_hip_pitch$":  1.15,
    r"^right_knee$":      -1.25,
    r"^right_ankle$":     -1.05,
}

# Tuck depth, as a fraction of TUCK_OVERRIDES. MEASURED CHOICE (2026-09-07, CPU
# MuJoCo + BAM): 0.5 / 0.75 / 1.0 all settle without falling on the plate, at
# equilibrium tilts of 12.3 / 14.0 / 15.3 deg and 3 s x/y drifts of 7 / 11 /
# 15 mm. 0.75 sits in the middle of the whole-box closure region on both the
# stability and the rotation axes.
TUCK_FACTOR = 0.75

# ── The GROUND-LEVEL hold: a feet-flat SQUAT ─────────────────────────────────
# HOLD_LERP is HOME lerped this far toward TUCK_OVERRIDES, per joint (the same
# parametrisation roulade's mid-roll spawn uses) — NOT a scaling of the tuck
# targets. The distinction is the whole reason this pose exists: the lerp keeps
# the SOLES DOWN, and a feet-flat squat is the only pose measured to rest on a
# plate lying on the ground. The deep kneeling tuck hangs its feet ~8 cm BELOW
# whatever it rests on, which is fine on a plate held 10 cm up and impossible
# on one lying on the floor (measured: the feet reach the floor and the robot
# settles at 41 deg, sliding 5.5 cm).
#
# 0.65 is MEASURED (2026-09-07, CPU MuJoCo + BAM, 16 noisy trials): it settles
# UPRIGHT and stays there — tilt 3.5 deg at 0.1 s and 0.5 s, 2.1 at 1.0 s, 0.8
# at 3.0 s, x/y drift 9-13 mm, 0/16 fallen, and ZERO robot-floor contacts
# through the whole hold. 0.60 and 0.70 also hold (10-11 deg); 0.75 and deeper
# topple (49 deg by 1.0 s, 15/16 fallen) as the soles start to leave the plate.
HOLD_LERP = 0.65

# MEASURED resting trunk height of that squat above the plate TOP (same run:
# 0.0657-0.0661 m at HOLD_LERP 0.65, reached within 0.3 s from every spawn
# height tried). Do NOT derive it from STAND_Z or TUCK_Z — three different
# poses, three different heights.
HOLD_Z = 0.066

# MEASURED tucked trunk height above the plate TOP (2026-09-07, CPU MuJoCo with
# BAM actuators; `uv run python scripts/backflip_envelope.py --measure-tuck-z
# --bam --tuck 0.5,0.75,1.0`). Drop the tucked robot from 0.020 / 0.026 / 0.029
# / 0.032 m above the plate top holding the tuck ctrl: it comes to rest at
# 0.0286 m for tuck factors 0.5-0.75 and 0.0257 m at full tuck, from every one
# of those spawn offsets, within 0.1-0.3 s. Spawning much BELOW this jams the
# folded legs into the plate and the contact solver ejects the robot off it, so
# the number is a floor as well as a target.
#
# DO NOT derive this from STAND_Z. It is a different pose, and AGENTS.md
# records a 5 mm height carried across model revisions costing days.
TUCK_Z = 0.029

# launcher.xml: box half-extents 0.09 / 0.09 / 0.01. The plate BODY sits at z0;
# its top surface — where the feet are — is one half-thickness above that.
PLATE_HALF_THICKNESS = 0.01

# ── Launch envelope — a plausible human throw ────────────────────────────────
# These are v0 ranges: the spread a person's hands plausibly deliver, centred
# on what the CPU probe measured. They are NOT tuned so that an unskilled robot
# completes every throw.
#
# WHY NOT. An earlier revision required WHOLE-BOX OPEN-LOOP CLOSURE — every
# sampled combination had to complete 360 deg with the robot holding a fixed
# pose. That squeezed w0 to the single value 18.5 rad/s, which no human hand
# reproduces, and it is the wrong bar besides: compensating for an imperfect
# throw is exactly the policy's job. Open-loop closure is now reported as
# INFORMATION about starting difficulty, not a gate.
#
# TWO HARD GATES. Both are checked by `--box-check`, and a box that fails
# either is not a box.
#
# 1. DIRECTION: every cell must rotate BACKWARD. It binds from BOTH ends.
#    Above roughly w0 = 24-27 rad/s the flick overdrives the sole contact and
#    the robot comes out FORWARD, face-down. And a SHORT flick does the same
#    for a different reason: t_launch <= 0.08 s means a linear acceleration of
#    vz / t_launch = 28-56 m/s^2 under a body whose CoM (38% of it is the head)
#    sits ahead of the sole contact, which tips the robot forward over its toes
#    — measured, EVERY cell at t_launch 0.05-0.08 rotates forward, by -88 to
#    -269 deg. That is why the fix for the sweep was a LONGER flick, not a
#    shorter one.
#
# 2. PLATE SWEEP <= 45 deg: the plate may not sweep more than 45 deg of pitch
#    under the robot's feet during the flick.
#
#        sweep = 0.5 * w0 * t_launch          (mdp.backflip_plate_sweep_deg)
#
#    A human hand tossing an object sweeps 30-40 deg — it imparts an impulse
#    and lets go. The shipped box swept 72-110 deg, which is not a hand at all:
#    it is a catapult paddle levering the robot over, and the user watching the
#    video said the robot "takes something like a force that makes it rotate".
#    THIS QUANTITY WAS NEVER REPORTED BY ANY PROBE MODE until 2026-09-09, which
#    is how it survived thirteen measurement waves — every table showed the
#    robot's rotation, the landing speed and the apex, and none showed what the
#    plate itself did.
#
# WHAT THE SWEEP CAP COSTS, measured, and it is not small. With the sweep
# capped, w0 <= 1.571 / t_launch; with t_launch >= ~0.16 s (below that the
# launch flips forward), that means w0 <= ~9.8 rad/s. The robot leaves at
# roughly the plate's final rate, so closing 360 deg needs airtime >= 2*pi/w0
# ~ 0.64 s, which needs vz >= ~3.1 m/s, which lands at 3.7-4.9 m/s. So:
# A HAND-LIKE SWEEP AND OPEN-LOOP 360 DEG CLOSURE ARE INCOMPATIBLE at any
# landing speed the user would accept. The box below therefore closes 0%
# open-loop and the last 50-160 deg must come from the policy's own tuck. That
# is a real bet, stated plainly rather than hidden by widening the cap; the
# alternative (a harder landing) is the last row of the ladder.
#
# The ladder, every row measured with `--box-check --bam --posture standing`
# (rotation and landing are the full 243-cell min-max, closure the fraction
# reaching 360 deg with a FULL TUCK held from the flick):
#     vz        w0      t_launch    sweep deg   rot deg    landing m/s  closure
#     2.2-2.8   18-24   0.14-0.16   72-110 X    98-442     1.78-3.65    27%
#     2.2-2.8   4-7     0.18-0.22   21-44       163-355    2.36-3.95     0%
#     2.2-2.6   4-6     0.20-0.26   23-45       156-310    2.39-3.58     0%
#     2.2-2.6   5-6     0.22-0.26   32-45       198-305    2.37-3.50     0%  <- here
#     3.0-3.6   5-6     0.22-0.26   32-45       261-453    3.73-4.88    29%
# The first row is the SHIPPED-AND-WITHDRAWN box: it fails the sweep gate in
# every one of its 243 cells. The last row is what buying closure back costs.
# All rows pass the direction gate; the chosen row was additionally
# orientation-verified backward at 8/8 of its (vz, w0, t_launch) corners with
# `--check-direction`.
#
# ALSO MEASURED AND REJECTED: pivoting the plate about its REAR EDGE instead of
# its centre (`--pivot rear`), so the surface lifts as it tilts like a
# springboard rather than dropping its front edge. At the same swept angle it
# converts the sweep into LIFT, not into spin: same rotation per unit landing
# speed (188 deg at 2.42 m/s against the centre pivot's 221 at 2.62), a higher
# apex for the same vz, and a WORSE attitude at release (22-41 deg of tilt
# against 11-22). It is not the fix, so the plate still pivots about its
# centre.
HOLD_RANGE    = (1.0, 5.0)     # sampled uniformly from step 0, NO curriculum.
                               # A long, random wait so the policy has to learn
                               # to STAND on the launcher: the first training
                               # run collapsed into the plate before the
                               # impulse, and a later one launched leaning ~35
                               # deg back. EPISODE_LENGTH_S is derived from the
                               # 5.0 s worst case, and backflip_landing pays
                               # over a FIXED window so the hold draw does not
                               # change what a given backflip is worth.
LAUNCH_RANGE  = (0.22, 0.26)   # how long the hands stay with the robot, and
                               # HALF OF THE SWEPT-ANGLE PRODUCT: the plate
                               # sweeps 0.5 * w0 * t_launch under the feet
                               # (mdp.backflip_plate_sweep_deg). It was
                               # 0.14-0.16 with w0 18-24, which swept 72-110
                               # deg — a paddle levering the robot over, which
                               # is what the user saw and called "a force that
                               # makes it rotate". LONGER, not shorter: a short
                               # flick raises vz/t_launch enough to tip the
                               # robot FORWARD over its toes (measured: every
                               # cell at t_launch <= 0.08 comes out forward).
Z0_RANGE      = (0.010, 0.010)  # plate CENTRE = PLATE_HALF_THICKNESS, so the
                               # slab RESTS ON THE FLOOR. Not a range any
                               # more, and that is a MEASUREMENT, not tidying:
                               # the plate is a free body between rewrites (the
                               # step event runs once per CONTROL step and the
                               # physics takes 4 substeps in between), so any
                               # z0 that leaves the slab suspended puts it in
                               # FREE FALL for 20 ms at a time. Measured with
                               # `--plate-jitter`: at z0 >= 0.015 the plate
                               # drops 2.51 mm per control step, reaches
                               # 0.20 m/s, and is teleported back up into the
                               # soles, so the sole force cycles
                               # 15.0 -> 8.4 -> 4.2 -> 1.6 N at 50 Hz (peak
                               # 18.0 N against the robot's 7.85 N weight). At
                               # z0 = 0.010 the floor holds it: deviation
                               # 0.48 mm and 0.0007 deg, |vz| 0.005 m/s, four
                               # steady plate-floor contacts, and a steady
                               # 8.1 N under the soles. The user asked for "une
                               # plateforme immobile" and this is what makes it
                               # one. There is no operator variation to model
                               # here: the ground sets the height.
VZ_RANGE      = (2.20, 2.60)   # lift. Landing speed is the user's binding
                               # constraint: this box lands at 2.37-3.50 m/s
                               # (apex 0.45-0.65 m). Raising it is the ONLY
                               # lever left for open-loop closure now that the
                               # sweep is capped, and it is expensive: vz
                               # 3.0-3.6 buys 29% closure at 3.73-4.88 m/s.
W0_RANGE      = (5.0, 6.0)     # backward flick. Down from 18-24, which is a
                               # 12-16 rad/s flick a hand does not deliver
                               # anyway. The ceiling is the SWEEP GATE:
                               # w0 * t_launch <= 1.571 keeps the sweep under
                               # 45 deg, so at t_launch 0.26 the most w0 can be
                               # is 6.0. The robot leaves at roughly the
                               # plate's final w0, which is why rotation is now
                               # 198-309 deg open-loop instead of 98-442.
# ready_stance's weight and its pose-factor width. Both are chosen from the
# mass table in the module docstring and the measured drift; see the reward
# term's comment for the retraction of the old 1.10 ceiling.
READY_STANCE_WEIGHT = 3.0
POSE_STD = 0.20

MAX_PAID_RATE = 25.0           # rad/s. Deliberately SLACK now: the launcher
                               # only imparts 5-6 rad/s, and a flip that
                               # closes 360 deg in the 0.5-0.6 s of airtime
                               # needs ~11-13 rad/s — the difference is the
                               # policy's own tuck, which is the whole skill.
                               # The cap exists to price genuine violence, and
                               # 25 is above anything the measured box or a
                               # legitimate tuck produces.

# Spawn scatter on the plate. x/y: the plate is 18 cm across and the robot's
# feet span ~8 cm, so 1 cm of jitter is what fits. yaw: the flick axis is world
# +y — a random heading would make it a SIDE flip, so heading noise is small and
# is DR for the operator's aim, not a task variation.
SPAWN_XY_NOISE   = 0.01
SPAWN_YAW_NOISE  = 0.05
SPAWN_TILT_NOISE = 0.02   # rad of roll/pitch — the operator's hands are not level

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


def _apply_final_curriculum(cfg) -> None:
    """Fast-forward every curriculum term to its LAST stage, then remove it.

    WHY THIS EXISTS. ``play`` builds a fresh env with
    ``common_step_counter == 0``, so every curriculum term evaluates at stage
    ZERO no matter which checkpoint is loaded. A viewer therefore showed a
    0.1-0.3 s hold while the policy had been trained on 1-5 s — the user could
    not see the behaviour they had asked for, and any hand-tuned range a
    curriculum widens was equally invisible. Applying the final stage is what a
    play/deploy cfg is for.

    The terms must also be DELETED, not just pre-applied: the curriculum
    manager runs every step and would immediately write stage 0 back.

    Handles the three curriculum shapes this env uses — ``event_param_curriculum``
    (``param_stages``), ``reward_weight`` (``weight_stages``) and
    ``com_range_curriculum`` (``range_stages``). A new shape must be added here
    or it will silently keep its stage-0 value in play.
    """
    for name, term in list(cfg.curriculum.items()):
        params = term.params
        if "param_stages" in params and "event_name" in params:
            last = params["param_stages"][-1]["params"]
            cfg.events[params["event_name"]].params.update(last)
        elif "weight_stages" in params:
            last = params["weight_stages"][-1]["weight"]
            cfg.rewards[params["reward_name"]].weight = last
        elif "range_stages" in params:
            last = params["range_stages"][-1]["range"]
            cfg.events[params["event_name"]].params["ranges"] = (-last, last)
        elif "param_stages" in params and "reward_name" in params:
            last = params["param_stages"][-1]["params"]
            cfg.rewards[params["reward_name"]].params.update(last)
        else:
            raise AssertionError(
                f"curriculum term {name!r} has no stage list this function "
                "understands; teach it the new shape rather than letting play "
                "silently run at stage 0"
            )
        del cfg.curriculum[name]


def make_microduck_backflip_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the Microduck backflip environment configuration.

    ``play=True`` fast-forwards every curriculum to its FINAL stage and drops
    the curriculum terms, so the viewer shows the env the policy was actually
    trained on (most visibly the 1-5 s hold instead of 0.1-0.3 s). The training
    path is untouched.
    """

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
    # The robot entity is the standup one with a BACKFLIP-SPECIFIC initial
    # height: standing on the plate top, at the middle of Z0_RANGE. That init
    # state is normally irrelevant (every episode's reset events overwrite it)
    # — except for the FIRST episode under `uv run play`, which mjlab runs
    # without ever calling reset (see _backflip_state's comment). Before this,
    # play's first 4 s showed a robot standing on the floor with the plate
    # hovering through its body, which got reported as broken. deepcopy so
    # standup/roulade keep MICRODUCK_STANDUP_ROBOT_CFG's own init.
    _z0_mid = 0.5 * (Z0_RANGE[0] + Z0_RANGE[1])
    backflip_robot_cfg = deepcopy(MICRODUCK_STANDUP_ROBOT_CFG)
    backflip_robot_cfg.init_state = deepcopy(backflip_robot_cfg.init_state)
    backflip_robot_cfg.init_state.pos = (
        0.0, 0.0, _z0_mid + PLATE_HALF_THICKNESS + STAND_Z,
    )
    cfg.scene.entities = {
        "robot": backflip_robot_cfg,
        "plate": MICRODUCK_LAUNCHER_CFG,
    }
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, robot_ground_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

    # Contact budget, MEASURED not guessed (AGENTS.md: contact overflow presents
    # as sudden NaN, so this was the first suspect for the 2026-09-08 crash and
    # it was RULED OUT). Peak simultaneous contacts on CPU MuJoCo with the
    # full-collision robot and the plate in the scene: 16 standing on the plate,
    # 3-10 sprawled on the floor for 5 s in every orientation a flip can end
    # in, plus 4 persistent plate-floor contacts once z0 puts the slab on the
    # ground — so ~20 worst case. 50 leaves 2.5x headroom; roulade uses 35 with
    # no prop and sitstand 200. A test pins the floor.
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
        params={
            "target_angle":      2 * math.pi,
            "max_paid_rate":     MAX_PAID_RATE,
            "launch_gate_floor": LAUNCH_GATE_FLOOR,
            "hold_z":            STAND_Z + PLATE_HALF_THICKNESS,
        },
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
            "window_s":    LANDING_WINDOW_S,
            "launch_gate_floor": LAUNCH_GATE_FLOOR,
            "hold_z":            STAND_Z + PLATE_HALF_THICKNESS,
            "sensor_name": feet_ground_cfg.name,
        },
    )

    # STAND STILL AND STRAIGHT on the operator's hands, HOLD-phase only. The
    # user's own words for the whole ask: "je veux juste qu'il reste immobile
    # droit sur une plateforme immobile". This is the term that pays for it.
    #
    # window x height x upright x pose:
    #   height  — "be on the plate at standing height" (and the thing that
    #             prices stepping OFF it: the floor beside the plate is ~3
    #             sigma down)
    #   upright — "be upright, not leaning" (10/45 deg)
    #   pose    — "and STRAIGHT": Gaussian on the mean squared joint error
    #             against the spawn pose, std 0.20. New. The other two factors
    #             say nothing about the joints, so a robot sagging into a
    #             different leg configuration at the same trunk height and tilt
    #             used to score identically.
    #
    # WEIGHT 3.0, up from 1.10, and the CEILING THAT KEPT IT AT 1.10 IS
    # RETRACTED. That ceiling came from "a stance worth more than the landing
    # annuity makes 'stand still and never flip' the argmax". The robot CANNOT
    # decline to flip: backflip_plate_step is a mode="step" event that fires on
    # its prescribed schedule whatever the policy does. There is no "never
    # flip" strategy to farm, so there was never a ceiling — the argument was
    # invalid and it blocked the obvious fix for three revisions. The only way
    # to dodge the flick is to walk off the plate, which pays nothing (height
    # collapses, the launch gate never latches, the annuity is gated on a flip
    # that never happens). See the function docstring for the full retraction;
    # do not reinstate a ceiling without first checking whether your reason
    # requires the robot to be able to REFUSE the flick.
    #
    # 3.0 makes the hold a co-equal major term rather than the task's
    # third-smallest positive one: mass 3.0-15.0 across the 1-5 s hold, median
    # 9.0, against the flip's 8.0 and the annuity's 5.6. That is what "worth
    # defending" means for a skill the robot has to hold for up to five
    # seconds. See the module docstring's table.
    #
    # WHY IT HAS TO BE MAJOR. A motionless hold on this robot is ACTIVE
    # BALANCING, measured with the ideal command frozen: tilt 3.5 deg at 0.3 s,
    # 7.3 at 0.5, 11.6 at 0.7, 23.2 at 1.0, toppled by 1.5 s — and IDENTICAL on
    # the bare floor, so it is the pose, not the launcher. At 1.10 the hardest
    # sustained requirement in the task was worth 1.1-5.5.
    #
    # THE GATE IS 10/45 deg, NOT 40/70. The wide 40/70 pair was sized for the
    # short-lived TUCKED hold, whose own equilibrium is pitched 14 deg, and it
    # made a LEAN FREE: the user saw the robot launch leaning ~35 deg back and
    # this term charged nothing for it, because 35 < 40. At 10/45 a genuinely
    # upright stance still costs nothing (the measured drift is 3.5 deg at
    # 0.3 s, 7.3 at 0.5), a 20 deg lean loses 20% of the term, a 35 deg lean
    # loses 80%, and every flop basin (80-126 deg) is still hard-zeroed.
    #
    # The upright factor is NOT optional and was dropped once, during the
    # tucked-hold experiment. height alone cannot tell an upright robot from an
    # inverted one at the same trunk height, and the side-lying basin --
    # passively stable, needing no balancing, costing less action_rate --
    # outscored the intended pose by 4%. The same trap applies to `pose`, which
    # is why it is a FACTOR and never an additive term of its own: measured
    # open loop, the joint error against HOME is LOWER after the robot has
    # toppled (rms 0.056-0.076 rad lying down against 0.086 upright at 1.0 s),
    # because a servo that is not fighting gravity sits closer to its command.
    # Re-run `--flop-audit` after touching any factor here.
    #
    # stand_z carries the plate half-thickness: the term measures trunk height
    # against the plate CENTRE (z0) while the robot stands on its top surface.
    cfg.rewards["ready_stance"] = RewardTermCfg(
        func=microduck_mdp.backflip_ready_stance,
        weight=READY_STANCE_WEIGHT,
        params={
            "stand_z":       STAND_Z + PLATE_HALF_THICKNESS,
            "height_std":    0.03,
            "tilt_full_deg": 10.0,   # free while genuinely upright
            "tilt_zero_deg": 45.0,   # << the 80-126 deg flop basins
            "pose_std":      POSE_STD,
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

    # Arrival damper — trunk omega_xy^2 inside an arrival WINDOW around standing
    # height, and only at low tilt. Introduced at 0, ramped by curriculum.
    #
    # The height band needs BOTH edges here, and the reason is a correction:
    # body_ang_vel_at_height's height_low/height_high pair is a FLOOR, not a
    # window — everything above height_high pays FULL cost. The whole flight of
    # a backflip is above 0.11 m (apex 0.29-0.40 m measured in the new box), so
    # the original "gated on standing height AND low tilt, so the flip itself
    # is never taxed" was wrong twice over: only the tilt gate was protecting
    # the flip, and a rotating robot passes tilt < 20 deg twice per revolution.
    # 0.121 / 0.132 keeps the landed STANDING trunk (~STAND_Z = 0.115) inside
    # the full-cost band and the flight outside it.
    #
    # HOLD-PHASE DAMPING IS NOW DELIBERATELY ACCEPTED. When the plate sat at
    # z0 = 0.10-0.20 the tucked hold trunk was 0.139-0.249 m and the ceiling
    # could separate the two cleanly. The retune dropped the plate to
    # z0 = 0.07-0.09, so the hold trunk is 0.109-0.129 m — i.e. straddling
    # STAND_Z. No ceiling can separate "held tuck" from "landed stand" by
    # HEIGHT any more; they are the same height. The term therefore also acts
    # during HOLD. That is priced and small: a held tuck's trunk omega_xy is
    # ~0 (measured x/y drift over 3 s is 8-11 mm), the weight is 0 until
    # iteration 2000, and what it does charge for — thrashing while waiting on
    # the launcher — is behaviour this env wants suppressed anyway. If a run
    # shows ready_stance being fought, gate this term on
    # backflip_phase == GONE rather than reaching for the height numbers.
    cfg.rewards["arrival_damping"] = RewardTermCfg(
        func=microduck_mdp.body_ang_vel_at_height,
        weight=0.0,
        params={
            "height_low":      0.09,
            "height_high":     0.11,
            "height_full_max": 0.121,   # > STAND_Z: the landing IS damped
            "height_zero_max": 0.132,   # < the flight; NOT below the hold
            "tilt_full_deg":   20.0,
            "tilt_zero_deg":   45.0,
            "asset_cfg":       SceneEntityCfg("robot", body_names=("trunk_base",)),
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

    # The critic's SENSOR-DERIVED terms are the one obs path `nan_state` cannot
    # protect in time: it checks joint + root state (and now contact forces),
    # but obs is computed in the same step, and a sensor can report non-finite
    # while the state is still clean. A single NaN here kills the whole run via
    # rsl_rl's check_nan — the 2026-08-21 Velocity2 crash and the 2026-09-08
    # backflip crash are both this. Critic-only, so sanitizing costs the policy
    # nothing. The velocity cfg does the same; this env builds on mjlab's base
    # template and so did not inherit it.
    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_height",         microduck_mdp.foot_height_safe),
        ("foot_air_time",       microduck_mdp.foot_air_time_safe),
    ):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

    # CRITIC-ONLY launcher state (asymmetric actor-critic). The actor stays
    # plate-blind — there is no launcher sensor on the real robot — while the
    # critic gets the plate's relative pose/velocity and the phase, which is
    # what makes the value function's job tractable across a randomized toss.
    #
    # The terms are the ball-kick env's critic-only ball terms wrapped in a
    # phase mask (backflip_plate_pos_obs / _vel_obs) that ZEROES them once the
    # plate is GONE. That mask is not about hiding information, it is about the
    # obs normalizer: the plate sits ~8.7 m away for ~85% of every episode's
    # steps, so an unmasked plate_position normalizer converges to std ~3 m and
    # squashes the informative HOLD/LAUNCH range (0-0.3 m) into under 0.1
    # normalized units, destroying the z0 signal the value function needs to
    # price a randomized toss. Do NOT add a plate term to the actor group,
    # masked or not.
    cfg.observations["critic"].terms["plate_position"] = ObservationTermCfg(
        func=microduck_mdp.backflip_plate_pos_obs, params={"asset_name": "plate"},
    )
    cfg.observations["critic"].terms["plate_velocity"] = ObservationTermCfg(
        func=microduck_mdp.backflip_plate_vel_obs, params={"asset_name": "plate"},
    )
    cfg.observations["critic"].terms["plate_phase"] = ObservationTermCfg(
        func=microduck_mdp.backflip_phase_obs,
    )
    # SECONDS OF HOLD LEFT — critic-only, and it earns its place now that
    # `ready_stance` is a major term paying per second of holding. The hold is
    # drawn uniformly 1-5 s, so the stance is worth 3.0-15.0 points depending
    # on a draw the value function has no other way to see: a static plate
    # looks identical at t = 1 s and t = 4 s, and `plate_phase` only says
    # "still holding". Without this the whole 5x spread lands in the advantage
    # as noise. Privileged information the real robot does not have, which is
    # exactly what the critic group is for.
    cfg.observations["critic"].terms["hold_remaining"] = ObservationTermCfg(
        func=microduck_mdp.backflip_hold_remaining_obs,
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
    # NaN guard. `sensor_names` IS LOAD-BEARING and its absence caused the
    # 2026-09-08 crash (run 2026-09-08_16-26-30_backflip): rsl_rl's check_nan
    # found NaN in the CRITIC group while Episode_Termination/nan_state read
    # exactly 0.0000, i.e. this guard never fired. mjlab computes observations
    # AFTER _reset_idx (manager_based_rl_env.step), so anything this term
    # checks would have reset the env and returned clean obs — the NaN
    # therefore came through a quantity it does NOT check. Contact FORCES are
    # such a quantity: MuJoCo resolves a degenerate contact into an inf/NaN
    # impulse a step before the integrated state goes bad, and that force
    # feeds the critic's `foot_contact_forces` obs. The velocity cfg has
    # passed these sensors since its own 2026-08-21 crash; this env builds on
    # mjlab's make_velocity_env_cfg rather than the microduck one, so it
    # inherited neither the sensor list nor the _safe swap below.
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={
            "sensor_names": (
                feet_ground_cfg.name,
                self_collision_cfg.name,
                robot_ground_cfg.name,
            )
        },
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
    # backflip_launch_params (it reads the sampled z0); see the function's
    # docstring for why an independent spawn height is not an option, and why
    # the hold posture is standing rather than tucked.
    #
    # These are the SAME two numbers ready_stance scores against; a cfg test
    # asserts that, so editing one alone fails instead of silently spawning the
    # robot at a height its own hold reward calls wrong.
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
    #
    # t_override=0.0 is REQUIRED here, not decoration: episode_length_buf is
    # zeroed AFTER reset events run, so without it this event reads the
    # terminal episode's time, the phase comes out GONE, and it parks the plate
    # 5 m away instead of at z0 — for one control step after any manual
    # reset(env_ids=...), which is what play/eval do. See the function's
    # docstring.
    cfg.events["backflip_plate_reset"] = EventTermCfg(
        func=microduck_mdp.backflip_plate_step,
        mode="reset",
        params={"asset_name": "plate", "t_override": 0.0},
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

    # Tighten the launch-attitude gate's FLOOR once the stance consolidates.
    # Phase-aligned per AGENTS.md: at step 0 a bad stance still yields 30% of
    # the flip signal, which is what lets the flip be discovered at all while
    # the robot is still learning to balance on the launcher. Only after that
    # does a collapse become nearly worthless. If wandb shows ready_stance
    # flat while flip_progress stalls at a stage boundary, this ramp is too
    # early — stretch it, never move it earlier.
    #
    # BOTH reward terms must be moved together: the gate is one latched value
    # and applying different floors to flip_progress and landing would leave
    # part of the task collectable from a collapse. A cfg test pins them equal.
    for _term in ("flip_progress", "landing"):
        cfg.curriculum[f"launch_gate_floor_{_term}"] = CurriculumTermCfg(
            func=microduck_mdp.reward_param_curriculum,
            params={
                "reward_name": _term,
                "param_stages": [
                    {"step": 0, "params": {
                        "launch_gate_floor": LAUNCH_GATE_FLOOR}},
                    {"step": 2000 * 24, "params": {
                        "launch_gate_floor": 0.15}},
                    {"step": 4000 * 24, "params": {
                        "launch_gate_floor": LAUNCH_GATE_FLOOR_FINAL}},
                ],
            },
        )

    # NO HOLD CURRICULUM. HOLD_RANGE is sampled uniformly 1-5 s from step 0.
    #
    # There used to be a ramp (0.1-0.3 -> 1-2 -> 1-3.5 -> 1-5) whose only job
    # was to keep early episodes from spending most of their time standing
    # still. It bought little and cost a lot: the first real run reached
    # landing +1.72 by iteration 279, so flip discovery was never fragile, and
    # a curriculum on the hold meant `play` — which starts
    # common_step_counter at 0 — showed a 0.1-0.3 s hold whatever checkpoint
    # was loaded, so the long hold could not be seen in the viewer at all.
    # Sampling the final range from step 0 removes both problems.

    # NO launch-height DR tail. There used to be one, widening z0 from the
    # measured box toward the spec's 0.30 m "operator tail". It is gone, and
    # both ends of the argument are measured:
    #   * upward, landing speed is the binding constraint — at the old box the
    #     whole-box worst landing was 2.51 / 2.53 / 2.57 / 2.59 m/s at
    #     z0 = 0.200 / 0.210 / 0.215 / 0.225 against a ~2.6 m/s hardware limit;
    #   * downward, Z0_RANGE now spans 2 cm total, because the pose's own
    #     geometry floors it at 0.07 and the retune wants it as low as possible.
    # A curriculum stage that widens a 2 cm range is not worth its pacing risk.
    # If the hold posture ever changes, re-measure before adding one back.

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

    if play:
        _apply_final_curriculum(cfg)

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
