"""Microduck backflip task — launched by the operator's hands, land on the feet.

STATUS. First training run (4096 envs, 279 iterations) worked: flip_progress
+1.97, landing +1.72, every penalty negative, and the 300 deg landing gate
opening — so the env does produce real backflips. It also collapsed into the
plate before the impulse, because the hold was 0.1-0.3 s and ready_stance
could only ever earn 0.1-0.3 (it logged +0.036). This revision answers that:
the plate now rests on the ground, the hold is 1-5 s (curriculum-ramped), the
episode is 7.5 s to fit it, and the stance weight is re-derived by reward MASS.
Open-loop the robot lands at 2.4-4.9 m/s, above the ~2.6 m/s the operator
would like; a trained policy that tucks to spin faster and extends to brake
before contact may do better, and that — along with how much of the launch box
it can complete — is answered by training rather than by more probing.

WHY THE PLATE SITS ON THE GROUND NOW. An earlier revision floored Z0_RANGE at
0.07 m because the then-current KNEELING tuck rested on its shins with its feet
hanging ~8 cm below the slab, so a lower plate put them through the floor. That
posture is gone. Standing's LOWEST GEOMS ARE THE FEET, so the plate can rest on
the ground (z0 = PLATE_HALF_THICKNESS puts the slab exactly on the floor) and
the spawn still puts the soles on its top surface by construction. The honest
cost is altitude: a floor-level launch has less airtime, nothing closes at
vz 2.5 any more, and VZ_RANGE had to come up to 3.0-4.0 — which lands harder.

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

MEASURED LAUNCH ENVELOPE — WHOLE-BOX verified from the ACTUAL spawn
(docs/backflip_envelope_results.md "Tucked hold"; BAM actuators, probe mode
``--posture tucked_env --box-check``. These numbers are measured, not guessed
— do not "tidy" them):
  z0 in [0.10, 0.20] m, vz in [2.00, 2.10] m/s, w0 in [21, 23] rad/s,
  t_launch in [0.12, 0.14] s. All 486 sampled combinations of the CORNERS AND
  MIDPOINTS of those four ranges — crossed with the hold extremes (0.1 s and
  the curriculum's 1.0 s) and tuck depths 0.5 / 0.75 / 1.0 — close a full
  360 deg BACKWARD flip and land under the hardware limit. Worst cell in the
  box: 393.6 deg (33.6 deg of margin) and 2.51 m/s (0.09 m/s of margin). All
  16 (z0, vz, w0, t_launch) corners plus the two worst cells and both tuck
  extremes were orientation-verified backward (20/20).
  WHOLE-BOX is the acceptance rule, and it is not pedantry: the previous box
  was picked from best corners and its interior contained cells that rotated
  3 deg. If you widen any range, re-run --box-check; a box with one dead cell
  inside it is not a box.
  * A SHORT flick is the violent one: t_launch=0.08 lands at up to 3.97 m/s,
    half again over the limit. The operator's flick duration is not free DR.
  * vz above 2.10 pushes the z0=0.20 corner to 2.65-2.70 m/s.
  * w0 above 23 runs out of AIRTIME rather than spin: w0=25 at t_launch=0.14
    drops to 372 deg and w0=30 to 279 deg, because a harder flick trades apex
    for rotation rate.
  * The tuck depth is fixed at TUCK_FACTOR for the SPAWN, but the box was
    verified across 0.5-1.0 because the policy can deepen or open the tuck
    during HOLD and LAUNCH.
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
  * REWARD MASS AT EVERY CURRICULUM STAGE (episode sums, dt-scaled —
    AGENTS.md: compare mass, not weight; and price every stage, not just the
    last). With hold H, launch ~0.14 s and a measured 0.42-0.88 s flight, the
    post-landing settle is S = 7.5 - H - 0.16 - 0.88 = 6.46 - H seconds
    (the WORST-CASE flight, so the bounds below hold everywhere):

      stage   hold H      ready_stance   landing        flip  never-flip cap
      0       1.0-2.0 s   1.12 - 2.24    17.8 - 21.8    8.0   2.24
      1       1.0-3.5 s   1.12 - 3.92    11.8 - 21.8    8.0   3.92
      2       1.0-5.0 s   1.12 - 5.61     5.84 - 21.8   8.0   5.61

    (ready_stance = 0.975 x 1.15 x H; landing = 4.0 x S, with S from the
    WORST-CASE 0.88 s flight so the bound holds everywhere.)

    THE ACCEPTANCE STATEMENT, checked at every stage:
      1. Collapsing during the hold costs more than the rotation it buys.
         Collapsing forfeits the whole stance mass — at worst 1.12, at best
         5.61 — where the previous curriculum's first stage forfeited only
         0.1-0.3. And the rotation a pre-flick crouch buys is bounded: the
         policy can tuck AT the flick and get the same compactness for free
         (ready_stance dies at launch), and flip_progress is capped at one
         turn, so extra rotation beyond 360 deg pays nothing.
      2. Flipping and landing still beats never flipping, at every stage. The
         never-flip cap is the stance mass alone (2.24 / 3.92 / 5.61); flipping
         adds flip 8.0 + landing >= 5.84, i.e. at least 13.84 on top.
      3. `landing` stays the dominant attractor at every stage: its minimum
         (5.84, at the longest hold) still exceeds ready_stance's maximum
         (5.61).

    KNOWN CONSEQUENCE: the landing mass swings 3.7x across the hold DR (5.84 at
    H=5, 21.8 at H=1), which is noisy credit assignment. If that shows up as
    instability, cap the annuity's paying window rather than reaching for the
    weights.
  * The landing annuity (weight 4.0) is gated on a near-complete flip (300-345
    deg): "stand still and never flip" satisfies feet/upright/height/calm
    trivially, and without the gate it is the argmax. Reward MASS (episode
    sums, dt-scaled): flip 8.0, landing up to 4.0 x ~2 s of post-landing
    annuity = ~8.0, ready_stance 0.4-1.0 (weight x the hold, which the
    curriculum widens from 0.4 s to 1.0 s). So a flip-and-crash earns ~8 and a
    flip-and-land earns ~16 — the landing is worth a second flip, and the hold
    is worth 5-12% of one.
  * ``ready_stance`` (weight 1.0) pays only during HOLD and dies at launch, so
    it can never oppose the flip. It pays ``pose x height`` for HOLDING THE
    TUCK: the joint Gaussian says "be folded", the height Gaussian says "be
    folded ON THE PLATE" (a standing trunk is ~2.9 sigma out, scoring ~3e-4),
    and a WIDE tilt smoothstep (full below 40 deg, zero above 70) says "be
    folded UPRIGHT". Its ``tuck_z`` is TUCK_Z + PLATE_HALF_THICKNESS because
    the robot rests on the plate's TOP surface while the term measures against
    the plate's centre height ``z0``. The upright factor is load-bearing: pose
    and height cannot tell an upright tuck from an inverted one, and the
    SIDE-LYING tuck is a passively stable on-plate basin that measured 0.991
    against upright's 0.950 without it — a premium for flopping, which would
    have produced a SIDE flip. It costs zero at the tuck's own 14 deg resting
    tilt, so it does not fight the pose. It pays a crouched robot per step,
    which is the shape AGENTS.md warns about; the function's docstring argues
    why that is legitimate here (the tuck is the GOOD state, and the paying
    window is closed by the plate's PRESCRIBED schedule rather than by anything
    the policy does, so it cannot be camped).
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

SPAWN. ``reset_backflip_robot_on_plate`` folds the robot to
TUCK_OVERRIDES x TUCK_FACTOR and derives its spawn height from the SAME ``z0``
the plate uses, plus the MEASURED tucked resting height TUCK_Z. It runs after
``backflip_launch_params`` (which samples ``z0``) and after
``reset_robot_joints`` (whose +-0.05 rad of joint scatter it shifts rather than
overwrites), events firing in dict insertion order. The base template's +-0.5 m x/y scatter
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

# Episode budget, MEASURED rather than guessed: hold (up to 5.0 s once the
# curriculum has widened it) + launch ramp (<= 0.16 s) + the airborne window
# (0.42-0.88 s measured across the box, worst case at vz = 4.0) + settle. At
# 7.5 s the worst case leaves 7.5 - 5.0 - 0.16 - 0.88 = 1.46 s to settle on the
# feet, and a short hold leaves over 6 s.
EPISODE_LENGTH_S = 7.5

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
# THE ONE HARD LIMIT IS DIRECTION. Above roughly w0 = 24-27 rad/s from a
# standing hold the flick overdrives the sole contact and the robot comes out
# FORWARD, face-down — a different maneuver that the rotation accumulator would
# have to be re-signed to even score. The w0 ceiling stays inside that measured
# boundary with margin. Every corner of this box is direction-checked backward;
# re-check with `--box-check` if you move it.
HOLD_RANGE    = (1.0, 5.0)     # the END state, and the user's requirement:
                               # a long, random wait so the policy has to learn
                               # to STAND on the launcher rather than collapse
                               # into it. The first training run collapsed
                               # before the impulse (ready_stance +0.036
                               # against a weight of 1.0) precisely because a
                               # 0.1-0.3 s hold made collapsing almost free.
                               # A curriculum ramps into it — see the comment
                               # on backflip_hold_range below — because at a
                               # 5 s hold in a 7.5 s episode ~70% of collected
                               # experience is standing still, which slows the
                               # flip's discovery badly. The ramp is only for
                               # early discovery; 1-5 s is where it must end.
LAUNCH_RANGE  = (0.12, 0.16)   # how long the hands stay with the robot
Z0_RANGE      = (0.01, 0.03)   # plate CENTRE. 0.01 = PLATE_HALF_THICKNESS, so
                               # the slab rests exactly on the floor; the DR
                               # spread is operator variation. The old 0.07-0.09
                               # floor came from the KNEELING tuck, whose feet
                               # hung ~8 cm below the slab; standing's lowest
                               # geoms ARE the feet, so the plate can sit on the
                               # ground and a test asserts nothing tunnels.
VZ_RANGE      = (3.00, 4.00)   # lift. RAISED from 2.50-3.50 because the lower
                               # plate costs altitude: from a floor-resting
                               # plate nothing closes at vz 2.5 (best 334 deg),
                               # while 3.0-4.0 closes across most of the w0
                               # range. The honest cost is landing speed —
                               # 2.4-4.9 m/s open-loop, worse than before.
W0_RANGE      = (15.0, 24.0)   # backward flick; 24 keeps clear of the
                               # measured 24-27 direction reversal
MAX_PAID_RATE = 25.0           # rad/s; measured peak in the box is 23.0,
                               # mean over a flip 15.6-20.1 — 25 forfeits
                               # nothing a real flip needs

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
        if "param_stages" in params:
            last = params["param_stages"][-1]["params"]
            cfg.events[params["event_name"]].params.update(last)
        elif "weight_stages" in params:
            last = params["weight_stages"][-1]["weight"]
            cfg.rewards[params["reward_name"]].weight = last
        elif "range_stages" in params:
            last = params["range_stages"][-1]["range"]
            cfg.events[params["event_name"]].params["ranges"] = (-last, last)
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

    # Stand still on the operator's hands. HOLD-phase only — it dies at
    # launch, so it can never oppose the flip. upright x height: the height
    # Gaussian says "be on the plate at standing height", the WIDE tilt
    # smoothstep says "be upright".
    #
    # The upright factor is NOT optional and was dropped once, during the
    # tucked-hold experiment. height alone cannot tell an upright robot from an
    # inverted one at the same trunk height, and the side-lying basin --
    # passively stable, needing no balancing, costing less action_rate --
    # outscored the intended pose by 4%. Re-run `--flop-audit` after touching
    # any factor here.
    #
    # WEIGHT 1.15, chosen by MASS at EVERY CURRICULUM STAGE (AGENTS.md), not
    # just at the end. See the module docstring's per-stage table. The ceiling
    # is set by the long end — at H=5 the landing annuity is only 5.84 (the
    # settle window is shortest there, and this uses the WORST-CASE 0.88 s
    # flight, not the typical 0.65) and `landing` must stay the dominant
    # attractor, so 0.975*w*5 <= 5.84 gives w <= 1.19. The floor is set by the
    # SHORT end, which
    # is why the hold curriculum now starts at 1.0 s rather than 0.1: at a
    # 0.1-0.3 s hold no admissible weight makes this term matter (it caps at
    # 0.3*1.4 = 0.42 against ~30 for flip+landing), which is how the policy
    # came to crouch before the impulse.
    #
    # stand_z carries the plate half-thickness: the term measures trunk height
    # against the plate CENTRE (z0) while the robot stands on its top surface.
    cfg.rewards["ready_stance"] = RewardTermCfg(
        func=microduck_mdp.backflip_ready_stance,
        weight=1.15,
        params={
            "stand_z":       STAND_Z + PLATE_HALF_THICKNESS,
            "height_std":    0.03,
            "tilt_full_deg": 40.0,   # standing is ~0 deg: this costs nothing
            "tilt_zero_deg": 70.0,   # < the 91-103 deg flop basins
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

    # Hold-window widening, to HOLD_RANGE = 1-5 s.
    #
    # THE FLOOR IS 1.0 s AT EVERY STAGE, and that is the whole point. An
    # earlier revision started at 0.1-0.3 s to keep early episodes cheap, and
    # the policy learned to CROUCH before the impulse — rationally, because a
    # compact body rotates much further at the same flick, and at a 0.1-0.3 s
    # hold ready_stance's episode-sum mass was 0.1-0.3 against ~30 for
    # flip+landing. The term was sized for the END of the curriculum and the
    # behaviour is acquired at its START. Curriculum stages must be priced at
    # EVERY stage; the per-stage table is in the module docstring.
    #
    # Starting at 1-2 s is affordable now: the first real run reached
    # landing +1.72 by iteration 279 and open-loop closure is 62%, so flip
    # discovery is not fragile, and the cheap-hold window was buying less than
    # it cost. What the ramp still buys is the range WIDTH — a narrower early
    # window keeps each episode's launch at a similar time so the policy can
    # learn what the flick FEELS like before it has to handle 5 s of timing
    # uncertainty. Phase-aligned with the flip existing first.
    cfg.curriculum["backflip_hold_range"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "backflip_launch_params",
            "param_stages": [
                {"step": 0,         "params": {"hold_range": (1.0, 2.0)}},
                {"step": 1500 * 24, "params": {"hold_range": (1.0, 3.5)}},
                {"step": 3000 * 24, "params": {"hold_range": HOLD_RANGE}},
            ],
        },
    )

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
