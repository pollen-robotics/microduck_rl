"""Hop task variant — a phase-commanded periodic hop in place.

``make_hop_variant(cfg)`` converts a velocity env cfg into a hop task, in
the same shape as ``tasks/run.py`` and ``tasks/backlash.py``. Composed as
``make_sprung_variant(make_hop_variant(cfg), ...)`` so the sprung machinery is
reused unchanged.

Periodic rather than a one-shot jump because the spring's energy comes
overwhelmingly from IMPACT loading: a drop-rig probe measured a 100 mm drop
rebounding 33 mm, while quasi-static actuator loading only just reaches the
49.7 N needed for full travel (52.4 N available, knee-limited) before BAM
back-EMF derates it at launch speed. Each landing charges the next launch.

Six changes:

1. Replace the twist command with the CYCLIC phase command already on develop.
2. Retarget the ported height reward onto RISE ABOVE TAKEOFF HEIGHT, which
   removes the standing-height datum from the reward path entirely.
3. Drop the forward-locomotion rewards (this is a hop in place) AND the walking
   `air_time` reward, which outscores hopping with a march in place.
4. Register the three hop rewards, the LOAD-PHASE reward and the energy monitor.
5. Raise the `com_height_target` band's UPPER edge, which otherwise puts a step
   penalty inside the height range the experiment needs explored.
6. Silence `com_height_target` during the launch half — it was the single largest
   reward for standing perfectly still, which is what the first sweep learned.

Steps 4 (the load reward) and 6, together with the 4x on the hop weights, are the
rebalance that followed that null. See the budget comment beside the weights.
"""

import dataclasses
import os

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.robot.microduck_constants import get_allcollisions_spec
from mjlab_microduck.robot.sprung_foot import SPRING_JOINTS, SPRING_PRELOAD
from mjlab_microduck.tasks import mdp as microduck_mdp

# Seconds per hop cycle. The spring-mass period at k=3900 and 0.877 kg is
# 2*pi*sqrt(m/k) = 94 ms, so this is deliberately well above it: the cycle must
# accommodate a load segment, a launch, a flight and a landing, not just one
# spring oscillation. Provisional -- sweep only if the first result is ambiguous.
HOP_PERIOD = 1.0

# NO STANDING-HEIGHT DATUM IS ON THE REWARD PATH ANY MORE, and that is the
# single most important thing to know about this file's history.
#
# Both airborne-gated hop rewards now measure RISE ABOVE TAKEOFF HEIGHT (see
# `microduck_mdp._HopRiseTracker`). Absolute base height was tried, and it failed
# three times over: it needed a standing-height datum that was wrong twice
# (0.120 was the robot_walk.xml SPAWN height, not a standing height; then the
# SETTLED height double-counted 7.6 mm of actuator sag against a flight-only
# reward), and even with the datum right it stayed farmable -- first by foot
# flutter, then by a tall-tuck that no threshold can exclude. Rise is measured
# against the robot's own body milliseconds earlier, so no datum, no h_add and
# no posture correction enters the reward at all.
#
# THE MEASUREMENTS ARE KEPT because they still justify the CoM band and are the
# reason the frame changed:
#
#   settled rigid stand height   0.1095   (MEASURED 2026-08-24; the registered
#       LOCKED arm compiled onto a floor plane in HOME_FRAME, base pinned to
#       vertical travel only -- it topples in ~1 s otherwise and an unpinned
#       settle measures tipping -- settled 3000 steps, root z 0.1395, minus
#       H_ADD 0.030. An 80 s settle creeps to 0.1035, so the settled value is
#       inside [0.1035, 0.1171] on any reading.)
#
#   sag-free kinematic height    0.12114  (UNLOADED_RIGID_HEIGHT below: upright,
#       HOME_FRAME, lowest pad corner exactly on the floor, h_add removed.
#       Was 0.1171 before the 2026-09-04 footprint correction; ANKLE_TO_SOLE
#       moved +3.53 mm with it, which is where the 4 mm went.)
#
#   max sag-free stance height   0.16133  (grid search over symmetric
#       hip_pitch/knee/ankle poses, pad kept flat, vs HOME_FRAME's 0.14710.)
#
# That last pair is what killed absolute height: ~14.2 mm of POSTURE HEADROOM.
# The robot can stand tall at root ~0.155 and then tuck; a 51 ms tuck dips the
# trunk only ~4 mm, so it sits above any threshold set within 14.2 mm of stance
# for the whole tuck -- ~0.20/step of `pose` spent to unlock ~1.8/step of
# airborne reward, scoring 5.1/step against a genuine 33 mm hop's 5.4. And the
# threshold could not simply be raised past it: anything above 14.2 mm would
# also gate out the Locked arm's expected ~5 mm hop and destroy the controlled
# comparison that IS the experiment.

# The SAG-FREE KINEMATIC height (HOME_FRAME, upright, lowest pad corner exactly
# on the floor), rigid, i.e. with h_add removed. Pinned against the compiled
# Locked arm by a test in tests/test_hop_cfg.py.
#
# NOT a reward datum -- nothing in the reward path reads it. It survives as the
# STANCE reference that justifies `HOP_COM_HEIGHT_MAX`: the CoM band is about
# standing behaviour, so it still needs to know how tall the robot stands.
UNLOADED_RIGID_HEIGHT = 0.12114

# RISE above takeoff height that the Gaussian peaks at, and its width.
#
# These two transfer UNCHANGED from the absolute-height formulation, and they
# are more correct in the rise frame, not less: both were always chosen as a
# GAIN ABOVE STANDING, which is exactly what rise measures directly. Under the
# old formulation that gain had to be reconstructed by adding a standing-height
# datum to an absolute target, and the datum was wrong twice.
#
# Both were raised from 0.015 / 0.008: the old pair saturated the reward
# at roughly 15-20 mm of gain, while the drop-rig evidence this campaign is built
# on spans 5 mm (Locked) to 33 mm (k3900), so all three arms sat at the ceiling
# and the arm-to-arm comparison -- which IS the experiment -- measured nothing.
#
# Why 40 mm. Physical ceiling estimate at k=3900: full travel stores
# 0.5*3900*0.012**2 + 2.9*0.012 = 0.3156 J per foot (the second term is work
# against the k*SPRING_PRELOAD ~ 2.9 N preload), so 0.631 J for both feet. At
# zeta = 0.3 a damped oscillator returns exp(-pi*zeta/sqrt(1-zeta**2)) = 0.372 of
# that, i.e. ~0.234 J, which lifts 0.877 kg by 27 mm. Actuator work adds on top.
# So 40 mm sits deliberately just ABOVE the sprung expectation, which keeps the
# whole 5-33 mm discriminating band on the RISING limb of the Gaussian instead of
# straddling its peak (where 27 mm and 53 mm would score alike).
#
# Why std = 20 mm. `hop_body_height` uses exp(-((rise - target)/std)**2), so with
# target +40 mm and std 20 mm the term reads: 5 mm rise -> 0.047, 27 mm -> 0.655,
# 33 mm -> 0.885. Monotone increasing across the band, ~19x discrimination
# between a Locked-like 5 mm and a sprung-like 33 mm. Under the old std = 0.008
# all three of those read ~0.000 -- indistinguishable, which was the bug.
#
# The tradeoff, deliberately accepted: at zero rise the term is exp(-4) = 0.018,
# so it gives almost no gradient until the robot is already leaving the ground.
# `hop_upward_velocity` -- ungated, dense -- is the DISCOVERY term; this one only
# shapes how high once airborne.
HOP_HEIGHT_GAIN = 0.040
HOP_HEIGHT_STD = 0.020

# Rise above takeoff height that `hop_both_feet_airborne` demands before it pays.
#
# Deliberately tiny. This gate exists to price out a TUCK -- both pads retracted
# with the trunk left where it was, which rises ~0 by construction -- not to
# shape hop height, which is HOP_HEIGHT_GAIN's job. 3 mm clears the ~4 mm of
# trunk dip a 51 ms tuck produces while still admitting the Locked control arm's
# expected ~5 mm hop, so the arm-to-arm comparison survives the gate. That
# lower bound is the binding constraint: a gate at, say, 10 mm would score the
# Locked arm at zero airborne reward and make the controlled comparison
# meaningless, which is precisely why the earlier absolute-height threshold
# could not simply be raised past the 14.2 mm posture headroom.
# 1 mm, not the 3 mm this started at. The rigid (Locked) arm's PHYSICAL ceiling
# is ~3.4 mm of CoM rise at converged timestep, so a 3 mm gate sat AT that
# ceiling and scored the control arm at exactly zero however well it learned --
# the previous three-arm sweep was never a fair comparison. The earlier
# justification ("still admits the Locked arm's expected ~5 mm hop") conflated
# drop-rig REBOUND with an achievable hop from standing.
MIN_RISE = 0.001

# Upward base velocity at which `hop_upward_velocity` saturates (it clamps
# vel_z/max_vel to [0, 1]). A ballistic launch at v rises v**2/(2*g), so the old
# 0.5 m/s saturated at 0.25/19.62 = 12.7 mm of rise -- BELOW the entire 5-33 mm
# discriminating band, meaning the velocity term stopped paying long before the
# height term peaked. 1.0 m/s saturates at 1.0/19.62 = 51 mm, above the 40 mm
# HOP_HEIGHT_GAIN target, so the two terms now peak in the right order.
#
# This makes the term LESS generous early -- at vz = 0.5 m/s it now reads 0.5
# rather than 1.0 -- which is intended: a 0.5 m/s launch is a 13 mm hop, not a
# finished behaviour.
HOP_MAX_LAUNCH_VEL = 1.0

# The pre-merge walking values for `com_height_target`, which the hop task now
# registers itself (develop's 4d34d845 stopped registering the term). Named so
# the tests can pin the band against what the hop task actually owns, rather
# than reading it back out of a walking env that no longer has one.
COM_BAND_FLOOR = 0.11
COM_BAND_CEILING_BASE = 0.14
COM_BAND_WEIGHT = 1.2

# Upper edge of `com_height_target`'s band, for the RIGID robot, in the hop task
# only. See the in-place edit in make_hop_variant for the reasoning.
HOP_COM_HEIGHT_MAX = 0.20

SENSOR_NAME = "feet_ground_contact"

# Everything on the robot EXCEPT the two spring boots, against the terrain.
#
# WHY THIS EXISTS. `hop_both_feet_airborne` asks only "is neither foot
# touching?", which is not the same question as "is the robot airborne" once
# the robot has body geometry that can rest on the floor. Switching the hop arms
# to robot_allcollisions.xml (70 collidable geoms, 22 of them head and neck)
# made that difference reachable, and the policy found it within 3000
# iterations: run 0zvhsz7f falls HEAD FIRST, comes to rest on its head with the
# feet in the air, and pushes off the feet from there. Measured: airborne 88.2%
# of all steps, mean "flight" 765 ms -- a real 765 ms flight peaks at 718 mm,
# against 22 mm of actual CoM rise -- body tilt 52.8 deg, and CoM/root rise
# 0.562 against a genuine hop's 0.980. Spring compression p95 fell to 0.00 mm
# and hop_load_force to 0.0008: the exploit does not use the springs at all.
#
# `fell_over` did not catch it either. Its limit_angle is 1.2217 rad = 70 deg,
# and the resting posture sits at 52.8 deg mean / 62.7 deg p95 -- under the
# threshold, so episodes ran to 980 of 1000 steps.
#
# The rule this encodes: THE ONLY PARTS ALLOWED TO TOUCH THE GROUND ARE THE
# SPRING BOOTS. Anything else is a fall.
BODY_SENSOR_NAME = "body_ground_contact"

# Bodies allowed to touch the ground: the boots, and the ankles they hang from.
#
# BY BODY, NOT BY GEOM, and not by subtree -- both of the obvious spellings fail
# here. Only 2 of the 70 collidable geoms carry names (the two pads), so a geom
# regex cannot address the other 68. And `mode="subtree"` resolves its pattern
# to a SINGLE name -- the subtree root -- so `exclude` has nothing to filter and
# the sensor silently watches the whole robot, boots included: every env then
# terminates on its first step. Measured that way round before this comment
# existed: 128/128 envs terminating at step 1 while standing normally.
#
# THE ANKLES ARE EXCLUDED TOO, and that is a deliberate loosening. The sprung
# arms carry their pad on its own `*_foot_pad` body, so excluding just those two
# would be exact for them -- but the STANDARD arm has no pad body at all; its
# `*_foot_collision` geom sits directly on `ankle_*`. One list has to serve both
# or the control arm terminates on contact with its own feet, so the ankles are
# excluded everywhere and the rule reads "the foot assembly may touch the
# ground; nothing above it may". An ankle scraping the floor therefore does not
# terminate on its own -- `fell_over` at 50 deg is what covers that case.
_GROUND_CONTACT_ALLOWED = (
    "left_foot_pad",
    "right_foot_pad",
    "ankle_left",
    "ankle_right",
)

# Fall threshold for the hop arms, radians. The velocity env ships 1.2217 (70
# deg); the head-rest exploit sat at 52.8 deg mean / 62.7 deg p95 and so never
# terminated. 50 deg is comfortably above the 5.9 deg the honest skip reaches at
# takeoff, and below anything that can become a stable resting posture.
FALL_LIMIT_ANGLE = 0.8727  # 50 deg

# THE REWARD BUDGET. Do not "tidy" these back down -- the 4x is the whole fix
# for the first Phase 4 sweep's null, and the 1/pi below is where it comes from.
#
# All three hop terms gate on `launch = clamp(sin(2*pi*phi), 0)`, whose mean over
# a cycle is exactly 1/pi = 0.318. So a term's per-step ceiling is not its weight,
# it is weight * 0.318 -- and it only reaches that if the shaped factor it
# multiplies is pinned at 1.0 the whole launch half, which no real hop achieves.
#
# AT THE CURRENT WEIGHTS:
#
#   term                      weight   ceiling/step
#   hop_both_feet_airborne     12.0       3.820
#   hop_upward_velocity         8.0       2.546
#   hop_body_height             8.0       ~1.2    (Gaussian, never pinned at 1)
#   ------------------------------------------------
#   naive ceiling, all pinned at 1.0      8.91    <- what the budget test uses
#   with the Gaussian's realistic share   ~7.6
#
# Against standing perfectly still, which now pays `com_height_target`
# 1.2 * 1/pi = 0.382 (it is recovery-gated -- see make_hop_variant step 6) plus
# `upright` 1.0 plus `pose` 2.0 = 3.38/step. Ratio 2.64x.
#
# HOW IT LOOKED BEFORE THE 4x, which is why these numbers are what they are: at
# 3.0 / 2.0 / 2.0 the same three terms ceilinged at 0.955 + 0.637 + ~0.3 = ~1.9,
# against 2.2/step for standing (com_height_target was then ungated and paid its
# full 1.2), at lower `action_rate_l2` cost and with near-zero fall risk. A
# PERFECT HOP LOST TO STANDING STILL before any risk was counted. The first sweep
# -- three arms, 8000 iterations each -- did not fail to find hopping; it
# correctly found that hopping was worse, and all three converged to standing
# (both feet airborne 0.07-0.3% of the time).
#
# `tests/test_hop_cfg.py::test_hop_budget_beats_standing_still` pins the ratio at
# >= 2x, computed from the registered weights rather than from this table.
AIRBORNE_WEIGHT = 12.0
UPWARD_VELOCITY_WEIGHT = 8.0
BODY_HEIGHT_WEIGHT = 8.0

# The other half of the fix: the load phase. All three terms above gate on the
# LAUNCH half (sin > 0), so nothing rewarded the load half at all -- see
# `microduck_mdp.hop_load_force` for why that blocks the whole mechanism.
# Held well below the launch terms on purpose: this is the enabling countermovement,
# not the objective. Its ceiling is 4.0 * 1/pi = 1.27/step.
LOAD_FORCE_WEIGHT = 4.0

# Symmetric-push reward. Sized to sit BETWEEN the load term it complements
# (4.0) and the launch terms it must not outbid (8.0-12.0): the goal is a
# two-footed launch, not a policy that presses evenly and never leaves the
# ground. Ceiling 6.0 * 1/pi = 1.91/step, against hop_load_force's 1.27.
SYMMETRIC_PUSH_WEIGHT = 6.0

# Drift containment. `stillness_at_zero_command` is registered at weight 3.0 by
# the velocity env and pays EXACTLY ZERO on this task: it gates on
# `norm(cmd[:2]) + |cmd[2]| < 0.01`, and the hop command [cos, sin, 0] has norm
# 1.0 by construction, always. Measured consequence on run 59yiy9h6: 132 mm of
# horizontal drift in 14 s (p95 339 mm) at 0.126 m/s. Raising the THRESHOLD past
# 1.0 re-arms the existing, tested term rather than adding a near-duplicate of
# it; "zero command" then reads as "always", which is correct here because this
# task has no velocity command at all -- the twist slots carry the phase.
IN_PLACE_THRESHOLD = 2.0

# THE TERM IS THEN DELIBERATELY DETUNED, and both numbers matter.
#
# The goal is "do not run away", NOT "stand frozen". The robot is not asked to
# hold a position -- nothing commands velocity or heading on this task -- it is
# only asked not to travel fast while hopping. The velocity env sets
# vel_std = 0.07 m/s (tighter than the function's own 0.1 default), which is a
# position hold: it reads 0.04 at the 0.126 m/s this policy drifts at and is
# numerically zero by 0.3. Widening to 0.4 leaves the current gait almost
# untouched and bites only where travel becomes real -- payouts at weight 1.0:
#
#   v = 0.00 m/s -> 1.000     v = 0.60 m/s -> 0.105
#   v = 0.13 m/s -> 0.906     v = 1.00 m/s -> 0.002
#   v = 0.30 m/s -> 0.570
DRIFT_VEL_STD = 0.4

# And the weight comes down from 3.0, because a large reward available for
# STANDING STILL is precisely the shape that produced the Phase 4 null, where
# standing paid 2.2/step against a perfect hop's 1.9 and all three arms learned
# to stand. A hopping robot earns this term anyway -- a vertical hop has near
# zero horizontal velocity -- so it does not need to be large to be obeyed, and
# every point of it is a point a non-hopper also collects.
DRIFT_WEIGHT = 1.0

# Fallback datum for the two force terms, in newtons. `apply_hop_corrections`
# OVERWRITES this per arm from the compiled mass -- 893.0 g sprung / 791.0 g
# standard, matching the scale's 908 g less the 15 g radiator now removed -- so
# this literal is only
# what a term is registered with before that runs, and what it keeps if the
# compile fails. Left at the historical 0.877 kg x 9.81 rather than restated,
# because a wrong-looking fallback is easier to notice than a plausible one.
BODY_WEIGHT_N = 8.60

# Multiple of body weight at which `hop_load_force` saturates: the term reads 0
# at a plain stand and 1.0 once the feet push with this many body weights.
#
# Sized against the SPRING'S TRAVEL, not picked round. At k = 3900 N/m each
# millimetre of pad travel costs 3.9 N, and this file's header records that FULL
# 12 mm travel needs 49.7 N per foot. So a ratio r saturates at
# r * 8.60 / 2 newtons per foot, i.e. r * 8.60 / (2 * 3900) metres of travel:
#
#   r = 2.0  ->  8.6 N/foot  ->  2.2 mm  ->  18% of travel   (REJECTED)
#   r = 6.0  -> 25.8 N/foot  ->  6.6 mm  ->  55% of travel
#
# At r = 2.0 the term had zero gradient across ~82% of the spring's working
# range -- precisely the range that stores energy, and precisely where the three
# arms differ. 6.0 puts the saturation point near half travel, leaving live
# gradient right through the discriminating band without asking for a press the
# knee-limited actuators (52.4 N available) cannot deliver.
#
# Threaded explicitly rather than left on the function default, for the same
# reason `sensor_name` and `stiffness` are: this campaign has been burned by
# defaults that were only correct for one arm.
LOAD_FORCE_MAX_RATIO = 6.0

ENERGY_MONITOR_WEIGHT = 1.0

# Forward-locomotion rewards read the twist command, which the phase command
# overwrites with [cos, sin, 0]. Left in place they would reward running away.
_LOCOMOTION_REWARDS = ("track_linear_velocity", "track_angular_velocity")

# Removed for a DIFFERENT reason than the locomotion rewards above, hence its own
# list: `air_time` does not reward running away, it rewards WALKING IN PLACE, and
# it pays more for that than the hop rewards pay for hopping.
#
# `air_time` is mjlab's `feet_air_time` at weight 5.0. It sums a PER-FOOT
# indicator over the window 0.10 s < air_time < 0.25 s, so it pays continuously
# for alternating single-foot flight. Integrated over one cycle: in-place
# alternating stepping with ~0.25 s swings earns ~3.0/step, whereas a 1.0 s hop
# with 0.2 s of two-foot flight earns ~1.0/step -- against at most ~1.1/step
# available from all three hop terms combined AT THE ORIGINAL 3.0/2.0/2.0 weights
# (see the budget comment beside them; the 4x raises that ceiling to ~8.9/step,
# which does not make `air_time` safe to keep -- it still pays for the wrong
# behaviour, it just no longer wins outright). Marching in place therefore
# strictly outscored hopping, identically on all three arms, and the campaign
# would conclude "compliance does not help" from three runs that never hopped.
#
# It is also permanently latched ON: its gate is ||cmd[:2]|| + |cmd[2]| > 0.01,
# and the phase command [cos(2*pi*phi), sin(2*pi*phi), 0] has magnitude
# identically 1.0. There is no commanded speed left for it to gate on.
#
# NOT swapped for `feet_air_time_capped`: capping fixes double-payment for
# two-foot flight, but the defeat here comes from the continuous SINGLE-foot
# incentive, which capping leaves intact. And the hop task already has its own
# airborne reward, `hop_both_feet_airborne` at AIRBORNE_WEIGHT -- which pays only
# when BOTH feet are off the ground, i.e. for the behaviour we actually want.
_WALKING_GAIT_REWARDS = ("air_time",)

# NOT removed, deliberately: `foot_swing_height` (weight -0.25, relative-squared
# cost, target_height=0.02) is the same CLASS of walking-gait term as `air_time`
# above -- a retained term whose interaction with the hop rewards needs
# checking, not assuming -- but its shape doesn't create the same conflict.
# It's a bowl centred on 20 mm of foot peak height: harmless across the 5-33 mm
# evidence band (<1% of the per-cycle total there), but quadratic above it
# (-0.36 per landing at 33 mm gain, -0.72 at 40 mm, -2.42 at 60 mm). It is
# IDENTICAL across all three arms, so it biases none of them relative to each
# other -- but it is where the reward's new ceiling actually comes from: the
# hop total now peaks around ~65 mm of gain because this is the term that
# finally overtakes the airborne terms (which grow only as sqrt(h)). Its
# logged `Metrics/peak_height_mean` is the arm-comparison observable to watch.

# LANDING SURVIVAL: the spec asks for a landing-survival term. It is met by the
# terms already present rather than by a new one -- the velocity env's `upright`
# reward (weight 1.0) pays for staying vertical through the landing, and the
# `fell_over` termination (bad_orientation, 70 deg) ends an episode that fails.
# Adding a third redundant survival term would double-count the same behaviour
# and make the hop rewards harder to balance against it.


def make_hop_variant(
    cfg: ManagerBasedRlEnvCfg,
    stiffness: float = 3900.0,
) -> ManagerBasedRlEnvCfg:
    """Convert a velocity env cfg into the periodic hop task.

    NOTE: this transform TAKES NO ``h_add``, and that absence is a result rather
    than an oversight. It used to, because both airborne-gated rewards placed an
    ABSOLUTE base-height target that had to be shifted by the boot's height. They
    now measure RISE ABOVE TAKEOFF HEIGHT, which is invariant to how tall the
    robot stands, so the hop rewards no longer care about the boot at all. The
    only remaining h_add-dependent quantity in the hop arms is the
    ``com_height_target`` BAND, and ``make_sprung_variant`` already owns that
    shift. One less pair of values to keep in sync, and one less place to get a
    datum wrong -- which this campaign did twice.

    Args:
        stiffness: N/m spring rate to report through `hop_energy_monitor`. Must
            match the arm's actual spring stiffness -- Task 4 registers a k2500
            arm alongside the k3900 default, and a hardcoded value would report
            that arm's stored spring energy wrong (56% high for k2500 vs k3900).
    """
    # 0. THE ALL-COLLISIONS ROBOT, for every hop arm.
    #
    #    robot_walk.xml has no head or neck collision geometry -- only
    #    trunk_base, leg and leg_2, on their own contype=2/conaffinity=2 layer
    #    -- so the head swept straight through the body. The hop policy uses
    #    the head hard (~35% of body mass) and was exploiting a
    #    self-intersection that cannot happen on hardware.
    #
    #    `make_sprung_variant` REPLACES the robot entity wholesale with one
    #    built from `get_allcollisions_spec`, so for the sprung arms this line
    #    is redundant. It exists for the STANDARD arm, which never goes through
    #    that transform: without it Standard would keep the 5-geom walk model
    #    while every sprung arm ran with 70, and the control comparison would
    #    silently differ in contact as well as in compliance.
    #
    #    Same robot either way -- both XMLs compile to 737.2 g with 15 joints
    #    and 14 actuators, and both are ballasted -- so this changes contact
    #    only.
    robot = cfg.scene.entities["robot"]
    cfg.scene.entities = {
        **cfg.scene.entities,
        "robot": replace(robot, spec_fn=get_allcollisions_spec),
    }

    # 0b. Any non-boot contact with the ground is a FALL, and terminates.
    #
    #     This is the fix for the head-rest exploit described on
    #     BODY_SENSOR_NAME above. Terminating is the right lever rather than
    #     merely gating the airborne reward: the exploit's value came from
    #     SURVIVING in the resting posture (episode length 980 of 1000 while
    #     `fell_over` read 0.114), so removing the reward alone would leave the
    #     robot free to keep lying there for the rest of the episode. Ending the
    #     episode prices the posture correctly -- it is a fall, and the robot
    #     already pays for falls by losing the remaining reward.
    #
    #     `exclude` carries the foot assembly, which is what makes the rule
    #     "only the boots may touch the ground" rather than "nothing may touch
    #     the ground". See _GROUND_CONTACT_ALLOWED for why it is by body.
    #
    #     force_threshold is deliberately absent: `illegal_contact` falls back
    #     to `data.found` when no force history is configured, so ANY contact
    #     counts. A threshold would let the robot rest lightly on its head.
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (
        ContactSensorCfg(
            name=BODY_SENSOR_NAME,
            primary=ContactMatch(
                mode="body",
                pattern=r".*",
                entity="robot",
                exclude=_GROUND_CONTACT_ALLOWED,
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found",),
            reduce="none",
            num_slots=1,
        ),
    )
    cfg.terminations["body_ground_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": BODY_SENSOR_NAME},
        time_out=False,
    )

    # 0c. And tighten the fall angle, because 70 deg is what made the resting
    #     posture reachable in the first place. A hopping robot has no business
    #     past 50 deg, and the exploit parked at 52.8 deg mean precisely because
    #     it was under the old threshold.
    fell = cfg.terminations.get("fell_over")
    if fell is not None and "limit_angle" in fell.params:
        fell.params["limit_angle"] = FALL_LIMIT_ANGLE

    # 1. Cyclic phase command, reusing the class already on develop.
    #
    # `make_microduck_velocity_env_cfg` sets cfg.commands["twist"] to
    # VelocityCommandCommandOnlyCfg, which carries a `rel_turn_in_place_envs`
    # field that GroundPickPhaseCommandCfg doesn't declare (it isn't derived
    # from that class -- the ground_pick reference call site starts from
    # mjlab's base UniformVelocityCommandCfg, which has no such field, so it
    # never hits this). Filter vars(command) down to the fields
    # GroundPickPhaseCommandCfg actually accepts instead of forwarding all of
    # them verbatim.
    #
    # `rel_turn_in_place_envs` is the one field we know is safe to drop: it is
    # only ever read by VelocityCommandCommandOnly._resample_command, and
    # GroundPickPhaseCommand's own _resample_command override is a no-op
    # `pass` -- nothing in the built command ever looks at it. Any OTHER
    # dropped field is unproven and must fail loudly rather than vanish
    # silently into a training run (this campaign has been burned by silent
    # drops before), so we raise if drift introduces one.
    command = cfg.commands["twist"]
    valid_fields = {f.name for f in dataclasses.fields(microduck_mdp.GroundPickPhaseCommandCfg)}
    dropped = set(vars(command)) - valid_fields
    _KNOWN_INERT_DROPS = {"rel_turn_in_place_envs"}
    unexpected_drops = dropped - _KNOWN_INERT_DROPS
    if unexpected_drops:
        raise ValueError(
            f"make_hop_variant: cfg.commands['twist'] ({type(command).__name__}) carries "
            f"field(s) {sorted(unexpected_drops)} that GroundPickPhaseCommandCfg does not "
            "declare and that are not in the known-inert allow-list. Either mirror the "
            "field onto GroundPickPhaseCommandCfg or confirm it is unused (like "
            "rel_turn_in_place_envs) and add it to _KNOWN_INERT_DROPS."
        )
    command_kwargs = {k: v for k, v in vars(command).items() if k in valid_fields}
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
        **{
            **command_kwargs,
            "class_type": microduck_mdp.GroundPickPhaseCommand,
            "period": HOP_PERIOD,
        }
    )

    # 2/3. This is a hop in place: forward tracking would reward running away,
    #      and its command has just been overwritten anyway.
    for name in _LOCOMOTION_REWARDS:
        cfg.rewards.pop(name, None)

    # ...and the walking gait reward, which pays more for marching in place than
    # the hop rewards pay for hopping. See _WALKING_GAIT_REWARDS above.
    for name in _WALKING_GAIT_REWARDS:
        cfg.rewards.pop(name, None)

    # Optional: silence the push perturbation for VISUAL INSPECTION.
    #
    # play=True deliberately SHORTENS the push interval to (0.5, 1.0) s, which
    # is right for a walking policy but not here: HOP_PERIOD is 1.0 s, so the
    # robot gets shoved about once per hop and the natural behaviour is
    # impossible to see. Opt in with HOP_NO_PUSH=1, and it announces itself so
    # it cannot silently remove domain randomisation from a training run.
    if os.environ.get("HOP_NO_PUSH"):
        if cfg.events.pop("push_robot", None) is not None:
            print("  [hop] HOP_NO_PUSH set -> push_robot event REMOVED "
                  "(inspection only; do NOT train like this)")

    # 4. Hop rewards. All three gate internally on sin(2*pi*phase) > 0.
    cfg.rewards["hop_both_feet_airborne"] = RewardTermCfg(
        func=microduck_mdp.hop_both_feet_airborne,
        weight=AIRBORNE_WEIGHT,
        params={
            "sensor_name": SENSOR_NAME,
            "command_name": "twist",
            # Anti-tuck. Only two collision geoms exist on this robot (the two
            # pads), so "both feet airborne" alone says nothing about the 877 g
            # body: retracting both 70 g feet farms this term at a real hop's
            # rate with the trunk motionless. A rise threshold closes that; an
            # absolute-height threshold does NOT, because the robot can stand
            # tall first and tuck from there. See microduck_mdp._HopRiseTracker.
            "min_rise": MIN_RISE,
        },
    )
    cfg.rewards["hop_upward_velocity"] = RewardTermCfg(
        func=microduck_mdp.hop_upward_velocity,
        weight=UPWARD_VELOCITY_WEIGHT,
        params={"command_name": "twist", "max_vel": HOP_MAX_LAUNCH_VEL},
    )
    cfg.rewards["hop_body_height"] = RewardTermCfg(
        func=microduck_mdp.hop_body_height,
        weight=BODY_HEIGHT_WEIGHT,
        params={
            "command_name": "twist",
            # RISE above takeoff, not an absolute height, so no standing-height
            # datum and no h_add enter this. The value is unchanged from the
            # absolute formulation because it was always a GAIN ABOVE STANDING.
            "target_rise": HOP_HEIGHT_GAIN,
            "std": HOP_HEIGHT_STD,
            # Threaded explicitly: this term is gated on BOTH FEET AIRBORNE
            # (otherwise it pays for a ground-level bob), and that gate reads the
            # same contact sensor as hop_both_feet_airborne above.
            "sensor_name": SENSOR_NAME,
        },
    )

    # The load half. The three terms above all gate on sin > 0; this one gates on
    # sin < 0, and is what pays for the countermovement that charges the spring.
    cfg.rewards["hop_load_force"] = RewardTermCfg(
        func=microduck_mdp.hop_load_force,
        weight=LOAD_FORCE_WEIGHT,
        params={
            "sensor_name": SENSOR_NAME,
            "command_name": "twist",
            "body_weight_n": BODY_WEIGHT_N,
            "max_ratio": LOAD_FORCE_MAX_RATIO,
        },
    )

    # Energy instrument. Returns zeros, so the weight only has to be non-zero
    # for RewardManager.compute to call it at all.
    cfg.rewards["hop_energy_monitor"] = RewardTermCfg(
        func=microduck_mdp.hop_energy_monitor,
        weight=ENERGY_MONITOR_WEIGHT,
        params={
            "joint_names": SPRING_JOINTS,
            "stiffness": stiffness,
            "preload": SPRING_PRELOAD,
        },
    )

    # 5. Lift the CoM band's ceiling out of the discriminating range.
    #
    #    `com_height_target` returns +1 flat while in band and -(z - max)**2 once
    #    above it, so crossing the top forfeits the whole +1 as a STEP -- times
    #    its weight of 1.2. With the base band at [0.11, 0.14] rigid (shifted to
    #    [0.14, 0.17] by make_sprung_variant), that step landed at
    #    0.17 - 0.1471 = 23 mm of gain, i.e. right inside the 5-33 mm range this
    #    experiment exists to resolve. It penalised exactly the hops we want.
    #
    #    Safe for STANCE, which is what the band is actually for: the rigid
    #    sag-free kinematic maximum is UNLOADED_RIGID_HEIGHT = 0.12114, already
    #    BELOW the old 0.14 top, so the upper edge was unreachable while standing
    #    and only ever fired airborne. Raising it therefore changes nothing about
    #    standing behaviour on any arm. (Sprung, same argument: a 0.1471 stand vs
    #    a 0.17 top.) `target_height_min` is deliberately untouched -- it still
    #    pays for not collapsing during stance.
    #
    #    Only the RIGID upper edge moves, and only in the hop variant. The
    #    Phase-2 `h_add` translation in make_sprung_variant is untouched (that
    #    "CoM band shift" is the out-of-scope item in the spec): running after
    #    this, it shifts both edges by h_add and yields [0.14, 0.23] for the
    #    sprung arms -- comfortably above the apex the hop reward asks for, which
    #    is UNLOADED_RIGID_HEIGHT + H_ADD + HOP_HEIGHT_GAIN = 0.19114 for a hop
    #    launched from a nominal stance. (The reward itself no longer names that
    #    number -- it shapes RISE -- but the CoM band still has to clear the
    #    absolute height a successful hop reaches, so the arithmetic belongs
    #    here. Posture headroom can add up to ~14 mm on top; the band's 0.23 top
    #    covers that too.)
    #    REGISTERED IF ABSENT. develop's 4d34d845 ("merge velocity2 into
    #    velocity: one walking recipe") stopped registering `com_height_target`
    #    in the walking env -- the function is still in mdp.py, only the term
    #    went. The hop task genuinely needs it (see below and
    #    `make_sprung_variant`'s band shift), so it owns it here rather than
    #    inheriting one. Weight and band are the pre-merge walking values,
    #    which is what every hop number in the campaign was measured against.
    if "com_height_target" not in cfg.rewards:
        cfg.rewards["com_height_target"] = RewardTermCfg(
            func=microduck_mdp.com_height_target,
            weight=COM_BAND_WEIGHT,
            params={
                "target_height_min": COM_BAND_FLOOR,
                "target_height_max": COM_BAND_CEILING_BASE,
            },
        )
    cfg.rewards["com_height_target"].params["target_height_max"] = HOP_COM_HEIGHT_MAX

    # 6. ...and stop paying it at all during the LAUNCH half.
    #
    #    Even with the ceiling lifted, the term's flat +1-in-band (x1.2) was the
    #    single largest reward for standing perfectly still, which is what all
    #    three arms of the first sweep learned. During launch we want the robot
    #    LEAVING the band, so swap the func for the recovery-gated wrapper.
    #
    #    MUTATE IN PLACE, do not rebuild the term. `make_sprung_variant` runs
    #    AFTER this transform and looks the term up by the key
    #    "com_height_target", then shifts `target_height_min`/`target_height_max`
    #    by h_add. Renaming the key or dropping either param silently breaks the
    #    band shift on EVERY sprung arm -- which is why the wrapper takes those
    #    two params through unchanged and only adds `command_name`.
    com = cfg.rewards["com_height_target"]
    com.func = microduck_mdp.com_height_target_recovery_only
    com.params["command_name"] = "twist"

    return cfg


def make_symmetric_variant(cfg):
    """Pay for a TWO-FOOTED launch, so the hop stops being a skip.

    Deliberately a separate transform rather than part of `make_hop_variant`:
    every published hop arm (Locked, K2500, K3344, K3900, Standard) keeps the
    exact reward set its numbers were measured with, and the symmetric arm
    differs from the plain in-place arm in this one term. That is what makes
    "did symmetry buy height?" answerable by comparing two runs.

    See `microduck_mdp.hop_symmetric_push` for the reward's shape and for why it
    is a `min` over the two feet rather than a difference penalty (which two
    unloaded feet satisfy perfectly).
    """
    cfg.rewards["hop_symmetric_push"] = RewardTermCfg(
        func=microduck_mdp.hop_symmetric_push,
        weight=SYMMETRIC_PUSH_WEIGHT,
        params={
            "sensor_name": SENSOR_NAME,
            "command_name": "twist",
            "body_weight_n": BODY_WEIGHT_N,
        },
    )
    return cfg


def make_in_place_variant(cfg):
    """Hop ON THE SPOT, and let the head help.

    Two independent corrections that happen to share a motive -- both were
    holding the in-place hop down, and neither is about the spring.

    1. DRIFT CONTAINMENT -- "do not run away", not "stand frozen". Nothing on
       this task commands velocity or heading; the robot only needs to hop
       without travelling fast while it does.

       `stillness_at_zero_command` is registered at weight 3.0 by the
       velocity env, and on this task it pays EXACTLY ZERO, always. It gates on
       `norm(cmd[:2]) + |cmd[2]| < 0.01`; the hop command is
       [cos(2*pi*phi), sin(2*pi*phi), 0], whose norm is 1.0 by construction. So
       the gate is never open and 3.0 of weight has been inert for every hop run
       in the campaign. Measured on run 59yiy9h6: 132 mm of horizontal drift in
       14 s, p95 339 mm, at 0.126 m/s.

       Fixed by raising the THRESHOLD past the command's magnitude rather than
       writing a near-duplicate term. "Zero command" then means "always", which
       is the correct reading here: this task has no velocity command at all --
       the twist slots carry the phase, and there is no other velocity for the
       robot to be tracking instead.

       The term is then DETUNED -- vel_std 0.07 -> 0.4, weight 3.0 -> 1.0. The
       stock values are a position hold, and a large standing-still reward is
       the shape that produced the Phase 4 null. See DRIFT_VEL_STD and
       DRIFT_WEIGHT for both numbers and why.

    2. THE HEAD IS FREED. The head subtree is 279.9 g of a 791.0 g robot -- 35%,
       proportionally a far bigger swing mass than a human arm -- and it sits at
       the top of the body where it does the most good. Two rewards were
       actively preventing it from being used:

         head_pose_tracking   +3.0   pays for tracking the COMMANDED head pose
         neck_action_rate_l2  -0.1   penalises moving the head at all

       Both are dropped. `head_command` STAYS IN THE OBSERVATION -- the 61-D
       layout is a deployment contract (docs/sim2real/hop_deployment.md) and
       must not move -- the policy simply stops being paid to obey it.

       THE COST, and it is a real one: those two terms plus the neck-offset
       randomisation are why head pose is independently commandable at runtime
       (see CLAUDE.md). Freeing the head spends that on this task. It is the
       right trade while the goal is a maximum-height in-place hop, and it is
       the first thing to revisit if head control is wanted back.
    """
    #    REGISTERED IF ABSENT, and this one must not fail quietly. develop's
    #    4d34d845 also dropped `stillness_at_zero_command` from the walking
    #    recipe. The old `cfg.rewards.get(...) / if not None` spelling then
    #    became a SILENT NO-OP: the drift containment simply vanished, with no
    #    error and no log line, and the only symptom would have been a robot
    #    that travels while it hops -- exactly the bug this transform exists to
    #    fix, reintroduced by a merge. Register it instead.
    if "stillness_at_zero_command" not in cfg.rewards:
        cfg.rewards["stillness_at_zero_command"] = RewardTermCfg(
            func=microduck_mdp.stillness_at_zero_command,
            weight=DRIFT_WEIGHT,
            params={},
        )
    still = cfg.rewards["stillness_at_zero_command"]
    still.params["command_threshold"] = IN_PLACE_THRESHOLD
    still.params["vel_std"] = DRIFT_VEL_STD
    still.weight = DRIFT_WEIGHT

    # Pop rather than zero the weight: RewardManager.compute short-circuits on
    # weight == 0.0 without calling the term, so a zeroed term is a dead entry
    # that still shows up in every log and reward-bar panel.
    cfg.rewards.pop("head_pose_tracking", None)
    cfg.rewards.pop("neck_action_rate_l2", None)

    return cfg


from copy import deepcopy
from dataclasses import replace

# H_ADD dropped from this import: make_hop_variant no longer takes one. The hop
# rewards measure rise above takeoff, so the boot's height is irrelevant to them;
# its one remaining consumer is the CoM band, owned by make_sprung_variant.
from mjlab_microduck.robot.sprung_foot import K_MEASURED, PAD_MASS, TRAVEL
from mjlab_microduck.tasks.run import MicroduckRunRlCfg

# (label, stiffness N/m, travel m, pad mass kg).
#
# Three arms, not six. The drop-rig probe already narrowed stiffness: at a
# 100 mm drop k2500 rebounded 35.3 mm, k3900 32.8 mm, k5500 28.3 mm, and k1500
# only 20.6 mm because it bottoms out and slams. So the optimum is the softest
# spring that does NOT bottom, and 2500-3900 brackets it.
#
# Mass is held at the measured 70 g because Stage 1's locked arms already
# measured the mass penalty separately (-17.7% at 30 g, -61.9% at 90 g vs the
# rigid running baseline).
HOP_ARMS = (
    ("locked", K_MEASURED, 0.0, PAD_MASS),
    ("k2500", 2500.0, TRAVEL, PAD_MASS),
    # The MEASURED prototype stiffness (gripper bench, 2026-09-03). This is the
    # arm to train and compare against Locked; k2500/k3900 were a stiffness
    # BRACKET and they served that purpose -- they gave 23 mm and 27 mm of rise,
    # so 3344 should land at ~25-26 mm.
    ("k3344", K_MEASURED, TRAVEL, PAD_MASS),
    ("k3900", 3900.0, TRAVEL, PAD_MASS),
)

HOP_ARM_SUFFIX = {"locked": "Locked", "k2500": "K2500",
                  "k3344": "K3344", "k3900": "K3900"}



# --- corrections measured on the real bench, 2026-09-02 ---------------------
#
# The first three-arm sweep was run with all four of these wrong. Applied to the
# HOP PATH ONLY -- the shared actuator cfg object is deep-copied -- so the
# velocity/run/backlash tasks keep their previous behaviour and remain
# comparable with their own history.
HOP_TIMESTEP = 0.002       # was 0.005
HOP_DECIMATION = 10        # was 4; 0.002 x 10 keeps the 50 Hz control period,
                           # which a gain sweep found to be the optimum
HOP_KP_FW = 400.0          # was 200. Measured on the bench: achieved amplitude
                           # peaks at kp 400-800 and FALLS at 1600 while current
                           # rises 71%, so the benefit is exhausted by ~400 and
                           # the predicted 3.1x at kp 2000 is not real.
_MEASURED_PARAMS = "xl330_m6_measured_friction.json"


def apply_hop_corrections(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """Apply the bench-measured corrections to a composed hop cfg.

    Call AFTER make_sprung_variant, since that swaps the robot entity.

    Timestep: dt=0.005 manufactures energy. Measured drop-rig energy retention
    was 0.872 at dt=0.005 against a 0.468 LOSSLESS ceiling -- the integrator was
    creating energy via the 37.6 Hz boot mode at ~5 steps/period. Two thirds to
    three quarters of the sprung advantage was numerical.

    Delay: delay_*_lag are counted in SIM STEPS, so halving the timestep would
    silently halve the physical latency. Rescaled to hold 15-30 ms.

    Friction: the gearbox supports 75% of external load (see the params file).
    """
    import dataclasses
    from copy import deepcopy

    cfg.sim.mujoco.timestep = HOP_TIMESTEP
    cfg.decimation = HOP_DECIMATION

    # CONTACT SOLVER HARDENING, and it is load-bearing now rather than
    # defensive. Switching the hop arms to robot_allcollisions.xml took the
    # robot from 5 collidable geoms to 70, 22 of them on the head and neck.
    # nconmax = 35 is the FLAT-terrain default, sized for the 5-geom model; a
    # hop arm falls 48-92% of the time and lands with trunk, folded legs and
    # head all in close ground/self contact, which is precisely the state the
    # sitstand env documents as overflowing the solver at 35/10 -> NaN ->
    # nan_state terminations that punish the very behaviour being learned.
    #
    # Worse than a crash, an overflow here would be SILENT AND SELF-DEFEATING:
    # dropped contacts re-admit the head-through-body intersection that
    # switching models exists to prevent, and they drop it exactly during the
    # collapses where the head is most likely to be inside the body.
    #
    # Values follow the sitstand env, which already runs this collision set.
    cfg.sim.nconmax = 200
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50

    scale = 0.005 / HOP_TIMESTEP
    params = str(Path(__file__).resolve().parents[1] / "robot" / _MEASURED_PARAMS)

    # Both force terms normalise ground reaction force by body weight, and the
    # arms no longer share a mass: 893.0 g with spring boots (2 x 51 g delta),
    # 791.0 g standard. Compile the actual robot and use its real weight, so no
    # arm gets a normalisation advantage. Hardcoding one value made the lighter
    # Standard arm's load term read 13.6% high.
    #
    # BOTH TERMS, NOT JUST hop_load_force. `hop_symmetric_push` is designed to
    # saturate on the same push hop_load_force does -- each foot at body weight
    # against the pair at max_ratio = 2.0 -- and that only holds if the two read
    # the SAME body weight. Patching one and leaving the other on the 8.60 N
    # literal put them 3.5% apart and quietly broke the property the symmetric
    # term was built around.
    robot_ent = cfg.scene.entities["robot"]
    try:
        model = robot_ent.spec_fn().compile()
        w = float(sum(model.body_mass)) * 9.81
        patched = []
        for name in ("hop_load_force", "hop_symmetric_push"):
            term = cfg.rewards.get(name)
            if term is not None and "body_weight_n" in term.params:
                term.params["body_weight_n"] = w
                patched.append(name)
        if patched:
            print(f"  [hop] body weight {w:.3f} N "
                  f"({sum(model.body_mass)*1000:.1f} g) -> {', '.join(patched)}")
    except Exception as exc:  # noqa: BLE001 - never block registration on this
        print(f"  [hop] WARNING could not compute body weight ({exc}); "
              f"the force terms keep their defaults")

    robot = cfg.scene.entities["robot"]
    new_acts = []
    for act in robot.articulation.actuators:
        a = deepcopy(act)
        for field, value in (("kp_fw", HOP_KP_FW),
                             ("json_path", params),
                             ("motor_name", None),
                             ("model", None)):
            if hasattr(a, field):
                object.__setattr__(a, field, value) if dataclasses.is_dataclass(a) and \
                    getattr(type(a), "__dataclass_params__", None) and \
                    type(a).__dataclass_params__.frozen else setattr(a, field, value)
        for field in ("delay_min_lag", "delay_max_lag"):
            if hasattr(a, field) and getattr(a, field) is not None:
                setattr(a, field, int(round(getattr(a, field) * scale)))
        new_acts.append(a)
    robot.articulation = dataclasses.replace(
        robot.articulation, actuators=tuple(new_acts)
    )
    return cfg


def hop_rl_cfg(label: str):
    """Per-arm RL cfg: identical learner, distinct logging identity.

    ``replace`` is shallow, so the nested cfgs are deep-copied -- otherwise all
    three arms would share one actor object AND share it with the Run baseline,
    so a later change to any arm would silently alter the others.
    """
    return replace(
        MicroduckRunRlCfg,
        actor=deepcopy(MicroduckRunRlCfg.actor),
        critic=deepcopy(MicroduckRunRlCfg.critic),
        algorithm=deepcopy(MicroduckRunRlCfg.algorithm),
        experiment_name=f"hop_{label}",
        run_name=f"hop_{label}",
    )
