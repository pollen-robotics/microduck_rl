"""Sprung-foot robot model — an idealised 1-DoF compliant foot accessory.

Built PROGRAMMATICALLY from the canonical ``robot_walk.xml`` rather than as a
forked XML. The abandoned ``test_spring`` branch forked the XML, and its 50-line
delta became unusable once ``robot_walk.xml`` moved by 310 insertions. Adding two
bodies to the live spec tracks every upstream change to the base model for free.

The mechanism modelled here is deliberately idealised: one prismatic spring per
foot. That is not a shortcut — it is the design target. A rigid 1-DoF
translating mechanism (a prismatic slide, or a Sarrus linkage) maps exactly onto
a MuJoCo ``slide`` joint, so the kinematics carry no sim-to-real gap. A
Kangoo-style leaf flexure would need a discretised multi-body chain or
deformables, and was rejected on that basis. See the Phase 2 spec.
"""

from __future__ import annotations

from typing import Callable

import math

import mujoco
import numpy as np

from mjlab.entity import EntityCfg, EntityArticulationInfoCfg

from mjlab_microduck.robot.microduck_constants import (
    FULL_COLLISION,
    HOME_FRAME,
    actuators,
    get_allcollisions_spec,
)

# Local +y of the ankle bodies maps to world [0, 0.087, 0.996] — almost straight
# up. So a slide along +y means positive q = compression (pad moves toward the
# body). Local +z is nearly HORIZONTAL; using it would slide the foot sideways.
SPRING_AXIS = (0.0, 1.0, 0.0)

# Distance from the ankle body origin down to the existing sole's contact plane,
# measured at the home pose. The pad is placed h_add BELOW that, which is what
# makes the sprung robot taller than the rigid one.
#
# Tuned (not the naive mesh measurement of 0.025) so that the settled
# rigid-vs-sprung trunk-height delta lands on H_ADD: the rigid sole is a mesh
# and the pad is a box, and the two settle to slightly different contact
# penetration depths under gravity, so the naive value overshot the delta by
# ~3-4 mm. See task-1-report.md fix-round-1 notes.
#
# Retuned for the measured H_ADD=0.030 (was 0.0215, tuned for the old
# H_ADD=0.025): the mesh-vs-box penetration mismatch this constant corrects
# for is unaffected by H_ADD, so it re-manifested as the same ~4 mm overshoot
# on the LOCKED (zero-compliance) arm's settled delta and was re-tuned down
# by that amount. See FIX 5's settling measurement in the prototype-update
# report.
#
# RETUNED AGAIN 2026-09-04, +3.53 mm, when the pad footprint was corrected to
# the measured 25 x 40 mm. THIS CONSTANT IS COUPLED TO THE FOOTPRINT and that is
# not obvious: it exists to cancel the contact-penetration difference between
# the mesh sole and the box pad, and penetration depends on contact PRESSURE.
# Shrinking the sole from 40x28 to 25x40 mm cut its area 11% and pushed the
# settled Locked-minus-Standard delta to 26.47 mm against the 30.00 mm H_ADD
# target. Anyone touching `_PAD_HALF_EXTENTS` must re-measure that delta and
# re-tune this; `test_sprung_foot.py` pins it so the drift cannot pass silently.
ANKLE_TO_SOLE = 0.02097

H_ADD = 0.030      # measured on the Sarrus prototype (was an assumed 0.025)
# MEASURED stiffness of the prototype boot, 2026-09-03, on the RobStride gripper
# bench: 3344 N/m (two runs 3.6% apart, r^2 > 0.998). Method: jaw position from
# the encoder against a scale in series with the boot -- motor torque is not in
# the measurement, so it is immune to the rack friction and cogging that spoil
# a torque-derived figure (that route gave 5026 N/m, r^2 0.91; discarded).
#
# The earlier hand figure of 3900 N/m (3 mm ~ 1500 g, 8 mm ~ 3500 g) is 17% HIGH.
# The registered hop arms are k2500 and k3900, which bracket 3344, and the sweep
# gave 23 mm and 27 mm of rise respectively -- so ~25-26 mm is the honest
# expectation for the real spring. Not worth re-running for; do it when the load
# cells land a real damping number.
#
# Damping was NOT measured, only bounded: c <= 12.5 N.s/m, i.e. zeta <= 0.41 vs
# the pad / <= 0.115 vs the whole robot. DAMPING_RATIO = 0.3 below gives
# c = 9.18 N.s/m (zeta_eff 0.12 vs the robot), which sits INSIDE that bound -- so
# the assumption is not contradicted and is, if anything, slightly pessimistic.
# Loss was also rate-independent (+5.5% over a 20x frequency span), so the
# viscous term is small. The boot's own Coulomb friction is negligible: it
# returns fully to free length, where 6.29 N in its load path would leave a
# 1.88 mm stiction dead band.
#
# See rebot-lerobot/bench/RESULTS.md and
# docs/sim2real/spring_boot_identification_spec.md.
K_MEASURED = 3344.0

# DELTA mass of fitting a spring boot, per foot: the 69 g spring boot REPLACES
# the 18 g standard pad foot, so 69 - 18 = 51 g. The common motor-to-boot
# interface (16.5 g) is present in BOTH configurations and cancels.
#
# This is a delta, not the boot's mass, because the pad body is ADDED on top of
# the stock model (737.2 g -> compiles with both pads on top) and the stock model
# already matches the real bare robot, i.e. it already contains the standard
# pads. Using the boot's full 70 g over-counted by 19 g per foot.
PAD_MASS = 0.051   # measured 2026-09-03: 69 g spring boot - 18 g standard pad
TRAVEL = 0.012     # measured (was an assumed 0.015)
# Damping is specified as a RATIO, not an absolute rate, and derived per arm as
# c = 2*zeta*sqrt(k*pad_mass).
#
# Why: an absolute c=0.5 N.s/m ("a good steel spring, low hysteresis") is the
# right figure for the SPRING but leaves the PAD-ON-SPRING subsystem essentially
# undamped — zeta came out at 0.013-0.023, the pad rang at 33-57 Hz against a
# 50 Hz controller, and it retained 65-87% of its amplitude across a 51 ms
# stance, so it never settled between steps. The 30 g pad rang ABOVE the control
# rate entirely. In the first Stage 1 sweep this made sprung speed *improve*
# monotonically with pad mass (lighter pad = faster ringing = worse), the
# opposite of the locked arms' trend — resonance masquerading as a mass effect.
#
# A ratio also keeps resonance CONSTANT across a mass sweep instead of letting it
# confound the axis, and is physically defensible: a larger mechanism carries
# proportionally more joint friction. Reaching zeta=0.3 needs c = 6.5-11 N.s/m
# across the 30-90 g range, i.e. 13-22x the old absolute value.
#
# This is provisional. The real number is measurable on the prototype as
# loading-vs-unloading hysteresis; that measurement should replace this estimate.
DAMPING_RATIO = 0.3

DAMPING = None     # absolute N.s/m; None derives it from DAMPING_RATIO


def damping_for(stiffness: float, pad_mass: float, ratio: float = DAMPING_RATIO) -> float:
    """Critical-damping-scaled rate: c = 2*zeta*sqrt(k*m) for the pad on its spring."""
    return 2.0 * ratio * math.sqrt(stiffness * pad_mass)

# Intentional spring preload in the Sarrus mechanism, as a DISPLACEMENT.
# Measured: 2.9 N offset at k = 3920 N/m -> 2.9/3920 = 0.74 mm of precompression.
# Parameterised as displacement rather than force because the linkage geometry
# fixes the precompression at assembly: a stiffer spring in the same boot keeps
# the 0.74 mm and produces proportionally MORE preload force. Consequence, which
# is physically faithful rather than a modelling artifact: preload force varies
# across the sweep (1.1 N at k=1500, 4.1 N at k=5500).
# Preload holds the pad firmly at full extension during flight instead of
# letting it float within its travel and chatter against the hard stop.
SPRING_PRELOAD = 0.00074   # m of precompression at rest

# These exist to OVERRIDE the `microduck` childclass joint defaults
# (frictionloss=0.1, armature=0.005 in robot_walk.xml), which the spring joint
# would otherwise inherit silently — the joint is added inside that childclass
# scope. Zero is not a physical claim about a real mechanism: it makes the model
# match the spec's *idealised* spring, whose only dissipation is the
# viscous DAMPING_RATIO term.
# Mechanism stiction and mechanism inertia are hardware-phase concerns the spec
# explicitly defers.
SPRING_FRICTIONLOSS = 0.0
SPRING_ARMATURE = 0.0

# Joint-limit solver settings for the mechanical end-stop.
#
# MuJoCo's default limit ([0.02, 1] with dt=0.002, i.e. a 10*dt time constant) is
# soft enough that a hard landing squishes straight through it: a drop-rig probe
# drove k=1500 to 149% of its 12 mm travel from a 100 mm drop. A stop that
# stores-and-returns instead of dissipating systematically overstates rebound,
# which matters because bottoming is common in hop and jump regimes.
#
# 0.004 is the solver's stability floor (2*timestep). Tuned empirically, not
# guessed: it cuts worst-case penetration from 149% to 109% of travel, and the
# stiff arms stay well inside range. MuJoCo cannot make this stop truly rigid at
# a 2 ms timestep, so a residual ~9% overshoot remains under the worst impacts.
# Read a high `spring_bottomed_fraction` as "this arm bottoms out" rather than
# trusting its rebound magnitude to the millimetre.
SPRING_SOLREF_LIMIT = (0.004, 1.0)          # (timeconst, dampratio)
SPRING_SOLIMP_LIMIT = (0.9, 0.99, 0.001, 0.5, 2.0)   # dmax 0.95 -> 0.99

SPRING_JOINTS = ("passive_left_foot_spring", "passive_right_foot_spring")

# The `<default class="collision">` block in robot_walk.xml (group=3). The pad's
# contact geom MUST inherit it: `foot_height_scan` rays are restricted to
# `include_geom_groups=(0,)` (terrain only), so a group-0 pad geom would be hit
# by the opposite foot's height ray and reported as ground, corrupting
# `foot_clearance` and `foot_swing_height`.
_COLLISION_CLASS = "collision"

# Contact pad half-extents (m), as (fore-aft, thickness, lateral).
#
# MEASURED on the Sarrus prototype, 2026-09-04: the contact sole is 25 mm
# fore-aft x 40 mm lateral. The linkage eats most of the original foot's
# footprint, so the boot's sole is much SMALLER than the mesh sole it replaces.
#
# THESE WERE TRANSPOSED UNTIL 2026-09-04, and it flattered every landing in the
# campaign. The old (0.020, 0.004, 0.014) gave 40 mm fore-aft x 28 mm lateral --
# 60% MORE fore-aft base than the robot has, and 30% less lateral. Fore-aft is
# the axis that matters: laterally a biped is braced by having two feet spaced
# apart, but nothing resists a fore-aft topple except sole length and the ankle,
# and the CoM sits ~150 mm up. So the simulated robot was landing on a pitch
# footprint it does not own, and `fell_over = 0.875` was measured on the
# forgiving geometry -- the real figure is worse.
#
# Axis mapping, verified against the compiled model at the home pose rather than
# assumed (local y is world-up here, so the middle number is half the thickness):
#
#   local x -> world [-1.000, 0.000, 0.000]  = fore-aft
#   local y -> world [ 0.000, 0.087, 0.996]  = vertical
#   local z -> world [ 0.000, 0.996,-0.087]  = lateral
_PAD_HALF_EXTENTS = (0.0125, 0.004, 0.020)


def make_sprung_foot_spec_fn(
    stiffness: float,
    travel: float = TRAVEL,
    damping: float | None = DAMPING,
    h_add: float = H_ADD,
    pad_mass: float = PAD_MASS,
    preload: float = SPRING_PRELOAD,
    damping_ratio: float = DAMPING_RATIO,
) -> Callable[[], mujoco.MjSpec]:
    """Build a zero-argument ``spec_fn`` for a sprung-foot MicroDuck.

    ``EntityCfg.spec_fn`` must take no arguments, so the spring parameters are
    captured in a closure. ``travel=0.0`` yields the LOCKED control variant:
    identical geometry and mass, no compliance.

    Args:
        stiffness: spring rate in N/m, applied to both feet.
        travel: stroke in m. 0.0 locks the spring.
        damping: absolute N.s/m on the spring DoF. ``None`` (the default)
            derives it from ``damping_ratio`` as ``2*zeta*sqrt(k*pad_mass)``,
            which holds the pad's damping ratio constant as pad_mass varies.
            Pass an explicit value once the prototype's real hysteresis is
            measured.
        damping_ratio: target zeta for the pad-on-spring subsystem, used only
            when ``damping`` is None.
        h_add: metres of height the mechanism adds below the existing sole.
        pad_mass: mass per pad in kg.
        preload: metres of precompression built into the mechanism at
            assembly. Applied as ``springref = -preload`` (see below).
    """

    resolved_damping = (
        damping if damping is not None else damping_for(stiffness, pad_mass, damping_ratio)
    )

    def _spec_fn() -> mujoco.MjSpec:
        # THE ALL-COLLISIONS BASE, not robot_walk.xml. The hop policy uses the
        # head hard -- it is ~35% of body mass and swinging it is worth real
        # height -- and robot_walk.xml has NO head or neck collision geometry
        # at all: only trunk_base, leg and leg_2, on their own
        # contype=2/conaffinity=2 layer. So the head swept straight through the
        # body and the policy was free to exploit a self-intersection that
        # cannot happen on hardware. apirrone's robot_allcollisions.xml
        # (develop 08680d3d) carries 70 collidable geoms, 16 of them on
        # `jaw_soft` and 6 on the neck.
        #
        # Same robot otherwise: both XMLs compile to 737.2 g with 15 joints and
        # 14 actuators, so this changes contact only, not mass or kinematics.
        spec = get_allcollisions_spec()
        for side in ("left", "right"):
            ankle = spec.body(f"ankle_{side}")

            # Retire the rigid sole: rename it and switch off its contact, so
            # the name `{side}_foot_collision` is free for the pad below. Left
            # in place it would keep answering the feet_ground_contact sensor
            # while floating h_add above the ground.
            old_geom = spec.geom(f"{side}_foot_collision")
            old_geom.name = f"{side}_sole_disabled"
            old_geom.contype = 0
            old_geom.conaffinity = 0
            spec.site(f"{side}_foot").name = f"{side}_foot_old"
            # -y is downward in world at the home pose, so a negative y offset
            # puts the pad below the ankle.
            pad = ankle.add_body(
                name=f"{side}_foot_pad", pos=[0.0, -(ANKLE_TO_SOLE + h_add), 0.0]
            )
            # travel == 0.0 is the LOCKED control arm: no joint at all, so the
            # pad is a rigid child of the ankle (identical mass and height,
            # zero DoF). A slide joint with range [0, 0] compiles fine but is
            # NOT locked -- MuJoCo leaves `limited` at AUTO in that case
            # (range == the joint-type default), so the joint is actually
            # unconstrained and held only by the spring. That silently turns
            # the control arm into an infinite-travel spring, which defeats
            # its purpose of isolating "extra height/mass" from "compliance".
            if travel > 0.0:
                joint = pad.add_joint(
                    name=f"passive_{side}_foot_spring",
                    type=mujoco.mjtJoint.mjJNT_SLIDE,
                )
                joint.axis = list(SPRING_AXIS)
                joint.range = [0.0, travel]
                joint.limited = 1
                # These MUST be 3-arrays; MjsJoint rejects a scalar. Only
                # element 0 is used by the compiler.
                joint.stiffness = np.array([stiffness, 0.0, 0.0])
                joint.damping = np.array([resolved_damping, 0.0, 0.0])
                # Stiffen the mechanical end-stop; see the constants above.
                joint.solref_limit = np.array(SPRING_SOLREF_LIMIT)
                joint.solimp_limit = np.array(SPRING_SOLIMP_LIMIT)
                # MuJoCo's spring force is -stiffness * (qpos - springref).
                # Our convention is q=0 extended, q>0 compressed, so a NEGATIVE
                # springref puts a compression-resisting force at q=0: the
                # spring is pressed against its own extension stop, i.e. the
                # preload. Unlike stiffness/damping, springref is a SCALAR on
                # MjsJoint (a 3-array raises TypeError).
                joint.springref = -preload
                # Set explicitly to override the `microduck` childclass joint
                # defaults (0.1 / 0.005) this joint would otherwise inherit.
                # See the constants above. Unlike stiffness/damping these two
                # are SCALARS on MjsJoint (a 3-array raises TypeError).
                joint.frictionloss = SPRING_FRICTIONLOSS
                joint.armature = SPRING_ARMATURE
            # Re-use the ORIGINAL names so the contact sensor, the terrain
            # height-scan frames, foot_clearance and foot_slip all keep working
            # with no config change.
            pad.add_geom(
                spec.find_default(_COLLISION_CLASS),
                name=f"{side}_foot_collision",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=list(_PAD_HALF_EXTENTS),
                pos=[0.0, 0.0, 0.0],
                mass=pad_mass,
            )
            pad.add_site(name=f"{side}_foot", pos=[0.0, 0.0, 0.0])
        return spec

    return _spec_fn


def make_sprung_foot_robot_cfg(
    stiffness: float,
    travel: float = TRAVEL,
    damping: float | None = DAMPING,
    h_add: float = H_ADD,
    pad_mass: float = PAD_MASS,
    preload: float = SPRING_PRELOAD,
) -> EntityCfg:
    """EntityCfg for a sprung-foot MicroDuck, spawned h_add higher.

    The spawn must rise by exactly ``h_add`` or the taller foot starts inside
    the floor.
    """
    init_state = EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, h_add),
        joint_pos=dict(HOME_FRAME.joint_pos),
        joint_vel={".*": 0.0},
    )
    return EntityCfg(
        spec_fn=make_sprung_foot_spec_fn(
            stiffness, travel, damping, h_add, pad_mass, preload
        ),
        init_state=init_state,
        collisions=(FULL_COLLISION,),
        articulation=EntityArticulationInfoCfg(
            actuators=(actuators,),
            soft_joint_pos_limit_factor=0.9,
        ),
    )
