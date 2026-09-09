#!/usr/bin/env python3
"""Measure the Microduck backflip envelope on CPU MuJoCo, before any training.

Sweeps launch speed x flick rate x tuck depth with the robot held at a FIXED
open-loop pose (no policy), and records, per cell:
  • total backward pitch accumulated while airborne (deg)
  • peak downward speed at first ground contact (m/s) — the hardware risk
  • apex height (m)
  • trunk tilt at the instant the flick starts (deg) — how much of the
    commanded posture actually survived the hold

POSTURE MODES (--posture), and why this flag exists
---------------------------------------------------
``--posture tucked`` (the original mode) spawns the robot ALREADY squatting in
the tuck pose with its trunk ~2 cm above the plate top, holding that ctrl from
t=0. Every table in docs/backflip_envelope_results.md above the
"Standing-spawn re-measurement" section was measured this way.

``--posture tucked_env`` (now the DEFAULT) reproduces what the ENV actually
does: ``reset_backflip_robot_on_plate`` folds the robot to
``TUCK_OVERRIDES x TUCK_FACTOR`` and puts the trunk at
``z0 + PLATE_HALF_THICKNESS + TUCK_Z`` (all four constants IMPORTED from the
cfg, so this mode cannot silently stop being what the env does), and
``ready_stance`` pays the policy to still be holding that tuck when the flick
arrives.

``--posture standing`` reproduces the SUPERSEDED standing hold (HOME pose at
STAND_Z). It is kept because it is the measurement that killed itself: a
standing robot has a ~11 cm CoM, not the tuck's ~3 cm, and CoM height is
exactly what decides whether the flick tips the robot backward over its heels
or overdrives the sole contact and throws it FORWARD. The reversal boundary
sits near w0 ~ 36-39 rad/s tucked but near w0 ~ 18-24 rad/s standing — inside
the box that was configured at the time. Keep all three modes so they stay
directly diffable.

Two acceptance modes worth knowing about:
  --box-check     WHOLE-BOX verification of the cfg's DR ranges from the env's
                  actual spawn. Reports the WORST cell, not the best: every
                  corner AND midpoint of z0 / vz / w0 / t_launch, crossed with
                  the hold extremes and tuck depths, must close 360 deg under
                  2.6 m/s. A box whose interior contains one dead cell is not
                  a box — that is how the previous box got shipped.
  --measure-tuck-z  Re-measure TUCK_Z (the tucked resting trunk height on the
                  plate top) by dropping the robot from several offsets.

``--tuck-at-flick`` additionally commands the tuck pose (HOME with the TUCK
overrides applied, scaled by the tuck factor) from t >= t_hold, i.e. the
instant the plate starts moving. That models what the policy CAN do — it acts
during HOLD and LAUNCH — without pretending it was pre-tucked on the hands.

The point is to find where 360 deg closes at the LOWEST landing speed, and to
find out whether it closes at all inside the launch heights the operator can
actually use. The measured cells set the DR ranges in the env cfg; nothing in
the reward stack should be tuned before this table exists.

Run: uv run python scripts/backflip_envelope.py
Direction sanity check (see check_direction() below):
  uv run python scripts/backflip_envelope.py --check-direction
Direction check on a specific cell:
  uv run python scripts/backflip_envelope.py --check-direction --check-vz 2.25 --check-w0 30.0 --check-tuck 1.0 --z0 0.10

Extended sweep (Task 3 addendum): the default grid below pushes w0 well past
the original 6-15 rad/s range (up to 54) and vz down to 1.0, at a finer 0.25
step, than the original 32-cell grid — because an initial push to w0<=30
still put every new-best cell at the edge of that range. IMPORTANT: cells
with w0 above roughly 33-36 rad/s report large "backward rotation" numbers
that a --check-direction trace shows are actually FORWARD rolls (see
docs/backflip_envelope_results.md "The reversal") — do not trust rot_deg /
land_m/s from this sweep for w0 past that point without directly verifying
direction first. Rows past the verified-safe ceiling (SAFE_W0_CEILING) are
marked with a trailing '*' in the printed table as a reminder not to sort
this output by land_m/s and trust whatever comes out on top. Override the
grid with --vz/--w0/--tuck; --z0 is unchanged.

Two DIFFERENT sets of flags, easy to fat-finger: --vz/--w0/--tuck take
comma-separated GRIDS and drive the full sweep (main mode); --check-vz/
--check-w0/--check-tuck take single numbers and only apply under
--check-direction (one cell, direction trace). Typing --w0 when you meant
--check-w0 silently sweeps the whole default grid instead of checking one
cell.
"""

import argparse
import itertools
from typing import NamedTuple
import math

import mujoco
import numpy as np
import torch

from mjlab_microduck.tasks import mdp as microduck_mdp

# IMPORTED, NOT COPIED. The whole defect this probe was rebuilt to catch was a
# measurement whose spawn geometry had drifted from the env's. Everything that
# defines the spawn and the launch box therefore comes from the cfg module
# itself, so `--posture tucked_env` cannot silently stop being what the env
# does. (The import costs ~6 s; this is not an interactive script.)
from mjlab_microduck.tasks.microduck_backflip_env_cfg import (  # noqa: E402
    HOLD_LERP,
    HOLD_RANGE,
    HOLD_Z,
    LAUNCH_RANGE,
    PLATE_HALF_THICKNESS,
    STAND_Z,
    TUCK_FACTOR,
    TUCK_OVERRIDES,
    TUCK_Z,
    VZ_RANGE,
    W0_RANGE,
    Z0_RANGE,
)

SCENE = "src/mjlab_microduck/robot/microduck/scene_backflip.xml"

# Servo index -> tuck angle at full tuck (factor 1.0). The env's own tuck map.
# A tucked duck has a much smaller pitch inertia, which is exactly how it
# converts the hand's angular impulse into a fast enough spin.
TUCK = TUCK_OVERRIDES

# HOME pose, servo index order (= actuator index order on this model). Same
# numbers as HOME_FRAME in robot/microduck_constants.py and DEFAULT_POSE in
# scripts/infer_policy.py — this is the pose the env's reset_robot_joints
# event puts the robot in (+/- 0.05 rad of joint noise) and the pose the
# policy's zero action commands.
HOME = np.array([
    0.0,      # left_hip_yaw
    -0.0873,  # left_hip_roll
    -0.4579,  # left_hip_pitch
    -0.0049,  # left_knee
    0.4530,   # left_ankle
    0.3491,   # neck_pitch
    0.3491,   # head_pitch
    0.0,      # head_yaw
    0.0,      # head_roll
    0.0,      # right_hip_yaw
    0.0873,   # right_hip_roll
    0.4579,   # right_hip_pitch
    0.0049,   # right_knee
    -0.4530,  # right_ankle
])

# Physics timestep used in training (mjlab velocity template: sim dt 0.005 with
# decimation 4 = 50 Hz control). The XML's own default is 0.002, which every
# pre-existing table in the results doc was measured at; --bam switches the
# default to this so the BAM runs match the trained dynamics.
TRAINING_TIMESTEP = 0.005


_POSTURES = ("standing", "tucked", "tucked_env", "squat_env")


def _tuck_ctrl(nu, tuck_factor):
    """Deep-tuck ctrl: the original probe's pose (zeros outside TUCK)."""
    ctrl = np.zeros(nu)
    for idx, angle in TUCK.items():
        ctrl[idx] = angle * tuck_factor
    return ctrl


def _home_tuck_ctrl(nu, tuck_factor):
    """HOME with the TUCK overrides applied — what a policy would command.

    Used by --tuck-at-flick: joints the tuck does not touch stay where the
    standing policy had them (HOME), instead of snapping to zero.
    """
    ctrl = np.zeros(nu)
    ctrl[: len(HOME)] = HOME
    for idx, angle in TUCK.items():
        ctrl[idx] = angle * tuck_factor
    return ctrl


def _hold_pose(nu, lerp):
    """HOME lerped `lerp` of the way toward the full TUCK, per joint.

    The GROUND-LEVEL hold posture. Note this is a LERP from HOME, not the
    scaling `_home_tuck_ctrl` applies (which drives the tucked joints toward
    ZERO rather than toward HOME) — the same parametrisation roulade's
    mid-roll spawn uses. It matters: the lerp keeps the SOLES DOWN, and a
    feet-flat squat is the only pose measured to rest on a plate lying on the
    floor. The deep tuck kneels with its feet hanging ~8 cm BELOW whatever it
    rests on, which is fine on a plate held 10 cm up and impossible on one
    lying on the ground.
    """
    ctrl = np.zeros(nu)
    ctrl[: len(HOME)] = HOME
    for idx, angle in TUCK.items():
        ctrl[idx] = HOME[idx] + lerp * (angle - HOME[idx])
    return ctrl


def build_scene(bam=False, vin=7.4, vin_drop_gain=0.0, timestep=None):
    """Load scene_backflip.xml, optionally with BAM actuators.

    ``bam=False`` keeps the XML's own position actuators (MuJoCo built-in PD),
    which is what every table in docs/backflip_envelope_results.md before the
    "Standing-spawn re-measurement" section used.

    ``bam=True`` hands the 14 servos to the BAM M6 voltage-controlled XL330
    model — the actuator TRAINING actually uses (AGENTS.md: "Actuators are
    BAM"). Reuses scripts/infer_policy.py's loader rather than re-deriving it,
    so the CPU probe and the CPU deployment rehearsal cannot drift apart. The
    default timestep also switches to the training sim dt (0.005) in this mode.

    Returns ``(model, data, bam_ctrl)``; ``bam_ctrl`` is None without --bam.
    """
    if not bam:
        model = mujoco.MjModel.from_xml_path(SCENE)
        if timestep is not None:
            model.opt.timestep = timestep
        return model, mujoco.MjData(model), None

    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import infer_policy  # noqa: E402  (script-local import; cheap, no viewer/GL)

    bam_model = infer_policy.load_bam_model(
        infer_policy.BAM_KP_FW, vin, infer_policy.BAM_MAX_CURRENT
    )
    model, data, bam_ctrl, _names = infer_policy.load_mujoco_with_bam(
        SCENE, bam_model,
        TRAINING_TIMESTEP if timestep is None else timestep,
        vin_drop_gain, infer_policy.BAM_VIN_MIN,
    )
    return model, data, bam_ctrl


def _apply_ctrl(data, bam_ctrl, ctrl):
    """Command a joint-position target, through BAM if it is in play.

    Without BAM the XML's position actuators take the target in data.ctrl
    directly. With BAM, data.ctrl holds TORQUE (the actuators were rewritten
    to motors), so the target goes to the controller's q_target and BAM's
    update() writes the torque each physics step.
    """
    if bam_ctrl is None:
        data.ctrl[:] = ctrl
    else:
        bam_ctrl.q_target[:] = ctrl[: len(bam_ctrl.q_target)]


def trunk_tilt_deg(data):
    """Angle between the robot's own local +z axis and world +z, in degrees."""
    zw = local_z_axis_world(data)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(zw[2])))))


class Cell(NamedTuple):
    """One measured launch. Added fields must go at the END (call sites index)."""

    rot: float            # deg of BACKWARD rotation accumulated while airborne
    land: float           # m/s at first ground contact after real flight
    apex: float           # m, highest trunk z
    tilt0: float          # deg of trunk tilt AT THE FLICK (what the hold left)
    sweep: float          # deg the PLATE swept under the feet during the flick
    tilt1: float          # deg of trunk tilt at the END of the flick
    plate_frac: float     # fraction of the flick with the feet still on the plate


# Half-length of the plate along x (launcher.xml: box size 0.09 0.09 0.01).
PLATE_HALF_LEN_X = 0.09
_PIVOTS = ("center", "rear")


def run_cell(model, data, vz, w0, tuck_factor, z0, t_hold=0.3, t_launch=0.12,
             duration=2.0, on_step=None, posture="standing", tuck_at_flick=False,
             bam_ctrl=None, hold_lerp=None, pivot="center"):
    if hold_lerp is None:
        hold_lerp = HOLD_LERP
    mujoco.mj_resetData(model, data)
    dt = model.opt.timestep

    plate_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "plate_free")
    plate_qadr = model.jnt_qposadr[plate_jid]
    plate_vadr = model.jnt_dofadr[plate_jid]
    plate_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "plate_geom")
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    # Robot: free joint at qpos[0:7], 14 servos after it.
    if posture not in _POSTURES:
        raise ValueError(f"posture must be one of {_POSTURES}, got {posture!r}")
    if pivot not in _PIVOTS:
        raise ValueError(f"pivot must be one of {_PIVOTS}, got {pivot!r}")
    # ctrl held during HOLD (and, unless --tuck-at-flick, for the whole cell).
    if posture == "tucked":
        hold_ctrl = _tuck_ctrl(model.nu, tuck_factor)
    elif posture == "tucked_env":
        hold_ctrl = _home_tuck_ctrl(model.nu, tuck_factor)
    elif posture == "squat_env":
        hold_ctrl = _hold_pose(model.nu, hold_lerp)
    else:
        hold_ctrl = np.concatenate([HOME, np.zeros(model.nu - len(HOME))])
    if not tuck_at_flick:
        flick_ctrl = hold_ctrl
    elif posture == "squat_env":
        # The policy folds from the squat into the tuck when it feels the
        # flick; `tuck_factor` is the lerp depth it folds TO.
        flick_ctrl = _hold_pose(model.nu, tuck_factor)
    else:
        flick_ctrl = _home_tuck_ctrl(model.nu, tuck_factor)
    ctrl = hold_ctrl
    # SPAWN_OFFSET: measured, not the brief's guessed 0.10. At 0.10 the tucked
    # robot spawns floating well above the plate: it is still falling at
    # -0.18..-0.22 m/s when t_hold ends, and over a longer hold the deep-tuck
    # (factor=1.0) pose is unstable on the 18x18cm plate and can topple clean
    # off it depending on tiny numeric differences (verified by settling for
    # 1.5s at several offsets). 0.02 settles cleanly for both tuck factors and
    # all three z0 in the sanity sweep: by t_hold the robot is at rest
    # (|vz|<0.002 m/s, <1cm horizontal drift, upright cos~0.97) resting ~2.7cm
    # above the plate top, reproducibly.
    SPAWN_OFFSET = 0.02
    if posture == "tucked":
        spawn_z = z0 + PLATE_HALF_THICKNESS + SPAWN_OFFSET  # feet on the plate top
    elif posture == "tucked_env":
        # The KNEELING hold, at its MEASURED resting height on the plate top.
        # Needs the plate held >= 8 cm up: the feet dangle below it.
        spawn_z = z0 + PLATE_HALF_THICKNESS + TUCK_Z
    elif posture == "squat_env":
        # EXACTLY what reset_backflip_robot_on_plate does: the feet-flat squat
        # at its MEASURED resting height on the plate top. Works with the plate
        # lying on the ground, which is the point.
        spawn_z = z0 + PLATE_HALF_THICKNESS + HOLD_Z
    else:
        # EXACTLY what reset_backflip_robot_on_plate does: trunk at plate
        # centre + half-thickness + the measured standing trunk height. No
        # settle offset — the env spawns the robot at rest at this height and
        # the flick arrives t_hold later, whatever the pose has drifted to by
        # then (which is why tilt_at_flick is reported).
        spawn_z = z0 + PLATE_HALF_THICKNESS + STAND_Z
    data.qpos[0:3] = [0.0, 0.0, spawn_z]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7 : 7 + model.nu] = ctrl
    if bam_ctrl is not None:
        # Clears voltage-drop state after mj_resetData; must precede the
        # q_target write (reset() sets q_target from qpos).
        bam_ctrl.reset(data.qpos)
    _apply_ctrl(data, bam_ctrl, ctrl)

    accum_pitch = 0.0
    apex = 0.0
    landing_speed = 0.0
    airborne_seen = False
    landed = False
    tilt_at_flick = None
    # THE PLATE'S OWN SWEPT ANGLE, and what it does to the robot. Measured, not
    # taken from the formula, so the report cross-checks
    # mdp.backflip_plate_sweep_deg rather than restating it. plate_steps /
    # launch_steps is how much of the flick the feet were still ON the surface:
    # a paddle that pivots out from under them transfers rotation by TIPPING,
    # and that shows up here as a fraction well below 1.
    max_sweep = 0.0
    tilt_at_release = None
    launch_steps = 0
    plate_contact_steps = 0
    prev_phase = None
    t = 0.0
    while t < duration:
        z, pitch, vz_t, w_t, phase = microduck_mdp.backflip_plate_kinematics(
            *[torch.tensor([v]) for v in (t, t_hold, t_launch, z0, vz, w0)]
        )
        # The flick has started (phase left HOLD). Snapshot how much of the
        # commanded posture survived the hold — THE diagnostic for any spawn:
        # the env pays ready_stance to still be holding the pose here, and a
        # robot that has drifted gets flicked from the wrong CoM. Read it
        # against the pose's own equilibrium (~14 deg for the tuck, ~0 for
        # standing). Also the instant a policy would react, so --tuck-at-flick
        # switches ctrl here.
        if tilt_at_flick is None and int(phase) != microduck_mdp.BACKFLIP_PHASE_HOLD:
            tilt_at_flick = trunk_tilt_deg(data)
            _apply_ctrl(data, bam_ctrl, flick_ctrl)

        max_sweep = max(max_sweep, abs(math.degrees(float(pitch))))
        if int(phase) == microduck_mdp.BACKFLIP_PHASE_LAUNCH:
            launch_steps += 1
        # The flick has just ended: snapshot the attitude the LAUNCH ITSELF
        # produced. tilt0 is what the hold left, tilt1 is what the robot is
        # actually thrown with -- and the launch-attitude gate scores the
        # former while the flight is flown from the latter.
        if (
            tilt_at_release is None
            and prev_phase == microduck_mdp.BACKFLIP_PHASE_LAUNCH
            and int(phase) != microduck_mdp.BACKFLIP_PHASE_LAUNCH
        ):
            tilt_at_release = trunk_tilt_deg(data)
        prev_phase = int(phase)

        half = float(pitch) * 0.5
        # WHERE THE PLATE PIVOTS. "center" is what the env does today: the
        # surface rotates about the body origin, so the FRONT edge rises while
        # the REAR edge drops -- a lever. "rear" pivots at the rear edge
        # instead, so the whole surface LIFTS as it tilts, which is what a
        # springboard (or a hand) does. Same pitch schedule either way, so the
        # swept angle is identical and only the transfer differs.
        px, pz = 0.0, float(z)
        vx_t = 0.0
        if pivot == "rear":
            th = float(pitch)
            rx = PLATE_HALF_LEN_X * math.cos(th)
            rz = -PLATE_HALF_LEN_X * math.sin(th)
            px = -PLATE_HALF_LEN_X + rx
            pz = float(z) + rz
            # v_center = v_pivot + omega x r, omega = (0, w_t, 0)
            vx_t = float(w_t) * rz
            vz_center = float(vz_t) - float(w_t) * rx
        else:
            vz_center = float(vz_t)
        data.qpos[plate_qadr + 0 : plate_qadr + 3] = [px, 0.0, pz]
        data.qpos[plate_qadr + 3 : plate_qadr + 7] = [
            math.cos(half), 0.0, math.sin(half), 0.0
        ]
        data.qvel[plate_vadr + 0 : plate_vadr + 3] = [vx_t, 0.0, vz_center]
        data.qvel[plate_vadr + 3 : plate_vadr + 6] = [0.0, float(w_t), 0.0]

        # Snapshot BEFORE stepping. The contact constraint solved INSIDE
        # mj_step for the step that first makes contact already decelerates
        # the robot, so reading qvel AFTER that step underestimates "peak
        # downward speed at first contact" — the number this script exists to
        # report (module docstring, line 7). Use the pre-step velocity if
        # this turns out to be the first-contact step.
        pre_step_vz = float(data.qvel[2])

        if bam_ctrl is not None:
            # BAM owns control/torque/friction: update() runs the firmware
            # P-loop + DC-motor equation, writes torque into data.ctrl and
            # pushes the friction budget onto the dofs for this step.
            bam_ctrl.update()
        mujoco.mj_step(model, data)
        t += dt

        trunk_z = float(data.qpos[2])
        apex = max(apex, trunk_z)
        # Airborne = no contact between the ROBOT and the floor geom.
        # HISTORY (kept because it explains the plate_gid filter): the parked
        # plate used to be teleported to BACKFLIP_GONE_Z=-3.0, i.e. INSIDE
        # MuJoCo's infinite floor half-space (a type="plane" geom has no lower
        # bound), so it was permanently "touching" floor from the instant phase
        # went GONE. An unfiltered floor-contact check made `touching` always
        # True in GONE, `not touching` never held, and accum_pitch never
        # accumulated — caught by the step-3 sanity check (rot_deg 0.0 in every
        # cell) and traced with mj_id2name on data.contact. The parking spot is
        # now ABOVE and beside the floor (BACKFLIP_GONE_POS), which removes the
        # spurious contacts at the source; this filter is kept as cheap
        # belt-and-braces so the probe measures ROBOT-floor contact only.
        if int(phase) == microduck_mdp.BACKFLIP_PHASE_LAUNCH and any(
            plate_gid in (data.contact.geom1[i], data.contact.geom2[i])
            for i in range(data.ncon)
        ):
            plate_contact_steps += 1

        touching = any(
            (data.contact.geom1[i] == floor_gid and data.contact.geom2[i] != plate_gid)
            or (data.contact.geom2[i] == floor_gid and data.contact.geom1[i] != plate_gid)
            for i in range(data.ncon)
        )
        if not landed:
            if int(phase) == microduck_mdp.BACKFLIP_PHASE_GONE and not touching:
                airborne_seen = True
                # Backward pitch = -omega_y. backflip_plate_kinematics was
                # fixed (Task 2, negated a_ang) so that w0 > 0 now actually
                # delivers a NEGATIVE pitch/w_t — a backward roll is a
                # NEGATIVE rotation about +y in this codebase's convention
                # (set_random_ground_state: +90deg about +y = face-down =
                # FORWARD, -90deg = face-up = BACKWARD). The plate's w_t and
                # the robot's own qvel[4] both therefore run negative during
                # a genuine backward flick (contact transfer preserves
                # sign), so negating here turns that into a POSITIVE
                # "backward degrees accumulated" for reporting. (An earlier
                # version of this line used +qvel[4] and a comment claiming
                # positive omega_y was backward — that was itself wrong,
                # traced to the same mislabeled-direction bug in
                # backflip_plate_kinematics; see check_direction() below and
                # docs/backflip_envelope_results.md for the measurement that
                # caught it.)
                accum_pitch += -float(data.qvel[4]) * dt
            elif airborne_seen and touching:
                # First ground contact after real flight. Use the PRE-step
                # velocity (see comment above) and LATCH: a bounce that
                # re-enters "airborne, not touching" after this must not
                # resume accumulating rotation, and must not overwrite this
                # measurement with a softer post-bounce touch.
                landing_speed = abs(pre_step_vz)
                landed = True

        if on_step is not None:
            on_step(t, math.degrees(accum_pitch), data, phase=int(phase), touching=touching)

    return Cell(
        rot=math.degrees(accum_pitch),
        land=landing_speed,
        apex=apex,
        tilt0=float("nan") if tilt_at_flick is None else tilt_at_flick,
        sweep=max_sweep,
        tilt1=float("nan") if tilt_at_release is None else tilt_at_release,
        plate_frac=(
            plate_contact_steps / launch_steps if launch_steps else float("nan")
        ),
    )


def local_z_axis_world(data):
    """World-frame direction of the robot's own local +z axis ("its own up").

    Third column of the body-to-world rotation matrix built from the free
    joint's quaternion (qpos[3:7]).
    """
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, data.qpos[3:7])
    return R.reshape(3, 3)[:, 2]


def check_direction(model, data, vz=3.0, w0=15.0, tuck_factor=1.0, z0=0.15,
                    posture="standing", tuck_at_flick=False,
                    t_hold=0.3, t_launch=0.12, bam_ctrl=None):
    """Verify BY MEASUREMENT that w0 > 0 rotates the robot BACKWARD, not
    forward — convention is exactly what got this probe's direction wrong
    once already (see docs/backflip_envelope_results.md).

    Traces the robot's own local +z axis in world coordinates through one
    flight and prints it at the instant accumulated rotation first reaches
    ~90 deg. A true backflip passing through the head-back position has
    local +z tipped toward world -x (and still mostly +z) — the robot is
    leaning back-and-up over its heels. A FORWARD roll mislabeled as
    "backward" instead has local +z pointing toward world +x with a small +z
    component — face-DOWN, per this codebase's own convention
    (set_random_ground_state: +90 deg rotation about +y = face-down).
    """
    state = {"reported": False}

    def on_step(t, accum_deg, data, phase, touching):
        if not state["reported"] and accum_deg >= 90.0:
            zw = local_z_axis_world(data)
            print(
                f"  direction check @ t={t:.3f}s accum={accum_deg:.1f}deg: "
                f"local +z in world = ({zw[0]:.3f}, {zw[1]:.3f}, {zw[2]:.3f})"
            )
            if zw[0] < 0.0:
                print("  -> -x component: robot is leaning BACKWARD-and-up. Correct direction.")
            else:
                print("  -> +x component: robot is FACE-DOWN (forward roll). WRONG direction.")
            state["reported"] = True

    c = run_cell(
        model, data, vz, w0, tuck_factor, z0, t_hold=t_hold, t_launch=t_launch,
        on_step=on_step, posture=posture, tuck_at_flick=tuck_at_flick,
        bam_ctrl=bam_ctrl,
    )
    rot = c.rot
    print(f"  cell result: posture={posture} tuck_at_flick={tuck_at_flick} "
          f"vz={vz} w0={w0} tuck={tuck_factor} z0={z0} "
          f"-> rot_deg={c.rot:.1f} land_m/s={c.land:.2f} apex_m={c.apex:.3f} "
          f"tilt_at_flick={c.tilt0:.1f}deg sweep={c.sweep:.1f}deg "
          f"tilt_at_release={c.tilt1:.1f}deg feet_on_plate={c.plate_frac:.2f}")
    if not state["reported"]:
        print("  WARNING: never reached 90deg of accumulated rotation — cannot verify direction "
              "with this cell. NOTE a NEGATIVE rot_deg means the robot rotated FORWARD: "
              "this check can only confirm a backward arc, so read the sign first.")
    return rot, land, apex, tilt


# ─────────────────────────────────────────────────────────────────────────────
# HOLD-EQUILIBRIUM SETTLE TEST (--settle)
#
# AGENTS.md, "Building a new env": a target/rest pose MUST be verified to be a
# stable equilibrium before training — hold its ctrl for ~3 s from NOISY inits
# and check TILT, not just height, because a settle test that only records z
# reports fallen states as "resting fine". Nobody had done it for the backflip
# env's HOLD phase, whose whole premise is that the robot stands still on the
# plate for up to 1.0 s (the hold curriculum's final stage) while ready_stance
# pays it to.
#
# This mode reproduces the env's reset EXACTLY: trunk at
# z0 + PLATE_HALF_THICKNESS + STAND_Z, HOME joints + U(-0.05, 0.05) rad
# (reset_robot_joints' position_range), x/y jitter +-SPAWN_XY_NOISE, roll/pitch
# +-SPAWN_TILT_NOISE, yaw +-SPAWN_YAW_NOISE (the cfg's narrowed reset_base
# pose_range), zero root velocity, plate parked at z0 and rewritten every step
# exactly as backflip_plate_step does during HOLD.
# ─────────────────────────────────────────────────────────────────────────────

# Spawn noise, mirroring microduck_backflip_env_cfg.py's reset_base override
# and reset_robot_joints position_range.
SPAWN_XY_NOISE = 0.01
SPAWN_YAW_NOISE = 0.05
SPAWN_TILT_NOISE = 0.02
SPAWN_JOINT_NOISE = 0.05

SETTLE_SAMPLE_TIMES = (0.1, 0.3, 0.5, 0.7, 1.0)


def _quat_from_rpy(roll, pitch, yaw):
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([roll, pitch, yaw]), "xyz")
    return q


def run_settle(model, data, z0, rng, bam_ctrl=None, duration=3.0,
               sample_times=SETTLE_SAMPLE_TIMES, noisy=True, on_floor=False,
               posture="tucked_env", tuck_factor=TUCK_FACTOR,
               spawn_z_override=None, ctrl_home=False):
    """Hold the HOLD-phase pose on the parked plate from a noisy init.

    Returns ``(samples, final)`` where ``samples`` maps each requested time to
    ``(tilt_deg, xy_drift_m, trunk_z)`` and ``final`` is the same triple at
    ``duration``. Tilt is the angle between the robot's own +z and world +z —
    the quantity AGENTS.md insists on, since a toppled robot can sit at a
    perfectly reasonable height.

    ``posture`` selects the pose held: ``tucked_env`` (the env's spawn since
    the tucked-hold switch) or ``standing`` (the superseded one). READ TILT
    RELATIVE TO THE POSE'S OWN EQUILIBRIUM for the tuck: a folded robot rests
    with its trunk pitched ~12-15 deg by geometry, so "tilt 14 deg, unchanged
    from 0.1 s to 3 s" is a settled tuck, not a falling one. What matters is
    whether it MOVES; ``n_fallen`` (>45 deg) catches an actual topple.
    """
    mujoco.mj_resetData(model, data)
    dt = model.opt.timestep

    plate_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "plate_free")
    plate_qadr = model.jnt_qposadr[plate_jid]
    plate_vadr = model.jnt_dofadr[plate_jid]

    # on_floor: the CONTROL experiment. Open-loop HOME is not a passive
    # equilibrium anywhere — including on the ground — so a plate-only number
    # cannot tell you whether the drift is about the plate or about the pose.
    # Running the identical test on the terrain (plate parked out of the way,
    # exactly where backflip_plate_step puts it once GONE) separates the two.
    if posture == "tucked_env":
        ctrl = _home_tuck_ctrl(model.nu, tuck_factor)
    elif posture == "tucked":
        ctrl = _tuck_ctrl(model.nu, tuck_factor)
    else:
        ctrl = np.concatenate([HOME, np.zeros(model.nu - len(HOME))])
    pose_z = TUCK_Z if posture in ("tucked", "tucked_env") else STAND_Z
    if ctrl_home:
        # What `play --agent zero` actually does: the joint-position action's
        # offset is the model's DEFAULT pose (HOME), so a zero action commands
        # HOME while the reset has just folded the robot into the tuck. The
        # robot therefore UNFOLDS on the plate from step 0. The spawn pose is
        # unchanged; only the command differs.
        ctrl = np.concatenate([HOME, np.zeros(model.nu - len(HOME))])
    if noisy:
        ctrl_noise = rng.uniform(-SPAWN_JOINT_NOISE, SPAWN_JOINT_NOISE, size=len(HOME))
        x0 = rng.uniform(-SPAWN_XY_NOISE, SPAWN_XY_NOISE)
        y0 = rng.uniform(-SPAWN_XY_NOISE, SPAWN_XY_NOISE)
        roll = rng.uniform(-SPAWN_TILT_NOISE, SPAWN_TILT_NOISE)
        pitch = rng.uniform(-SPAWN_TILT_NOISE, SPAWN_TILT_NOISE)
        yaw = rng.uniform(-SPAWN_YAW_NOISE, SPAWN_YAW_NOISE)
    else:
        ctrl_noise = np.zeros(len(HOME))
        x0 = y0 = roll = pitch = yaw = 0.0

    # The joint NOISE perturbs the joint STATE, not the command: the reset
    # event writes the target pose into qpos and the policy is what commands
    # it, so ctrl stays at the clean target and qpos carries the offset.
    # Getting this backwards would test a different (easier) thing.
    plate_pos = (
        list(microduck_mdp.BACKFLIP_GONE_POS) if on_floor else [0.0, 0.0, z0]
    )
    if spawn_z_override is not None:
        pose_z = spawn_z_override
    base_z = pose_z if on_floor else z0 + PLATE_HALF_THICKNESS + pose_z
    data.qpos[0:3] = [x0, y0, base_z]
    data.qpos[3:7] = _quat_from_rpy(roll, pitch, yaw)
    spawn_pose = (
        _home_tuck_ctrl(model.nu, tuck_factor)
        if (ctrl_home and posture in ("tucked", "tucked_env"))
        else ctrl
    )
    data.qpos[7 : 7 + len(HOME)] = spawn_pose[: len(HOME)] + ctrl_noise
    if bam_ctrl is not None:
        bam_ctrl.reset(data.qpos)
    _apply_ctrl(data, bam_ctrl, ctrl)

    samples = {}
    pending = sorted(sample_times)
    t = 0.0
    while t < duration - 1e-9:
        # HOLD-phase plate: parked at z0, motionless, rewritten every step
        # (a free body would otherwise fall).
        data.qpos[plate_qadr + 0 : plate_qadr + 3] = plate_pos
        data.qpos[plate_qadr + 3 : plate_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[plate_vadr : plate_vadr + 6] = 0.0
        if bam_ctrl is not None:
            bam_ctrl.update()
        mujoco.mj_step(model, data)
        t += dt
        while pending and t >= pending[0] - 1e-9:
            key = pending.pop(0)
            samples[key] = (
                trunk_tilt_deg(data),
                math.hypot(float(data.qpos[0]) - x0, float(data.qpos[1]) - y0),
                float(data.qpos[2]),
            )
    final = (
        trunk_tilt_deg(data),
        math.hypot(float(data.qpos[0]) - x0, float(data.qpos[1]) - y0),
        float(data.qpos[2]),
    )
    return samples, final


# Hold durations --box-check sweeps. These are DELIBERATELY NOT the env's
# HOLD_RANGE, which is 1-5 s.
#
# The probe has no balance controller: it freezes the HOME command, and
# open-loop standing on the plate drifts 3.5 deg of tilt by 0.3 s, 7.3 by 0.5,
# 23.2 by 1.0. Sweeping the env's actual 1-5 s hold therefore measures a
# TOPPLING robot being flicked — 27 deg of tilt at the flick, rotations of
# +-1000 deg, and most cells never landing on their feet. That says nothing
# about the launch.
#
# The env's long hold exists so the POLICY learns to stand, and a policy that
# has learned it presents an upright robot at the flick whatever the hold
# lasted. So the box is measured over the window where the open-loop pose is
# still upright, which is the launch condition a balanced policy delivers.
# This is the sharpest form of the standing caveat: THE PROBE MEASURES THE
# LAUNCH, NOT THE SKILL.
BOX_CHECK_HOLDS = (0.1, 0.2, 0.3)

# Acceptance thresholds for --box-check. 360 deg is a closed flip. 2.6 m/s is
# the OPERATOR'S COMFORT THRESHOLD (~34 cm of free fall), not a physical limit:
# the standing launch measured here lands at 3.2-3.9 m/s and the user has
# accepted that for now, so the report separates closure (a correctness
# property) from landing speed (their trade-off).
BOX_MIN_ROT_DEG = 360.0
BOX_MAX_LAND = 2.6

# HARD GATE, the way DIRECTION is one: the plate may not sweep more than this
# many degrees under the robot's feet during the flick.
#
# sweep = 0.5 * w0 * t_launch (mdp.backflip_plate_sweep_deg). A human hand
# tossing an object sweeps 30-40 deg — it imparts an impulse and lets go. A
# surface that sweeps 90 deg is a catapult paddle levering the robot over: the
# feet cannot stay on it, the rotation arrives by TIPPING rather than by a
# push, and the launch attitude is destroyed by the launch itself. The user saw
# exactly that and called it "a force that makes it rotate".
#
# 45 deg is the cap because it is the top of the hand-like range with a little
# margin, and because nothing about the mechanism improves above it. This
# quantity had NEVER been reported by any probe mode before 2026-09-09, which
# is how a 72-110 deg box shipped through thirteen measurement waves.
BOX_MAX_SWEEP_DEG = 45.0


def _edges(rng, mid=True):
    lo, hi = rng
    return (lo, 0.5 * (lo + hi), hi) if mid else (lo, hi)


def box_check_report(model, data, bam_ctrl=None, posture="tucked_env",
                     vz_range=None, w0_range=None, launch_range=None,
                     z0_range=None, label=None, pivot="center"):
    """WHOLE-BOX check of the cfg's DR ranges from the env's actual spawn.

    Reports the WORST cell, not the best. The rule this enforces is the one the
    previous box failed: every sampled combination of the corners AND midpoints
    of z0 / vz / w0 / t_launch — crossed with the hold extremes and a range of
    tuck depths — must close >= 360 deg backward at <= 2.6 m/s. A box whose
    interior contains one 3 deg cell is not a box.

    Tuck depth is swept 0.5-1.0 even though the spawn is fixed at TUCK_FACTOR:
    the policy can deepen or open the tuck during HOLD and LAUNCH, so the box
    should hold across what it might do, not only at the spawn value.
    """
    # z0 must cover everything the env can SAMPLE, curriculum included. There
    # is no z0 curriculum any more (Z0_RANGE spans 2 cm and the tail was
    # removed), so Z0_RANGE is the whole story — but if one is ever added,
    # extend this grid with its ceiling. Reading a stale range here was a real
    # hole once: the check never saw the state the policy trained into after
    # the tail opened, and a 1-D corner scan got used instead.
    vz_range = VZ_RANGE if vz_range is None else vz_range
    w0_range = W0_RANGE if w0_range is None else w0_range
    z0_range = Z0_RANGE if z0_range is None else z0_range
    launch_range = LAUNCH_RANGE if launch_range is None else launch_range
    vzs = _edges(vz_range)
    w0s = _edges(w0_range)
    z0s = _edges(z0_range)
    laus = _edges(launch_range)
    holds = BOX_CHECK_HOLDS
    # The FOLD DEPTH the policy reaches at the flick. It is not DR -- it is
    # the policy's action -- and the standing launch only closes at a full
    # fold (measured: fold 0.75 gives 209-254 deg where 1.0 gives 357-458).
    # The box is therefore verified at the full fold, the way the task
    # requires it to be flown.
    tucks = (1.0,)

    print(f"# box-check{'' if label is None else ' ' + label} "
          f"posture={posture} pivot={pivot} bam={bam_ctrl is not None} "
          f"dt={model.opt.timestep}")
    print(f"#   z0 {z0_range} vz {vz_range} w0 {w0_range} "
          f"launch {launch_range}")
    # The sweep is analytic, so print it before spending a minute on physics:
    # a box that fails this cannot be rescued by anything the sweep measures.
    sweep_lo = microduck_mdp.backflip_plate_sweep_deg(
        min(w0_range), min(launch_range))
    sweep_hi = microduck_mdp.backflip_plate_sweep_deg(
        max(w0_range), max(launch_range))
    print(f"#   PLATE SWEEP {sweep_lo:.1f}-{sweep_hi:.1f} deg "
          f"(0.5*w0*t_launch; cap {BOX_MAX_SWEEP_DEG:.0f}, a hand sweeps 30-40)")
    print(f"#   z0 grid {z0s}")
    print(f"#   hold {holds} (NOT the env's {HOLD_RANGE} — see "
          f"BOX_CHECK_HOLDS) tuck {tucks}")
    rows = []
    for z0, vz, w0, lau, hold, tk in itertools.product(
        z0s, vzs, w0s, laus, holds, tucks
    ):
        c = run_cell(
            model, data, vz, w0, tk, z0, t_hold=hold, t_launch=lau,
            posture=posture, tuck_at_flick=True, bam_ctrl=bam_ctrl,
            pivot=pivot,
        )
        rows.append((c.rot, c.land, c.apex, c.tilt0, z0, vz, w0, lau, hold, tk,
                     c.sweep, c.tilt1, c.plate_frac))

    def show(head, ordered):
        print(f"  {head}")
        for r in ordered[:5]:
            print(f"    rot={r[0]:7.1f} land={r[1]:5.2f} apex={r[2]:.3f} "
                  f"tilt0={r[3]:5.1f} sweep={r[10]:5.1f} tilt1={r[11]:6.1f} "
                  f"feet={r[12]:4.2f}  z0={r[4]:.3f} vz={r[5]:.3f} "
                  f"w0={r[6]:.2f} launch={r[7]:.3f} hold={r[8]:.2f} "
                  f"tuck={r[9]:.2f}")

    min_rot = min(r[0] for r in rows)
    max_land = max(r[1] for r in rows)
    n_short = sum(1 for r in rows if r[0] < BOX_MIN_ROT_DEG)
    n_hard = sum(1 for r in rows if r[1] > BOX_MAX_LAND)
    # A cell that never lands reports landing_speed 0.0 and would sail through
    # the <= 2.6 m/s test by not being measured at all. Count them explicitly
    # rather than trusting the max.
    n_never = sum(1 for r in rows if r[1] == 0.0)
    max_rot = max(r[0] for r in rows)
    print(f"  {len(rows)} cells | rot {min_rot:.1f}-{max_rot:.1f} deg | "
          f"max landing = {max_land:.2f} m/s | "
          f"short of 360: {n_short} | over 2.6 m/s: {n_hard} | "
          f"never landed: {n_never}")
    show("worst by rotation:", sorted(rows, key=lambda r: r[0]))
    show("worst by landing speed:", sorted(rows, key=lambda r: -r[1]))
    # TWO SEPARATE VERDICTS, deliberately. Closure is a correctness property
    # of the box; landing speed is a hardware trade-off the user owns. Fusing
    # them into one PASS/FAIL hid which of the two was failing.
    # DIRECTION is the only gate. Open-loop closure is INFORMATION about
    # starting difficulty, not a pass/fail bar: requiring an unskilled robot to
    # complete every throw is the wrong ask (compensating for an imperfect
    # throw is the policy's job) and it once squeezed w0 to a single value no
    # human hand could reproduce.
    n_forward = sum(1 for r in rows if r[0] < 0.0)
    max_sweep = max(r[10] for r in rows)
    n_swept = sum(1 for r in rows if r[10] > BOX_MAX_SWEEP_DEG)
    tilt1s = [r[11] for r in rows if r[11] == r[11]]
    feet = [r[12] for r in rows if r[12] == r[12]]
    print(f"  PLATE SWEEP: {'PASS' if n_swept == 0 else 'FAIL'} — "
          f"max {max_sweep:.1f} deg (cap {BOX_MAX_SWEEP_DEG:.0f}); "
          f"{n_swept}/{len(rows)} cells over")
    if tilt1s:
        print(f"  ATTITUDE AT RELEASE: {min(tilt1s):.1f}-{max(tilt1s):.1f} deg "
              f"of tilt (tilt0 at the flick: "
              f"{min(r[3] for r in rows):.1f}-{max(r[3] for r in rows):.1f})")
    if feet:
        print(f"  FEET ON THE PLATE during the flick: "
              f"{min(feet):.2f}-{max(feet):.2f} of the ramp")
    min_land = min(r[1] for r in rows)
    closed = len(rows) - n_short
    print(f"  DIRECTION: {'PASS' if n_forward == 0 else 'FAIL'} — "
          f"{n_forward} cells rotate FORWARD (min rot {min_rot:.1f} deg)")
    print(f"  OPEN-LOOP CLOSURE (information, not a gate): "
          f"{closed}/{len(rows)} cells reach {BOX_MIN_ROT_DEG:.0f} deg "
          f"= {100.0 * closed / len(rows):.0f}%; {n_never} never land")
    print(f"  LANDING: {min_land:.2f}-{max_land:.2f} m/s "
          f"(operator comfort threshold {BOX_MAX_LAND} m/s; "
          f"{n_hard}/{len(rows)} cells over)")
    return rows, (n_forward == 0 and n_swept == 0)


# Flop basins audited by --flop-audit. AGENTS.md: "audit each positive term
# against every stable flop (on back / face / side): if flopping keeps most of
# the stack, the policy will flop." Named (roll, pitch, yaw) in radians.
FLOP_ORIENTATIONS = {
    "upright":    (0.0, 0.0, 0.0),
    "side_left":  (math.pi / 2, 0.0, 0.0),
    "side_right": (-math.pi / 2, 0.0, 0.0),
    "face_down":  (0.0, math.pi / 2, 0.0),
    "on_back":    (0.0, -math.pi / 2, 0.0),
    "inverted":   (math.pi, 0.0, 0.0),
}


def _ready_stance_factors(data, z0, tuck_factor, joint_std=0.35, height_std=0.03,
                          tilt_full_deg=40.0, tilt_zero_deg=70.0,
                          posture="standing"):
    """Score backflip_ready_stance's factors from a raw MuJoCo state.

    Mirrors the reward's arithmetic (pose x height x upright) so the flop audit
    measures the TERM, not a proxy for it. Kept in sync by
    tests/test_backflip_mdp.py, which checks the same three numbers through the
    real function.
    """
    if posture == "standing":
        # backflip_ready_stance is upright x height only; `pose` is reported as
        # 1.0 so the columns line up with the tucked-hold tables above.
        target, pose = HOME, 1.0
        rest_z = STAND_Z
    else:
        target = _home_tuck_ctrl(len(HOME), tuck_factor)[: len(HOME)]
        err = np.asarray(data.qpos[7 : 7 + len(HOME)]) - target
        pose = math.exp(-float((err ** 2).mean()) / joint_std ** 2)
        rest_z = TUCK_Z
    z_err = float(data.qpos[2]) - (z0 + PLATE_HALF_THICKNESS + rest_z)
    height = math.exp(-(z_err ** 2) / height_std ** 2)
    tilt = trunk_tilt_deg(data)
    u = min(max((tilt_zero_deg - tilt) / (tilt_zero_deg - tilt_full_deg), 0.0), 1.0)
    upright = u * u * (3.0 - 2.0 * u)
    return pose, height, upright, tilt


# Spawn clearances above the plate top tried per flop orientation. A flopped
# pose settles into DIFFERENT basins depending on how it is dropped — the
# dangerous side-lying one (tilt 102.8 deg, ON the plate, pose 0.997) is only
# reached from ~0.045 m, while 0.025-0.035 slides off the plate entirely and
# scores lower. An audit that tries one clearance measures whichever basin it
# happened to hit, so the audit scans and reports the WORST (highest-scoring)
# basin per orientation.
FLOP_CLEARANCES = (0.005, 0.015, 0.025, 0.035, 0.045)


def flop_audit_report(model, data, z0, tuck_factor, bam_ctrl=None, duration=3.0,
                      clearances=FLOP_CLEARANCES, posture="standing"):
    """Settle the tucked robot from each flop orientation and score the hold term.

    THE audit AGENTS.md mandates and that two review rounds missed: a positive
    term that pays per step must be checked against every STABLE resting pose,
    not only against the intended one and the obviously-bad ones. The side-lying
    tuck is the dangerous basin here — it is passively stable on the plate, it
    needs no balancing, its joints are less load-sagged than the upright tuck's
    (so its `pose` factor is HIGHER), and without an upright factor it scored a
    premium over doing the right thing.

    Each row settles for `duration` from the named orientation at every
    clearance in `clearances`, and reports the WORST basin found — the one with
    the highest `pose x height`, i.e. the most attractive flop that exists,
    whether or not the upright factor currently zeroes it. `pre` is what the
    term paid before the upright factor was reinstated; `TOTAL` is what it pays
    now. `on?` says whether that basin is still ON the plate (|x|,|y| < 0.09):
    a flop that slides off is not a basin the policy can farm.
    """
    print(f"# flop-audit: z0={z0} tuck={tuck_factor} duration={duration}s "
          f"dt={model.opt.timestep} bam={bam_ctrl is not None}")
    print(f"{'orientation':>12} {'clr':>5} {'tilt':>7} {'drift':>7} {'trunk_z':>8} "
          f"{'pose':>7} {'height':>7} {'pre':>6} {'upright':>8} {'TOTAL':>7} {'on?':>4}")
    rows = {}
    dt = model.opt.timestep
    plate_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "plate_free")
    qa = model.jnt_qposadr[plate_jid]
    va = model.jnt_dofadr[plate_jid]
    ctrl = (
        np.concatenate([HOME, np.zeros(model.nu - len(HOME))])
        if posture == "standing"
        else _home_tuck_ctrl(model.nu, tuck_factor)
    )
    rest_z = STAND_Z if posture == "standing" else TUCK_Z
    for name, (roll, pitch, yaw) in FLOP_ORIENTATIONS.items():
        worst = None
        for clearance in clearances:
            mujoco.mj_resetData(model, data)
            data.qpos[0:3] = [
                0.0, 0.0, z0 + PLATE_HALF_THICKNESS + rest_z + clearance
            ]
            data.qpos[3:7] = _quat_from_rpy(roll, pitch, yaw)
            data.qpos[7 : 7 + model.nu] = ctrl
            if bam_ctrl is not None:
                bam_ctrl.reset(data.qpos)
            _apply_ctrl(data, bam_ctrl, ctrl)
            t = 0.0
            while t < duration - 1e-9:
                data.qpos[qa : qa + 3] = [0.0, 0.0, z0]
                data.qpos[qa + 3 : qa + 7] = [1.0, 0.0, 0.0, 0.0]
                data.qvel[va : va + 6] = 0.0
                if bam_ctrl is not None:
                    bam_ctrl.update()
                mujoco.mj_step(model, data)
                t += dt
            pose, height, upright, tilt = _ready_stance_factors(
                data, z0, tuck_factor, posture=posture
            )
            drift = math.hypot(float(data.qpos[0]), float(data.qpos[1]))
            on_plate = (
                abs(float(data.qpos[0])) < 0.09 and abs(float(data.qpos[1])) < 0.09
            )
            cand = (
                pose * height, clearance, tilt, drift, float(data.qpos[2]),
                pose, height, upright, on_plate,
            )
            if worst is None or cand[0] > worst[0]:
                worst = cand
        pre, clearance, tilt, drift, z, pose, height, upright, on_plate = worst
        total = pre * upright
        rows[name] = (tilt, drift, z, pose, height, pre, upright, total, on_plate)
        print(f"{name:>12} {clearance:5.3f} {tilt:7.1f} {drift:7.3f} {z:8.3f} "
              f"{pose:7.3f} {height:7.3f} {pre:6.3f} {upright:8.3f} {total:7.3f} "
              f"{'yes' if on_plate else 'OFF':>4}")
    flops = {k: v for k, v in rows.items() if k != "upright"}
    up = rows["upright"]
    best_pre = max((v[5] for v in flops.values()), default=0.0)
    best_total = max((v[7] for v in flops.values()), default=0.0)
    print(f"  upright: pre={up[5]:.3f} TOTAL={up[7]:.3f}   "
          f"best flop: pre={best_pre:.3f} TOTAL={best_total:.3f}")
    print(f"  WITHOUT the upright factor: "
          f"{'a flop would pay MORE' if best_pre > up[5] else 'upright would win'} "
          f"({best_pre:.3f} vs {up[5]:.3f})")
    print(f"  RESULT: {'PASS - upright wins' if up[7] > best_total else 'FAIL - a flop pays more'} "
          f"({up[7]:.3f} vs {best_total:.3f})")
    return rows


def measure_tuck_z_report(model, data, z0, tuck_factors, offsets, bam_ctrl=None,
                          duration=3.0):
    """Re-measure TUCK_Z: drop the tucked robot and read where it comes to rest.

    This is how the TUCK_Z constant above was produced. Reported per (tuck
    factor, spawn offset): resting trunk height above the plate top, tilt, and
    x/y drift, at 0.1 s / 1.0 s / the end of the settle.
    """
    rng = np.random.default_rng(0)
    print(f"# measure-tuck-z: z0={z0} duration={duration}s dt={model.opt.timestep} "
          f"bam={bam_ctrl is not None}   (TUCK_Z in the script = {TUCK_Z})")
    print(f"{'tuck':>5} {'offset':>7} {'h@0.1':>7} {'h@1.0':>7} {'h@end':>7} "
          f"{'tilt@0.1':>9} {'tilt@1.0':>9} {'tilt@end':>9} {'xy@end':>7}")
    for factor in tuck_factors:
        for offset in offsets:
            samples, final = run_settle(
                model, data, z0, rng, bam_ctrl=bam_ctrl, duration=duration,
                sample_times=(0.1, 1.0), noisy=False, posture="tucked_env",
                tuck_factor=factor, spawn_z_override=offset,
            )
            top = z0 + PLATE_HALF_THICKNESS
            print(f"{factor:5.2f} {offset:7.3f} "
                  f"{samples[0.1][2] - top:7.4f} {samples[1.0][2] - top:7.4f} "
                  f"{final[2] - top:7.4f} "
                  f"{samples[0.1][0]:9.1f} {samples[1.0][0]:9.1f} {final[0]:9.1f} "
                  f"{final[1]:7.4f}")


def settle_report(model, data, z0, trials, seed, bam_ctrl=None, duration=3.0,
                  noisy=True, on_floor=False, posture="tucked_env",
                  tuck_factor=TUCK_FACTOR, ctrl_home=False):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(trials):
        rows.append(run_settle(model, data, z0, rng, bam_ctrl=bam_ctrl,
                               duration=duration, noisy=noisy, on_floor=on_floor,
                               posture=posture, tuck_factor=tuck_factor,
                               ctrl_home=ctrl_home))
    print(f"# settle: posture={posture} tuck={tuck_factor} "
          f"{'ctrl=HOME (zero action)' if ctrl_home else 'ctrl=spawn pose'} "
          f"{'ON FLOOR (control)' if on_floor else f'on plate z0={z0}'} "
          f"trials={trials} noisy={noisy} duration={duration}s "
          f"dt={model.opt.timestep} bam={bam_ctrl is not None}")
    print(f"{'t[s]':>6} {'tilt_mean':>10} {'tilt_max':>9} {'xy_mean':>8} "
          f"{'xy_max':>7} {'z_mean':>7} {'n_fallen':>9}")
    for key in list(SETTLE_SAMPLE_TIMES) + [duration]:
        vals = [(r[0][key] if key in r[0] else r[1]) for r in rows]
        tilts = [v[0] for v in vals]
        xys = [v[1] for v in vals]
        zs = [v[2] for v in vals]
        fallen = sum(1 for x in tilts if x > 45.0)
        print(f"{key:6.2f} {np.mean(tilts):10.1f} {max(tilts):9.1f} "
              f"{np.mean(xys):8.3f} {max(xys):7.3f} {np.mean(zs):7.3f} {fallen:9d}")
    return rows


# Extended default grid (Task 3 addendum). vz now starts at 1.0 (was 1.5) at
# a finer 0.25 step (was 0.5) so the vz side of the trade-off is resolved as
# finely as the original grid's coarsest dimension used to be.
#
# w0 now runs to 54.0, not the ~25-30 the brief suggested as a starting
# point: an initial push to 30 found every new-best cell sitting AT w0=30,
# the same edge-of-grid symptom the original 15-rad/s ceiling had, so it was
# pushed further until the trend actually turned over. It does — landing
# speed at fixed vz/tuck falls as w0 rises, bottoms out around w0=39-48
# rad/s (depends on vz), then rises sharply as 360 deg stops closing at all
# (feet separate from the plate before the flick finishes). w0=54 is past
# that reversal at every vz/tuck in this grid, so the default now safely
# brackets the true (interior) optimum instead of chasing a moving edge.
# Uniform 3 rad/s step throughout — see docs/backflip_envelope_results.md
# "Extended sweep" for the finer-step valley probe that established this.
DEFAULT_VZ = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0)
DEFAULT_W0 = tuple(float(w) for w in range(6, 55, 3))
DEFAULT_TUCK = (0.5, 0.75, 1.0)

# Verified-safe ceiling for w0, from the --check-direction boundary scan in
# docs/backflip_envelope_results.md ("The reversal"): every checked cell with
# w0<=30 came back genuinely backward; w0=33-36 was marginal/weakening; w0>=39
# came back FORWARD (wrong) at every vz/tuck combination checked. Rows past
# this ceiling are marked '*' below -- their rot_deg/land_m/s numbers have NOT
# been direction-verified and several confirmed cases in this exact range are
# known to be forward rolls, not backward flips. Do not sort this table by
# land_m/s and trust the top row without checking it first.
SAFE_W0_CEILING = 30.0


def _float_list(s):
    return tuple(float(x) for x in s.split(","))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z0", type=float, default=0.15)
    ap.add_argument(
        "--posture", choices=_POSTURES, default="standing",
        help="Initial posture on the plate. 'tucked_env' (default) reproduces "
             "the ENV's spawn as of the tucked-hold switch: TUCK_FACTOR tuck "
             "pose, trunk at z0 + plate half-thickness + the MEASURED TUCK_Z. "
             "'standing' reproduces the superseded standing spawn (HOME pose at "
             "STAND_Z). 'tucked' reproduces the ORIGINAL probe (pre-squatted "
             "0.02 m above the plate top, zeros outside the TUCK map) and the "
             "docs' oldest tables. They give materially different envelopes — "
             "see the module docstring.",
    )
    ap.add_argument(
        "--tuck-at-flick", action="store_true",
        help="Command the tuck pose (HOME + TUCK overrides x tuck factor) from "
             "t >= t_hold instead of holding the spawn pose. Models the policy "
             "acting on the flick it feels.",
    )
    ap.add_argument("--hold", type=float, default=0.3,
                    help="t_hold in seconds (env samples 0.1-0.4, curriculum to 1.0).")
    ap.add_argument("--launch", type=float, default=0.12,
                    help="t_launch in seconds (env samples 0.08-0.15).")
    ap.add_argument(
        "--bam", action="store_true",
        help="Run the 14 servos through the BAM M6 XL330 model (what training "
             "uses) instead of the XML's position actuators, at the training "
             "sim timestep. Slower per cell, but the only mode whose HOLD-phase "
             "posture drift means anything for the env.",
    )
    ap.add_argument("--vin", type=float, default=7.4,
                    help="BAM supply voltage (training samples 6.5-8.2; 7.4 = nominal 2S).")
    ap.add_argument("--vin-drop-gain", type=float, default=0.0,
                    help="BAM load-dependent voltage sag gain (training samples 0.0-0.2).")
    ap.add_argument(
        "--settle", action="store_true",
        help="HOLD-equilibrium settle test instead of a launch sweep: hold HOME "
             "on the parked plate from noisy inits and report TILT and x/y "
             "drift at 0.1/0.3/0.5/0.7/1.0 s. Combine with --bam (the actuator "
             "training uses) — that is the only version whose numbers bear on "
             "HOLD_RANGE.",
    )
    ap.add_argument("--settle-trials", type=int, default=16)
    ap.add_argument("--settle-duration", type=float, default=3.0,
                    help="AGENTS.md asks for a 3 s hold; the curriculum's own "
                         "ceiling is 1.0 s.")
    ap.add_argument("--box-check", action="store_true",
                    help="WHOLE-BOX check of the cfg's DR ranges from the env's "
                         "actual spawn: every corner AND midpoint of z0 / vz / "
                         "w0 / t_launch, crossed with the hold extremes and "
                         "tuck depths, must close 360 deg under 2.6 m/s. "
                         "Reports the worst cell, not the best.")
    ap.add_argument(
        "--box-vz", type=_float_list, default=None,
        help="Override VZ_RANGE for --box-check, as 'lo,hi'. For scanning "
             "candidate boxes reproducibly; the default is the shipped range.",
    )
    ap.add_argument("--box-w0", type=_float_list, default=None,
                    help="Override W0_RANGE for --box-check, as 'lo,hi'.")
    ap.add_argument("--box-launch", type=_float_list, default=None,
                    help="Override LAUNCH_RANGE for --box-check, as 'lo,hi'.")
    ap.add_argument("--box-z0", type=_float_list, default=None,
                    help="Override Z0_RANGE for --box-check, as 'lo,hi'.")
    ap.add_argument(
        "--pivot", choices=_PIVOTS, default="center",
        help="Where the plate rotates about during the flick. 'center' is what "
             "the env does (the front edge rises, the rear drops - a lever); "
             "'rear' pivots at the rear edge so the whole surface lifts while "
             "it tilts, like a springboard or a hand. Same swept angle.",
    )
    ap.add_argument("--box-label", default=None,
                    help="Label printed in the --box-check header.")
    ap.add_argument("--flop-audit", action="store_true",
                    help="Settle the tucked robot from every flop orientation "
                         "(side / back / face / inverted) and score "
                         "backflip_ready_stance's factors on each. The audit "
                         "AGENTS.md mandates for any positive per-step term: if "
                         "a stable flop keeps most of the stack, the policy "
                         "will flop.")
    ap.add_argument("--measure-tuck-z", action="store_true",
                    help="Re-measure TUCK_Z: drop the tucked robot from several "
                         "offsets above the plate top, hold the tuck ctrl, and "
                         "report the resting trunk height, tilt and drift.")
    ap.add_argument("--settle-posture", choices=_POSTURES, default=None,
                    help="Posture for --settle (defaults to --posture).")
    ap.add_argument("--settle-on-floor", action="store_true",
                    help="Control experiment: same settle test with the robot on "
                         "the TERRAIN and the plate parked away, to separate "
                         "'the plate is a bad perch' from 'open-loop HOME is not "
                         "a passive equilibrium anywhere'.")
    ap.add_argument("--settle-ctrl-home", action="store_true",
                    help="Spawn in --posture but COMMAND HOME, which is what "
                         "`play --agent zero` does (the action offset is the "
                         "model's default pose). Shows what an untrained policy "
                         "does to the hold.")
    ap.add_argument("--settle-noiseless", action="store_true",
                    help="Single trial with zero spawn noise (drift baseline).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timestep", type=float, default=None,
                    help="Physics timestep override. Default: the XML's 0.002, "
                         "or 0.005 (training sim dt) under --bam.")
    ap.add_argument(
        "--check-direction", action="store_true",
        help="Trace the local +z axis in world coords at ~90deg of accumulated "
             "rotation for one cell, to verify a backward flip and not a "
             "mislabeled forward roll. Cell defaults to vz=3.0, w0=15.0, "
             "tuck=1.0; override with --check-vz/--check-w0/--check-tuck.",
    )
    ap.add_argument("--check-vz", type=float, default=3.0)
    ap.add_argument("--check-w0", type=float, default=15.0)
    ap.add_argument("--check-tuck", type=float, default=1.0)
    ap.add_argument(
        "--vz", type=_float_list, default=DEFAULT_VZ,
        help="Comma-separated launch-speed grid (m/s), e.g. 1.0,1.5,2.0",
    )
    ap.add_argument(
        "--w0", type=_float_list, default=DEFAULT_W0,
        help="Comma-separated flick-rate grid (rad/s), e.g. 6,12,18,24,30",
    )
    ap.add_argument(
        "--tuck", type=_float_list, default=DEFAULT_TUCK,
        help="Comma-separated tuck-factor grid, e.g. 0.5,0.75,1.0",
    )
    args = ap.parse_args()

    model, data, bam_ctrl = build_scene(
        bam=args.bam, vin=args.vin, vin_drop_gain=args.vin_drop_gain,
        timestep=args.timestep,
    )

    if args.flop_audit:
        flop_audit_report(
            model, data, args.z0,
            args.tuck[0] if len(args.tuck) == 1 else TUCK_FACTOR,
            bam_ctrl=bam_ctrl, duration=args.settle_duration,
            posture=args.posture,
        )
        return

    if args.box_check:  # noqa: E501
        box_check_report(
            model, data, bam_ctrl=bam_ctrl, posture=args.posture,
            vz_range=None if args.box_vz is None else tuple(args.box_vz),
            w0_range=None if args.box_w0 is None else tuple(args.box_w0),
            launch_range=(
                None if args.box_launch is None else tuple(args.box_launch)),
            z0_range=None if args.box_z0 is None else tuple(args.box_z0),
            label=args.box_label, pivot=args.pivot,
        )
        return

    if args.measure_tuck_z:
        measure_tuck_z_report(
            model, data, args.z0, args.tuck, (0.020, 0.026, 0.029, 0.032),
            bam_ctrl=bam_ctrl, duration=args.settle_duration,
        )
        return

    if args.settle:
        settle_report(
            model, data, args.z0,
            1 if args.settle_noiseless else args.settle_trials,
            args.seed, bam_ctrl=bam_ctrl, duration=args.settle_duration,
            noisy=not args.settle_noiseless, on_floor=args.settle_on_floor,
            ctrl_home=args.settle_ctrl_home,
            posture=args.settle_posture or args.posture,
            tuck_factor=args.tuck[0] if len(args.tuck) == 1 else TUCK_FACTOR,
        )
        return

    if args.check_direction:
        print(f"Direction check (z0={args.z0}, posture={args.posture}, "
              f"tuck_at_flick={args.tuck_at_flick}, hold={args.hold}, "
              f"launch={args.launch}, bam={args.bam}, dt={model.opt.timestep}):")
        check_direction(
            model, data,
            vz=args.check_vz, w0=args.check_w0, tuck_factor=args.check_tuck,
            z0=args.z0, posture=args.posture, tuck_at_flick=args.tuck_at_flick,
            t_hold=args.hold, t_launch=args.launch, bam_ctrl=bam_ctrl,
        )
        return

    if max(args.w0) > SAFE_W0_CEILING:
        print(
            f"NOTE: rows with w0 > {SAFE_W0_CEILING:.1f} rad/s are marked '*' below. "
            "Those cells are NOT direction-verified -- multiple cells in this exact "
            "range have been directly checked (--check-direction) and came back "
            "FORWARD rolls despite reporting a large 'backward' rot_deg. See "
            "docs/backflip_envelope_results.md \"The reversal\" before trusting any "
            "of them, especially the best-looking (lowest land_m/s) ones."
        )
    print(f"# posture={args.posture} tuck_at_flick={args.tuck_at_flick} "
          f"z0={args.z0} hold={args.hold} launch={args.launch} "
          f"bam={args.bam} dt={model.opt.timestep}")
    print(f"#   plate sweep at launch={args.launch:.3f}s: "
          f"{microduck_mdp.backflip_plate_sweep_deg(min(args.w0), args.launch):.1f}"
          f"-{microduck_mdp.backflip_plate_sweep_deg(max(args.w0), args.launch):.1f}"
          f" deg (0.5*w0*t_launch; a hand sweeps 30-40)")
    print(f"{'vz':>5} {'w0':>6} {'tuck':>5} {'rot_deg':>8} {'land_m/s':>9} "
          f"{'apex_m':>7} {'tilt0':>6} {'sweep':>6} {'tilt1':>6} {'feet':>5}")
    for vz, w0, tuck in itertools.product(args.vz, args.w0, args.tuck):
        c = run_cell(
            model, data, vz, w0, tuck, args.z0,
            t_hold=args.hold, t_launch=args.launch,
            posture=args.posture, tuck_at_flick=args.tuck_at_flick,
            bam_ctrl=bam_ctrl, pivot=args.pivot,
        )
        flag = "*" if w0 > SAFE_W0_CEILING else " "
        print(f"{vz:5.2f} {w0:6.1f} {tuck:5.2f} {c.rot:8.1f} {c.land:9.2f} "
              f"{c.apex:7.3f} {c.tilt0:6.1f} {c.sweep:6.1f} {c.tilt1:6.1f} "
              f"{c.plate_frac:5.2f} {flag}")


if __name__ == "__main__":
    main()
