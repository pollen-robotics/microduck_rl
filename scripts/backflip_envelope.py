#!/usr/bin/env python3
"""Measure the Microduck backflip envelope on CPU MuJoCo, before any training.

Sweeps launch speed x flick rate x tuck depth with the robot held at a FIXED
tuck pose (no policy), and records, per cell:
  • total backward pitch accumulated while airborne (deg)
  • peak downward speed at first ground contact (m/s) — the hardware risk
  • apex height (m)

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
direction first. Override the grid with --vz/--w0/--tuck; --z0 is unchanged.
"""

import argparse
import itertools
import math

import mujoco
import numpy as np
import torch

from mjlab_microduck.tasks import mdp as microduck_mdp

SCENE = "src/mjlab_microduck/robot/microduck/scene_backflip.xml"

# Servo index -> tuck angle at full tuck (factor 1.0). Same joints the roulade
# TUCK_OVERRIDES uses: legs folded, chin tucked. A tucked duck has a much
# smaller pitch inertia, which is exactly how it converts the hand's angular
# impulse into a fast enough spin.
TUCK = {2: -1.15, 3: 1.25, 4: 1.05, 5: -1.0, 6: 1.0, 11: 1.15, 12: -1.25, 13: -1.05}


def run_cell(model, data, vz, w0, tuck_factor, z0, t_hold=0.3, t_launch=0.12,
             duration=2.0, on_step=None):
    mujoco.mj_resetData(model, data)
    dt = model.opt.timestep

    plate_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "plate_free")
    plate_qadr = model.jnt_qposadr[plate_jid]
    plate_vadr = model.jnt_dofadr[plate_jid]
    plate_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "plate_geom")
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    # Robot: free joint at qpos[0:7], 14 servos after it.
    ctrl = np.zeros(model.nu)
    for idx, angle in TUCK.items():
        ctrl[idx] = angle * tuck_factor
    data.ctrl[:] = ctrl
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
    data.qpos[0:3] = [0.0, 0.0, z0 + 0.01 + SPAWN_OFFSET]  # feet on the plate top
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7 : 7 + model.nu] = ctrl

    accum_pitch = 0.0
    apex = 0.0
    landing_speed = 0.0
    airborne_seen = False
    landed = False
    t = 0.0
    while t < duration:
        z, pitch, vz_t, w_t, phase = microduck_mdp.backflip_plate_kinematics(
            *[torch.tensor([v]) for v in (t, t_hold, t_launch, z0, vz, w0)]
        )
        half = float(pitch) * 0.5
        data.qpos[plate_qadr + 0 : plate_qadr + 3] = [0.0, 0.0, float(z)]
        data.qpos[plate_qadr + 3 : plate_qadr + 7] = [
            math.cos(half), 0.0, math.sin(half), 0.0
        ]
        data.qvel[plate_vadr + 0 : plate_vadr + 3] = [0.0, 0.0, float(vz_t)]
        data.qvel[plate_vadr + 3 : plate_vadr + 6] = [0.0, float(w_t), 0.0]

        # Snapshot BEFORE stepping. The contact constraint solved INSIDE
        # mj_step for the step that first makes contact already decelerates
        # the robot, so reading qvel AFTER that step underestimates "peak
        # downward speed at first contact" — the number this script exists to
        # report (module docstring, line 7). Use the pre-step velocity if
        # this turns out to be the first-contact step.
        pre_step_vz = float(data.qvel[2])

        mujoco.mj_step(model, data)
        t += dt

        trunk_z = float(data.qpos[2])
        apex = max(apex, trunk_z)
        # Airborne = no contact between the ROBOT and the floor geom. Excluding
        # plate_geom here is load-bearing, not cosmetic: BACKFLIP_GONE_Z=-3.0
        # teleports the plate into MuJoCo's infinite floor half-space (a
        # type="plane" geom has no lower bound), so the parked plate is
        # permanently "touching" floor from the instant phase goes GONE. An
        # unfiltered floor-contact check makes `touching` always True in GONE,
        # so `not touching` never holds and accum_pitch never accumulates —
        # this was caught by the step-3 sanity check (rot_deg was 0.0 in
        # every cell) and traced with mj_id2name on data.contact before this
        # fix. The bug is in this probe script, not in
        # backflip_plate_kinematics or the scene.
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

    return math.degrees(accum_pitch), landing_speed, apex


def local_z_axis_world(data):
    """World-frame direction of the robot's own local +z axis ("its own up").

    Third column of the body-to-world rotation matrix built from the free
    joint's quaternion (qpos[3:7]).
    """
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, data.qpos[3:7])
    return R.reshape(3, 3)[:, 2]


def check_direction(model, data, vz=3.0, w0=15.0, tuck_factor=1.0, z0=0.15):
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

    rot, land, apex = run_cell(model, data, vz, w0, tuck_factor, z0, on_step=on_step)
    print(f"  cell result: vz={vz} w0={w0} tuck={tuck_factor} z0={z0} "
          f"-> rot_deg={rot:.1f} land_m/s={land:.2f} apex_m={apex:.3f}")
    if not state["reported"]:
        print("  WARNING: never reached 90deg of accumulated rotation — cannot verify direction "
              "with this cell.")
    return rot, land, apex


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


def _float_list(s):
    return tuple(float(x) for x in s.split(","))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z0", type=float, default=0.15)
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

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)

    if args.check_direction:
        print(f"Direction check (z0={args.z0}):")
        check_direction(
            model, data,
            vz=args.check_vz, w0=args.check_w0, tuck_factor=args.check_tuck,
            z0=args.z0,
        )
        return

    print(f"{'vz':>5} {'w0':>6} {'tuck':>5} {'rot_deg':>8} {'land_m/s':>9} {'apex_m':>7}")
    for vz, w0, tuck in itertools.product(args.vz, args.w0, args.tuck):
        rot, land, apex = run_cell(model, data, vz, w0, tuck, args.z0)
        print(f"{vz:5.2f} {w0:6.1f} {tuck:5.2f} {rot:8.1f} {land:9.2f} {apex:7.3f}")


if __name__ == "__main__":
    main()
