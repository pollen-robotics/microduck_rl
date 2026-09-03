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
             duration=2.0):
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
        if int(phase) == microduck_mdp.BACKFLIP_PHASE_GONE and not touching:
            airborne_seen = True
            # Backward pitch = +omega_y, not -omega_y. Both the plate's
            # freejoint and the robot's freejoint index angular velocity the
            # same way (world-frame qvel[3:6]), and backflip_plate_kinematics
            # documents its own pitch as "positive = backward". Tracing
            # data.qvel[4] during LAUNCH shows it ramping up positive in lockstep
            # with the plate's positive w_t (contact transfer preserves sign,
            # as it must), so a "-omega_y" bookkeeping sign here just flips the
            # measured rotation negative — a bug in this probe script, caught
            # by the step-3 sanity check (fixed alongside the touching-check
            # bug above; not a bug in backflip_plate_kinematics or the scene).
            accum_pitch += float(data.qvel[4]) * dt
        elif airborne_seen and touching and landing_speed == 0.0:
            landing_speed = abs(float(data.qvel[2]))

    return math.degrees(accum_pitch), landing_speed, apex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z0", type=float, default=0.15)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)

    print(f"{'vz':>5} {'w0':>6} {'tuck':>5} {'rot_deg':>8} {'land_m/s':>9} {'apex_m':>7}")
    for vz, w0, tuck in itertools.product(
        (1.5, 2.0, 2.5, 3.0), (6.0, 9.0, 12.0, 15.0), (0.5, 1.0)
    ):
        rot, land, apex = run_cell(model, data, vz, w0, tuck, args.z0)
        print(f"{vz:5.1f} {w0:6.1f} {tuck:5.2f} {rot:8.1f} {land:9.2f} {apex:7.3f}")


if __name__ == "__main__":
    main()
