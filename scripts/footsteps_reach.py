"""Kinematic upper bound of the footstep (dx, dy, dθ) microduck can place — the footstep policy's training range.

  uv run python scripts/footsteps_reach.py [--samples 400000] [--z-band -0.015 0.005] [--flat-deg 5]

Pure kinematics on the plain (no-backlash) model, trunk level: each leg's joints are sampled uniformly within
--limit-factor x their range; a configuration is kept if its sole is flat (tilt < --flat-deg) and lies within
--z-band of its HOME depth below the trunk (walking height). Left/right samples whose soles sit at the same depth
(±2 mm) are paired; pairs whose sole rectangles overlap are rejected. The footstep is the left sole's planar pose
in the right sole's frame (the right-support convention of mjlab_microduck.footsteps; left support is the mirror).
Ignores dynamics and leg-leg collisions above the soles: an upper bound, not a promise.
"""

import argparse

import mujoco
import numpy as np

from mjlab_microduck import footsteps as F
from mjlab_microduck.robot.microduck_constants import HOME_FRAME, MICRODUCK_WALK_ROBOT_CFG

ap = argparse.ArgumentParser()
ap.add_argument("--samples", type=int, default=400_000, help="random configurations per leg")
ap.add_argument("--z-band", type=float, nargs=2, default=(-0.015, 0.005), help="sole depth vs HOME [m] (- = crouched)")
ap.add_argument("--flat-deg", type=float, default=5.0)
ap.add_argument("--limit-factor", type=float, default=0.9)
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
rng = np.random.default_rng(A.seed)

m = MICRODUCK_WALK_ROBOT_CFG.spec_fn().compile()
d = mujoco.MjData(m)
legs = {"left": [f"left_{j}" for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")],
        "right": [f"right_{j}" for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")]}
site = {s: m.site(n).id for s, n in zip(("left", "right"), F.FOOT_SITES)}


def set_home():
    mujoco.mj_resetData(m, d)
    d.qpos[3:7] = (1, 0, 0, 0)  # trunk level at the origin
    d.qpos[:3] = 0
    for name, q in HOME_FRAME.joint_pos.items():
        if name in [m.joint(i).name for i in range(m.njnt)]:
            d.qpos[m.jnt_qposadr[m.joint(name).id]] = q


def sole(side):
    """Sole centre position + rotation in the trunk frame."""
    R = d.site_xmat[site[side]].reshape(3, 3)
    return d.site_xpos[site[side]] + R @ np.array([*F.SOLE_CENTER_IN_SITE, 0.0]), R


set_home(); mujoco.mj_kinematics(m, d)
home_z = {s: sole(s)[0][2] for s in legs}
print(f"HOME sole depth below trunk origin: L {home_z['left']:.4f}  R {home_z['right']:.4f} m")

out = {}
for side, joints in legs.items():
    adr = [m.jnt_qposadr[m.joint(j).id] for j in joints]
    lo, hi = m.jnt_range[[m.joint(j).id for j in joints]].T
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * A.limit_factor
    Q = mid + half * rng.uniform(-1, 1, (A.samples, len(joints)))
    keep = []
    set_home()
    for q in Q:
        d.qpos[adr] = q
        mujoco.mj_kinematics(m, d)
        p, R = sole(side)
        if R[2, 2] < np.cos(np.radians(A.flat_deg)):
            continue
        dz = p[2] - home_z[side]
        if A.z_band[0] <= dz <= A.z_band[1]:
            keep.append((p[0], p[1], p[2], np.arctan2(R[1, 0], R[0, 0])))
    out[side] = np.array(keep)
    print(f"{side}: {len(keep)}/{A.samples} flat at walking height")

Lf, Rf = out["left"], out["right"]
# pair by depth: sort right by z, take each left sample with up to K random right partners within ±2 mm
order = np.argsort(Rf[:, 2]); Rz = Rf[order, 2]
steps = []
K = 8
for lx, ly, lz, lth in Lf:
    a, b = np.searchsorted(Rz, lz - 0.002), np.searchsorted(Rz, lz + 0.002)
    if b <= a:
        continue
    for rx, ry, _, rth in Rf[order[rng.integers(a, b, min(K, b - a))]]:
        c, s = np.cos(rth), np.sin(rth)
        ddx, ddy = lx - rx, ly - ry
        steps.append((c * ddx + s * ddy, -s * ddx + c * ddy, np.arctan2(np.sin(lth - rth), np.cos(lth - rth))))
S = np.array(steps)


def overlap(st):
    """SAT test: does the left sole rectangle (pose st in the right sole frame) overlap the right sole at the origin?"""
    hl, hw = F.SOLE_LENGTH / 2, F.SOLE_WIDTH / 2
    corners = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    x, y, th = st.T
    c, s = np.cos(th), np.sin(th)
    P = np.stack([x[:, None] + c[:, None] * corners[:, 0] - s[:, None] * corners[:, 1],
                  y[:, None] + s[:, None] * corners[:, 0] + c[:, None] * corners[:, 1]], -1)  # (n, 4, 2)
    sep = (P[..., 0].min(1) > hl) | (P[..., 0].max(1) < -hl) | (P[..., 1].min(1) > hw) | (P[..., 1].max(1) < -hw)
    for ax in (np.stack([c, s], -1), np.stack([-s, c], -1)):  # the left sole's own axes
        proj = (P * ax[:, None]).sum(-1)
        rc = (corners[None] * ax[:, None]).sum(-1)  # right sole corners projected (right sole = origin, axis-aligned)
        sep |= (proj.min(1) > rc.max(1)) | (proj.max(1) < rc.min(1))
    return ~sep


S = S[~overlap(S)]
N = np.array(F.to_neutral_frame((S[:, 0], S[:, 1], S[:, 2]), support_is_left=False)).T  # right support: no mirror
print(f"\n{len(S)} feasible footsteps (left lands, right support), relative to NEUTRAL (parallel, {1000 * F.FEET_SPACING:.0f} mm):")
mm = np.array([1000, 1000, np.degrees(1)])
for q in (0.5, 99.5):
    print(f"  p{q:<5}", "  ".join(f"{n} {v:+7.1f}" for n, v in zip(("dx mm", "dy mm", "dθ deg"), np.percentile(N, q, 0) * mm)))
near = lambda i, tol: np.all([np.abs(N[:, j]) < tol[j] for j in range(3) if j != i], 0)
tol = (0.010, 0.010, np.radians(5))
for i, n in enumerate(("dx", "dy", "dθ")):
    v = N[near(i, tol), i] * mm[i]
    print(f"  pure {n:3s} (others ≈ neutral): {v.min():+7.1f} … {v.max():+7.1f} {'deg' if i == 2 else 'mm'}  (n={len(v)})")
