#!/usr/bin/env python3
"""Sample the odometry's sole anchor points on the alpha sole mesh.

The robot's contact odometry (microduck/odometry) tracks the trunk from the
lowest of a handful of "sole corners" expressed in each foot's site frame.
This script derives those points from the sole collision mesh of the MuJoCo
model instead of a hand-typed bbox:

  v15      the four legacy corners: ±27.0 x ±20.6 mm on the site's Z = 0 plane
           (the v1.5 sole bbox, carried over as a placeholder). Same on both feet.
  alpha4   the four corners of the flat contact patch of the alpha sole, each
           dropped vertically onto the mesh's bottom surface.
  alpha16  an NX x NY grid over the whole footprint (bevels included), inset a
           little from the outline so every vertical ray hits, each point on
           the mesh's bottom surface.

"On the bottom surface" means: shoot a vertical ray through (x, y) in the site
frame and keep the lowest triangle hit. Left and right soles are mirror images
so each foot gets its own points.

Writes src/mjlab_microduck/odom_anchor_sets.json (consumed by infer.py) and, with
--rust, the `anchors.rs` module of the microduck odometry crate.
"""
import argparse
import json
import os
import sys

import mujoco
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SCENE = os.path.join(REPO, "src/mjlab_microduck/robot/microduck/scene.xml")
DEFAULT_JSON = os.path.join(REPO, "src/mjlab_microduck/odom_anchor_sets.json")

V15_HALF_LEN = 0.0270
V15_HALF_WIDTH = 0.0206


def quat2mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=np.float64))
    return m.reshape(3, 3)


class SoleSurface:
    """The sole collision mesh of one foot, as triangles in the foot-site frame."""

    def __init__(self, model, side):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot")
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
        if sid < 0 or gid < 0:
            raise SystemExit(f"scene lacks {side}_foot site or {side}_foot_collision geom")
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            raise SystemExit(f"{side}_foot_collision is not a mesh geom")
        if model.geom_bodyid[gid] != model.site_bodyid[sid]:
            raise SystemExit(f"{side}_foot site and sole geom are on different bodies")
        mid = model.geom_dataid[gid]
        v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        f0, nf = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        verts = model.mesh_vert[v0:v0 + nv].astype(np.float64)
        faces = model.mesh_face[f0:f0 + nf]
        # Compiled mesh vertices are centred with the centring folded into the
        # geom pose; geom and site are both body-relative, so one hop each.
        in_body = verts @ quat2mat(model.geom_quat[gid]).T + model.geom_pos[gid]
        in_site = (in_body - model.site_pos[sid]) @ quat2mat(model.site_quat[sid])
        self.tri = in_site[faces]                     # (nf, 3, 3)
        self.lo, self.hi = in_site.min(axis=0), in_site.max(axis=0)
        a, b, c = self.tri[:, 0], self.tri[:, 1], self.tri[:, 2]
        self._d = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        self._ok = np.abs(self._d) > 1e-15   # drop triangles seen edge-on from above

    def bottom_z(self, x, y):
        """Lowest mesh surface point on the vertical line through (x, y), or NaN."""
        a, b, c, d = (self.tri[self._ok, 0], self.tri[self._ok, 1],
                      self.tri[self._ok, 2], self._d[self._ok])
        l1 = ((b[:, 1] - c[:, 1]) * (x - c[:, 0]) + (c[:, 0] - b[:, 0]) * (y - c[:, 1])) / d
        l2 = ((c[:, 1] - a[:, 1]) * (x - c[:, 0]) + (a[:, 0] - c[:, 0]) * (y - c[:, 1])) / d
        l3 = 1.0 - l1 - l2
        hit = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
        if not hit.any():
            return float("nan")
        z = l1[hit] * a[hit, 2] + l2[hit] * b[hit, 2] + l3[hit] * c[hit, 2]
        return float(z.min())

    def project(self, xy, pull_in=True):
        """Drop each (x, y) onto the bottom surface. The footprint is a rounded
        rectangle, so a grid over its bbox has points past the outline; those
        are pulled straight towards the footprint centre, 0.25 mm a step, until
        their ray hits — so they end up on the bevel's outer edge."""
        cx, cy = (self.lo[:2] + self.hi[:2]) / 2
        out = []
        for x, y in xy:
            z = self.bottom_z(x, y)
            while pull_in and np.isnan(z):
                dx, dy = cx - x, cy - y
                norm = np.hypot(dx, dy)
                if norm < 1e-6:
                    break
                x, y = x + 0.00025 * dx / norm, y + 0.00025 * dy / norm
                z = self.bottom_z(x, y)
            out.append([float(x), float(y), z])
        return out

    def flat_patch(self, tol, samples=60):
        """Bbox (lo_xy, hi_xy) of the flat contact patch: where the bottom lies
        within `tol` of a plane fitted to the central half of the footprint.
        The sole bottom is a plane with rounded bevels, so this is where the
        bevels start."""
        xs = np.linspace(self.lo[0], self.hi[0], samples)
        ys = np.linspace(self.lo[1], self.hi[1], samples)
        X, Y = np.meshgrid(xs, ys)
        Z = np.array([[self.bottom_z(x, y) for x in xs] for y in ys])
        cx = (self.lo[0] + self.hi[0]) / 2
        cy = (self.lo[1] + self.hi[1]) / 2
        central = ((np.abs(X - cx) < (self.hi[0] - self.lo[0]) / 4)
                   & (np.abs(Y - cy) < (self.hi[1] - self.lo[1]) / 4) & ~np.isnan(Z))
        A = np.c_[X[central], Y[central], np.ones(central.sum())]
        coef, *_ = np.linalg.lstsq(A, Z[central], rcond=None)
        plane = coef[0] * X + coef[1] * Y + coef[2]
        flat = (np.abs(Z - plane) < tol) & ~np.isnan(Z)
        self.plane_tilt_deg = (np.degrees(np.arctan(coef[0])), np.degrees(np.arctan(coef[1])))
        self.plane = coef
        return (np.array([X[flat].min(), Y[flat].min()]),
                np.array([X[flat].max(), Y[flat].max()]))

    def flat_corners(self, tol):
        """The four corners of the flat patch, on the mesh. The patch has
        rounded corners, so each bbox corner is walked diagonally towards the
        centre until the surface there is within `tol` of the bottom plane."""
        lo, hi = self.flat_patch(tol)
        cx, cy = (lo + hi) / 2
        out = []
        for x, y in ((hi[0], hi[1]), (hi[0], lo[1]), (lo[0], lo[1]), (lo[0], hi[1])):
            z = self.bottom_z(x, y)
            while np.isnan(z) or abs(z - (self.plane[0] * x + self.plane[1] * y + self.plane[2])) > tol:
                dx, dy = cx - x, cy - y
                norm = np.hypot(dx, dy)
                x, y = x + 0.00025 * dx / norm, y + 0.00025 * dy / norm
                z = self.bottom_z(x, y)
            out.append([float(x), float(y), z])
        return lo, hi, out


def grid(lo, hi, nx, ny):
    xs = np.linspace(lo[0], hi[0], nx)
    ys = np.linspace(lo[1], hi[1], ny)
    return [(x, y) for y in ys for x in xs]


def rust_module(sets, scene, argv):
    def pts(points):
        rows = ",\n".join(f"        [{x:.6f}, {y:.6f}, {z:.6f}]" for x, y, z in points)
        return f"&[\n{rows},\n    ]"

    out = [
        "//! Sole anchor points for the contact odometry, in each foot's site frame",
        "//! (X front/back, Y left/right, Z up), metres.",
        "//!",
        "//! GENERATED by microduck_rl/scripts/odom_anchor_points.py — do not edit:",
        f"//!   {' '.join(argv)}",
        f"//! from {os.path.relpath(scene, REPO)}.",
        "",
        "/// A named set of candidate contact points, one list per foot.",
        "pub struct AnchorSet {",
        "    pub name: &'static str,",
        "    pub left: &'static [[f64; 3]],",
        "    pub right: &'static [[f64; 3]],",
        "}",
        "",
        "impl AnchorSet {",
        "    /// Points for foot 0 (left) or 1 (right).",
        "    pub fn foot(&self, foot: usize) -> &'static [[f64; 3]] {",
        "        if foot == 0 {",
        "            self.left",
        "        } else {",
        "            self.right",
        "        }",
        "    }",
        "}",
        "",
    ]
    for name, (doc, feet) in sets.items():
        out += [f"/// {doc}", f"pub const {name.upper()}: AnchorSet = AnchorSet {{",
                f'    name: "{name}",',
                f"    left: {pts(feet['left'])},",
                f"    right: {pts(feet['right'])},",
                "};", ""]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default=DEFAULT_SCENE)
    p.add_argument("--grid", type=int, nargs=2, default=(4, 4), metavar=("NX", "NY"),
                   help="grid size of the dense set (default 4 4 = 16 points)")
    p.add_argument("--inset", type=float, default=1.5,
                   help="mm to pull the dense grid in from the footprint outline (default 1.5)")
    p.add_argument("--flat-tol", type=float, default=0.4,
                   help="mm from the fitted bottom plane still counted as flat patch (default 0.4)")
    p.add_argument("--json", default=DEFAULT_JSON, help="where to write the sets for infer_policy")
    p.add_argument("--rust", default=None, help="also write the odometry crate's anchors.rs here")
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    nx, ny = args.grid
    inset, tol = args.inset / 1000, args.flat_tol / 1000
    v15 = [[V15_HALF_LEN, V15_HALF_WIDTH, 0.0], [V15_HALF_LEN, -V15_HALF_WIDTH, 0.0],
           [-V15_HALF_LEN, -V15_HALF_WIDTH, 0.0], [-V15_HALF_LEN, V15_HALF_WIDTH, 0.0]]
    sets = {
        "v15": ("Legacy: the v1.5 sole bbox on the site's Z = 0 plane, both feet alike.",
                {"left": v15, "right": v15}),
        "alpha4": ("The four corners of the alpha sole's flat contact patch, on the mesh.", {}),
        f"alpha{nx * ny}": (f"A {nx}x{ny} grid over the whole alpha footprint (bevels included), on the mesh.", {}),
    }
    dense = f"alpha{nx * ny}"
    for side in ("left", "right"):
        sole = SoleSurface(model, side)
        flo, fhi, corners = sole.flat_corners(tol)
        sets["alpha4"][1][side] = corners
        sets[dense][1][side] = sole.project(grid(sole.lo[:2] + inset, sole.hi[:2] - inset, nx, ny))
        print(f"{side} sole in site frame (mm): bbox X [{sole.lo[0]*1e3:+.1f}, {sole.hi[0]*1e3:+.1f}] "
              f"Y [{sole.lo[1]*1e3:+.1f}, {sole.hi[1]*1e3:+.1f}] Z [{sole.lo[2]*1e3:+.1f}, {sole.hi[2]*1e3:+.1f}]")
        print(f"  bottom plane tilt: {sole.plane_tilt_deg[0]:+.1f} deg along X, {sole.plane_tilt_deg[1]:+.1f} deg along Y")
        print(f"  flat patch: X [{flo[0]*1e3:+.1f}, {fhi[0]*1e3:+.1f}]  Y [{flo[1]*1e3:+.1f}, {fhi[1]*1e3:+.1f}]")
        for name in ("alpha4", dense):
            zs = [pt[2] * 1e3 for pt in sets[name][1][side]]
            if any(np.isnan(zs)):
                raise SystemExit(f"{name}/{side}: a ray missed the mesh even after pulling in")
            print(f"  {name}: {len(zs)} points, Z in [{min(zs):+.2f}, {max(zs):+.2f}] mm")

    payload = {
        "generated_by": "scripts/odom_anchor_points.py " + " ".join(sys.argv[1:]),
        "scene": os.path.relpath(args.scene, REPO),
        "units": "metres, foot-site frame",
        "sets": {name: feet for name, (_doc, feet) in sets.items()},
    }
    with open(args.json, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"wrote {os.path.relpath(args.json)}")
    if args.rust:
        with open(args.rust, "w") as f:
            f.write(rust_module(sets, args.scene, ["odom_anchor_points.py"] + sys.argv[1:]))
        print(f"wrote {args.rust}")


if __name__ == "__main__":
    main()
