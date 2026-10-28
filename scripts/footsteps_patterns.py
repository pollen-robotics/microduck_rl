"""Printable footstep patterns: FootstepNet plans, drawn 1:1 on A4, with the file robotctl walks.

    uv run scripts/footsteps_patterns.py --planner footstepnet/footstepnet_5cm.onnx [--out footstep_patterns]

For each pattern: FootstepNet plans from the start stance (feet parallel, FEET_SPACING apart, right foot support)
to a goal, plus one closing step that squares the feet up. Writes `<out>/footstep_patterns.pdf` (one A4 page per
pattern, PRINT AT 100 % / actual size — the 100 mm bar checks it) and `<out>/<name>.json`.

On the robot (microduck branch `footsteps`): stand it on the two dashed START soles, then
`robotctl robot footsteps <name>.json`. Poses are sole centres (7 mm ahead of the foot sites); each printed sole has
a cross at its centre, so a landing's miss can be read off the paper, and `journalctl -u robotd | grep footstep`
has the robot's own estimate. The JSON is in the start stance frame (origin midway between the start soles,
x forward, y left), which is how robotd anchors the pattern where the robot stands.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from mjlab_microduck.footstep_planner import FootstepPlanner, _frame, _pose
from mjlab_microduck.footsteps import FEET_SPACING, SOLE_LENGTH, SOLE_WIDTH

# name → (goal of the stance centre: x [m], y [m], yaw [deg]; the foot that lands on it)
PATTERNS = {
    "forward_15cm": ((0.15, 0.0, 0.0), "left"),
    "backward_10cm": ((-0.10, 0.0, 0.0), "left"),
    "turn_left_90": ((0.0, 0.0, 90.0), "left"),
    "turn_right_90": ((0.0, 0.0, -90.0), "right"),
    "arc_left_45": ((0.10, 0.04, 45.0), "left"),
    "side_left_4cm": ((0.0, 0.04, 0.0), "left"),
}
A4 = (210.0, 297.0)  # mm, portrait
MARGIN, HEADER = 12.0, 30.0  # mm
FOOT_RGB = {"left": (0.1, 0.45, 0.9), "right": (0.95, 0.5, 0.05)}
SIDE = {"left": 1.0, "right": -1.0}


def compose(a, b):
    return _pose(_frame(*a) @ _frame(*b))


def plan_pattern(planner, goal_centre, goal_foot):
    """Start stance → goal, then the other foot squares up. Returns (start soles, steps, reached)."""
    half = FEET_SPACING / 2
    start = {"left": np.array([0.0, half, 0.0]), "right": np.array([0.0, -half, 0.0])}
    goal = compose(goal_centre, (0.0, SIDE[goal_foot] * half, 0.0))
    plan, reached = planner.plan(start["right"], "right", goal, goal_foot, None, max_steps=50)
    steps = [(pose, foot) for pose, foot in plan]
    last_pose, last_foot = steps[-1] if steps else (start["right"], "right")
    n = planner.neutral(last_pose, last_foot)
    other = "left" if last_foot == "right" else "right"
    steps.append((np.array([n[0, 2], n[1, 2], math.atan2(n[1, 0], n[0, 0])]), other))
    return start, steps, reached


def corners(pose):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + c * a - s * b, y + s * a + c * b)
            for a, b in ((SOLE_LENGTH / 2, SOLE_WIDTH / 2), (-SOLE_LENGTH / 2, SOLE_WIDTH / 2),
                         (-SOLE_LENGTH / 2, -SOLE_WIDTH / 2), (SOLE_LENGTH / 2, -SOLE_WIDTH / 2))]


def draw_page(pdf, name, goal, start, steps, reached):
    fig = plt.figure(figsize=(A4[0] / 25.4, A4[1] / 25.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, A4[0])
    ax.set_ylim(0, A4[1])
    ax.set_aspect("equal")
    ax.axis("off")

    # pattern frame (x forward, y left, metres) → page (mm): forward is up the page
    poses = [p for p in start.values()] + [p for p, _ in steps]
    pts = np.array([c for p in poses for c in corners(p)]) * 1000
    u_lo, u_hi = -pts[:, 1].max(), -pts[:, 1].min()
    v_lo, v_hi = pts[:, 0].min(), pts[:, 0].max()
    area = (MARGIN, MARGIN + 18, A4[0] - MARGIN, A4[1] - MARGIN - HEADER)  # u0, v0, u1, v1 (bar row at the bottom)
    if u_hi - u_lo > area[2] - area[0] or v_hi - v_lo > area[3] - area[1]:
        raise SystemExit(f"{name}: {u_hi - u_lo:.0f} x {v_hi - v_lo:.0f} mm does not fit on A4")
    du = (area[0] + area[2]) / 2 - (u_lo + u_hi) / 2
    dv = (area[1] + area[3]) / 2 - (v_lo + v_hi) / 2

    def page(x, y):
        return du - y * 1000, dv + x * 1000

    def sole(pose, edge, lw, ls="-", toe=None, label=None):
        cs = [page(*c) for c in corners(pose)]
        ax.add_patch(patches.Polygon(cs, closed=True, fill=False, edgecolor=edge, lw=lw, ls=ls))
        if toe:
            ax.plot([cs[3][0], cs[0][0]], [cs[3][1], cs[0][1]], color=toe, lw=lw * 1.8, solid_capstyle="butt")
        cu, cv = page(pose[0], pose[1])
        ax.plot([cu - 3, cu + 3], [cv, cv], color=edge, lw=0.4)
        ax.plot([cu, cu], [cv - 3, cv + 3], color=edge, lw=0.4)
        if label:  # toward the heel, clear of the centre cross
            h = math.degrees(pose[2])
            lu, lv = page(*compose(pose, (-0.015, 0.0, 0.0))[:2])
            ax.text(lu, lv, label, ha="center", va="center", fontsize=7.5, color=edge, rotation=h,
                    rotation_mode="anchor", bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.7))

    for foot, pose in start.items():
        sole(pose, "black", 1.2, ls=(0, (3, 2)), label=f"START {foot[0].upper()}")
    for i, (pose, foot) in enumerate(steps, start=1):
        sole(pose, FOOT_RGB[foot], 0.9, toe=(0.85, 0.1, 0.1), label=f"{i} {foot[0].upper()}")

    gx, gy, gyaw = goal[0]
    ax.text(MARGIN, A4[1] - MARGIN - 6, f"microduck footsteps — {name}", fontsize=13, weight="bold")
    ax.text(MARGIN, A4[1] - MARGIN - 13,
            f"FootstepNet plan to ({gx * 100:+.0f} cm, {gy * 100:+.0f} cm, {gyaw:+.0f}°), {goal[1]} foot — "
            f"{len(steps)} steps{'' if reached else ' (goal NOT reached in 50)'}", fontsize=8)
    ax.text(MARGIN, A4[1] - MARGIN - 19,
            "Stand the robot on the dashed START soles, facing up the page, then:", fontsize=8)
    ax.text(MARGIN, A4[1] - MARGIN - 25, f"robotctl robot footsteps {name}.json", fontsize=9, family="monospace")
    # scale check
    ax.plot([MARGIN, MARGIN + 100], [MARGIN + 6, MARGIN + 6], color="black", lw=1.5)
    for u in (MARGIN, MARGIN + 100):
        ax.plot([u, u], [MARGIN + 3, MARGIN + 9], color="black", lw=1.0)
    ax.text(MARGIN + 104, MARGIN + 4.5, "100 mm — print at 100 % (actual size); measure this bar", fontsize=7)
    ax.text(MARGIN, MARGIN + 11, "Cross = sole centre (7 mm ahead of the foot site). Red edge = toe.", fontsize=7)
    pdf.savefig(fig)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", required=True, help="FootstepNet ONNX (footstepnet/export_onnx.py)")
    ap.add_argument("--out", default="footstep_patterns")
    ap.add_argument("--png", action="store_true", help="also write a PNG preview per page")
    A = ap.parse_args()
    out = Path(A.out)
    out.mkdir(parents=True, exist_ok=True)
    planner = FootstepPlanner(A.planner)
    with PdfPages(out / "footstep_patterns.pdf") as pdf:
        for name, goal in PATTERNS.items():
            (gx, gy, gyaw), goal_foot = goal
            start, steps, reached = plan_pattern(planner, (gx, gy, math.radians(gyaw)), goal_foot)
            fig = draw_page(pdf, name, goal, start, steps, reached)
            if A.png:
                fig.savefig(out / f"{name}.png", dpi=60)
            plt.close(fig)
            doc = {
                "name": name,
                "planner": Path(A.planner).name,
                "frame": "start stance: origin midway between the start sole centres, x forward, y left; m, rad",
                "start": [{"foot": f, "x": p[0], "y": p[1], "yaw": p[2]} for f, p in start.items()],
                "steps": [{"foot": f, "x": float(p[0]), "y": float(p[1]), "yaw": float(p[2])} for p, f in steps],
            }
            (out / f"{name}.json").write_text(json.dumps(doc, indent=1))
            print(f"{name:15s} {len(steps):2d} steps  reached={reached}")
    print(f"→ {out / 'footstep_patterns.pdf'} (print at 100 %) + {len(PATTERNS)} JSON")


if __name__ == "__main__":
    main()
