"""The footstep contract shared by the footstep-tracking policy and the FootstepNet planner.

A footstep is the planar pose (dx, dy, dθ) of the LANDING foot in the frame of the SUPPORT foot, both frames being
the sole centre projected on the floor, x forward, yaw = the foot's heading. As in FootstepNet
(github.com/rhoban/footstepnet_envs), the planner's action is that pose relative to the NEUTRAL landing pose —
parallel feet at FEET_SPACING beside the support foot — expressed as if the support foot were the right one (a
left support mirrors dy and dθ). `to_support_frame` / `to_neutral_frame` convert between the two.

Values (2026-10-28): feet spacing 95 mm (standing / straight walking), sole 54 x 41 mm, its centre 7 mm ahead of the
*_foot site; neutral is PARALLEL feet (velstand's ~11° toe-out habit is not part of the contract).

Step LIMITS are deliberately not set from any existing policy. Order: the footstep-tracking policy trains on a wide
target range (bounded by the robot's kinematics, widened by curriculum) → its achieved envelope is measured with
scripts/footsteps_* → MAX_* below are frozen to that envelope → only then is FootstepNet trained with them.
Velstand's envelope (+34/-54 mm, ±15 mm, ±10°) and the sprint reference's 78 mm steps are just a known-feasible
floor / the curriculum's starting point.
"""

import json
import math
from pathlib import Path

FEET_SPACING = 0.095  # [m] lateral distance between the sole centres, neutral stance
SOLE_LENGTH = 0.054  # [m]
SOLE_WIDTH = 0.041  # [m]
SOLE_CENTER_IN_SITE = (0.007, 0.0)  # [m] sole centre in the left_foot / right_foot site frame (x fwd, y left)
FOOT_SITES = ("left_foot", "right_foot")

# Support-foot rule = the robot's contact odometry (microduck/odometry, `Odometry::alpha()` in robotd, replicated in
# scripts/infer_policy.py PyOdometry): the anchor is the lowest of the sole points; the support switches to the other
# foot once that foot holds the lowest point for SWITCH_CONFIRM_TICKS control ticks. Same rule in sim → the policy's
# target frame and step timing are reproducible on the robot without contact sensors.
ODOM_ANCHOR_SET = "alpha16"  # the production set
SWITCH_CONFIRM_TICKS = 2
_ANCHOR_SETS_JSON = Path(__file__).resolve().parents[2] / "scripts" / "odom_anchor_sets.json"


def odom_anchor_points(name: str = ODOM_ANCHOR_SET):
    """[left (N, 3), right (N, 3)] sole points in their foot-site frames [m] (lists of lists)."""
    with open(_ANCHOR_SETS_JSON) as f:
        s = json.load(f)["sets"][name]
    return [s["left"], s["right"]]

# Planner limits: None until measured on the trained footstep policy (see module docstring).
MAX_DX_FORWARD: float | None = None  # [m]
MAX_DX_BACKWARD: float | None = None  # [m]
MAX_DY: float | None = None  # [m] around the neutral spacing, both ways
MAX_DTHETA: float | None = None  # [rad]


def _limits():
    if None in (MAX_DX_FORWARD, MAX_DX_BACKWARD, MAX_DY, MAX_DTHETA):
        raise RuntimeError("footstep planner limits are not set yet: measure them on the trained footstep policy")
    return MAX_DX_FORWARD, MAX_DX_BACKWARD, MAX_DY, MAX_DTHETA


def ellipsoid_clip(dx, dy, dtheta):
    """FootstepNet's step clip: scale the step back onto the limit ellipsoid if it lies outside (floats or arrays)."""
    fwd, bwd, max_dy, max_dth = _limits()
    norm = math.sqrt((dx / (fwd if dx >= 0 else bwd)) ** 2 + (dy / max_dy) ** 2 + (dtheta / max_dth) ** 2)
    k = 1.0 / norm if norm > 1.0 else 1.0
    return dx * k, dy * k, dtheta * k


def to_support_frame(step, support_is_left: bool):
    """Neutral-relative planner step (right-support convention) → landing pose in the support foot frame."""
    dx, dy, dth = step
    if support_is_left:
        dy, dth = -dy, -dth
    return dx, dy + (-FEET_SPACING if support_is_left else FEET_SPACING), dth


def to_neutral_frame(pose, support_is_left: bool):
    """Landing pose in the support foot frame → neutral-relative planner step (right-support convention)."""
    dx, dy, dth = pose
    dy -= -FEET_SPACING if support_is_left else FEET_SPACING
    if support_is_left:
        dy, dth = -dy, -dth
    return dx, dy, dth


def footstepnet_options() -> dict:
    """Constructor options of gym_footsteps_planning's FootstepsPlanningEnv for microduck (≈ Sigmaban x 0.4-0.6)."""
    fwd, bwd, max_dy, max_dth = _limits()
    return {
        "max_dx_forward": fwd,
        "max_dx_backward": bwd,
        "max_dy": max_dy,
        "max_dtheta": max_dth,
        "tolerance_distance": 0.02,  # ~1/3 foot, like Sigmaban's 5 cm on a 14 cm foot
        "tolerance_angle": math.radians(5),
        "obstacle_max_radius": 0.12,  # the kick ball is 70 mm; leaves room for bigger obstacles
        "foot_length": SOLE_LENGTH,
        "foot_width": SOLE_WIDTH,
        "feet_spacing": FEET_SPACING,
    }
