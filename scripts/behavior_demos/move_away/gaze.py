"""Isolated kinematic gaze layer for the move-away demo.

Why this is kinematic and not physical
--------------------------------------
The head is a large fraction of microduck's mass. Driving it physically while
the stock walking policy is running makes the robot fall: that policy was never
trained to compensate an externally imposed head trajectory. So this layer:

1. leaves locomotion physics and policy inference untouched in the primary
   ``MjData``;
2. poses head yaw/pitch in a SEPARATE ``MjData`` copy used only for perception
   and rendering;
3. never feeds that pose back into the locomotion dynamics.

That mirrors the gaze/locomotion split a real robot would use, and it makes
clear that this demo claims NO physical head-control stability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Keep the commanded joint angles this far from their hard limits.
HEAD_YAW_MARGIN: float = math.radians(3.0)
HEAD_PITCH_MARGIN: float = math.radians(5.0)


def camera_axes(cam_xmat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(forward, right, up)`` for a MuJoCo camera rotation matrix.

    MuJoCo cameras look down their local ``-Z`` with local ``+Y`` as image-up.
    """
    rot = np.asarray(cam_xmat, dtype=np.float64).reshape(3, 3)
    return -rot[:, 2], rot[:, 0], rot[:, 1]


def bearing_elevation(
    to_target: np.ndarray,
    forward: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
) -> tuple[float, float]:
    """Angles from the optical axis to ``to_target``.

    Returns ``(bearing, elevation)`` in radians, bearing positive to the
    camera's LEFT so it can be added directly to a yaw joint.
    """
    vec = np.asarray(to_target, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    unit = vec / norm if norm > 1e-9 else vec
    fwd = float(np.dot(unit, forward))
    bearing = math.atan2(-float(np.dot(unit, right)), fwd)
    elevation = math.atan2(float(np.dot(unit, up)), fwd)
    return bearing, elevation


def in_frustum(
    to_target: np.ndarray,
    forward: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    tan_half_h: float,
    tan_half_v: float,
) -> bool:
    """True when ``to_target`` falls inside the camera's rectangular frustum."""
    vec = np.asarray(to_target, dtype=np.float64)
    depth = float(np.dot(vec, forward))
    if depth <= 0.0:
        return False
    return (
        abs(float(np.dot(vec, right))) <= depth * tan_half_h
        and abs(float(np.dot(vec, up))) <= depth * tan_half_v
    )


@dataclass
class GazeState:
    """Head yaw/pitch setpoints for the isolated gaze layer.

    ``yaw_limits`` / ``pitch_limits`` are the model's joint ranges; the
    setpoints are kept a margin away from them so the render never shows a
    joint jammed against a hard stop.
    """

    yaw: float
    pitch: float
    yaw_limits: tuple[float, float]
    pitch_limits: tuple[float, float]
    tau: float = 0.02
    dt: float = 0.02

    def step(self, bearing: float, elevation: float) -> tuple[float, float]:
        """Servo the head towards a target at ``(bearing, elevation)``.

        Both angles are measured from the CURRENT optical axis, so this is a
        relative visual-servo correction rather than an absolute pointing
        command.
        """
        yaw_lo, yaw_hi = self.yaw_limits
        pitch_lo, pitch_hi = self.pitch_limits
        desired_yaw = min(
            max(self.yaw + bearing, yaw_lo + HEAD_YAW_MARGIN), yaw_hi - HEAD_YAW_MARGIN
        )
        # A positive head_pitch command points the camera DOWN, hence the minus.
        desired_pitch = min(
            max(self.pitch - elevation, pitch_lo + HEAD_PITCH_MARGIN),
            pitch_hi - HEAD_PITCH_MARGIN,
        )
        alpha = min(1.0, self.dt / self.tau) if self.tau > 0.0 else 1.0
        self.yaw += alpha * (desired_yaw - self.yaw)
        self.pitch += alpha * (desired_pitch - self.pitch)
        return self.yaw, self.pitch
