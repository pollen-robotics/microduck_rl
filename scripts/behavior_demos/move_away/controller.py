"""Pure state machine + velocity-command law for the move-away demo.

Deliberately free of MuJoCo, ONNX and numpy so the whole decision layer can be
unit-tested on CPU without a policy file or a physics step (see
``tests/test_move_away_demo.py``).

The behaviour is four states:

    IDLE     person not yet close enough / not visible -> stand still
    RETREAT  walk straight backward, holding the initial heading closed-loop
    TURN     keep walking backward while tracking a heading offset by TURN_TARGET
    CLEAR    keep walking backward on the new heading
    DONE     stop

Everything the demo knows about the gait was MEASURED in this repo against
``scene_move_away.xml`` + the stock walking policy, driving the policy with the
``imu_ang_vel`` sensor (see ``README.md`` in this directory for the sweeps).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# --- measured gait constants -------------------------------------------------
# The backward gait does NOT engage continuously: commanding vx > -0.30 makes
# the policy step in place (net displacement < 5 mm over 8 s). Backward walking
# engages from about vx = -0.32 and stays upright through at least -0.45.
# -0.36 sits comfortably inside that band.
VX_RETREAT: float = -0.36

# Heading is held closed-loop in EVERY walking state. Left unopposed the gait
# drifts: at vx = -0.36 with wz = 0 the robot ends 8 s at yaw = -55 deg. With
# this controller the same command holds the heading to within ~6 deg.
# MEASURED: the wz sign is NORMAL (positive wz -> positive/left yaw rate) when
# the policy is fed the real `imu_ang_vel` gyro.
YAW_KP: float = 2.0
WZ_MAX: float = 0.8

# Turn target and the tolerance that counts as "arrived". The heading is tracked
# as an absolute setpoint rather than cut early: with a real gyro the closed
# loop converges (+90 deg commanded -> +84 deg at 12 s, +87 deg at 20 s) instead
# of overshooting, so no early cut is needed.
TURN_TARGET: float = math.radians(90.0)
TURN_TOL: float = math.radians(15.0)

# Command low-pass time constant. 0.25 s is too slow to start the gait.
CMD_TAU: float = 0.08

# Timings.
RETREAT_HOLD: float = 5.0
CLEAR_HOLD: float = 2.5
TURN_MAX: float = 8.0

# Trigger distance. The person is scripted to start at 1.60 m, so the duck
# reacts with roughly 1.7 s of margin.
RETREAT_D: float = 1.15

STATES = ("IDLE", "RETREAT", "TURN", "CLEAR", "DONE")


def wrap_angle(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


@dataclass
class MoveAwayController:
    """State machine producing a ``(vx, vy, wz)`` twist command.

    The controller is driven by :meth:`update`, which takes the measured
    absolute trunk heading (rad, world frame), the planar distance to the
    person (m) and whether the person is currently visible.
    """

    ctrl_hz: float = 50.0
    retreat_distance: float = RETREAT_D
    retreat_hold: float = RETREAT_HOLD
    clear_hold: float = CLEAR_HOLD
    turn_max: float = TURN_MAX
    turn_target: float = TURN_TARGET
    turn_tol: float = TURN_TOL
    vx_retreat: float = VX_RETREAT
    yaw_kp: float = YAW_KP
    wz_max: float = WZ_MAX
    cmd_tau: float = CMD_TAU
    turn_sign: float = 1.0

    state: str = "IDLE"
    state_t: float = 0.0
    yaw_ref: float | None = None
    turned: float = 0.0
    command: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    @property
    def dt(self) -> float:
        return 1.0 / self.ctrl_hz

    @property
    def heading_setpoint(self) -> float:
        """Absolute heading the controller is currently steering towards."""
        if self.yaw_ref is None:
            return 0.0
        if self.state in ("TURN", "CLEAR"):
            return self.yaw_ref + self.turn_sign * self.turn_target
        return self.yaw_ref

    def _transition(self, distance: float, visible: bool) -> None:
        if self.state == "IDLE":
            # Detection is line-of-sight AND range, never range alone.
            if visible and distance < self.retreat_distance:
                self.state = "RETREAT"
        elif self.state == "RETREAT":
            if self.state_t >= self.retreat_hold:
                self.state = "TURN"
        elif self.state == "TURN":
            reached = abs(abs(self.turned) - self.turn_target) <= self.turn_tol
            if reached or self.state_t >= self.turn_max:
                self.state = "CLEAR"
        elif self.state == "CLEAR" and self.state_t >= self.clear_hold:
            self.state = "DONE"

    def update(self, yaw: float, distance: float, visible: bool) -> tuple[float, float, float]:
        """Advance one control tick and return the filtered ``(vx, vy, wz)``."""
        if self.yaw_ref is None:
            self.yaw_ref = yaw
        self.turned = wrap_angle(yaw - self.yaw_ref)

        self.state_t += self.dt
        previous = self.state
        self._transition(distance, visible)
        if self.state != previous:
            self.state_t = 0.0

        if self.state in ("RETREAT", "TURN", "CLEAR"):
            error = wrap_angle(self.heading_setpoint - yaw)
            target = [
                self.vx_retreat,
                0.0,
                clamp(self.yaw_kp * error, -self.wz_max, self.wz_max),
            ]
        else:
            target = [0.0, 0.0, 0.0]

        alpha = self.dt / self.cmd_tau
        for i in range(3):
            self.command[i] += alpha * (target[i] - self.command[i])
        return tuple(self.command)  # type: ignore[return-value]
