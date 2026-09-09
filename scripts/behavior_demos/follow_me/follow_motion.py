"""Leader choreography, world-space footprint trail and the follow controller.

This module is deliberately free of MuJoCo and ONNX imports below the
``animate_person`` helper so the behavior logic can be unit-tested on CPU
without a model or a policy file (see ``tests/test_follow_me_demo.py``).

The behavior is a thin layer ON TOP of the stock ``alpha_walking`` policy: it
only chooses the 3-DoF twist command ``(vx, vy, wz)`` that is written into the
policy's command slots. No locomotion network is trained, replaced or filtered
here.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass

import numpy as np

# Spatial gap between the leader and the queued footprint the duck pursues.
# This is a PATH LENGTH in metres, not a time delay: the duck walks to the
# place the leader stood 0.65 m of walking ago, so corners stay in the queue.
TRAIL_DISTANCE = 0.65

# Policies are trained and deployed at 50 Hz.
CTRL_HZ = 50.0
CMD_TAU = 1.0 / CTRL_HZ

# Leader route timing (seconds). The phase boundaries are shared with the HUD
# timeline so the video and the metrics cannot drift apart.
READY_END = 2.0
FORWARD_END = 7.0
LEFT_TURN_END = 15.0
STOP_END = 18.0
RIGHT_ARC_END = 26.0
RIGHT_EXIT_END = 35.0
BACKWARD_END = 41.0
DEMO_SECONDS = 44.0

# Leader speeds (m/s) and turn geometry.
FORWARD_SPEED = 0.055
TURN_SPEED = 0.120
TURN_DURATION = 8.0
EXIT_SPEED = 0.080
BACK_SPEED = 0.080

# Measured asymmetric turning authority of the stock walking policy. A single
# mirrored command does NOT produce mirrored body motion, so left and right
# corrections use separately measured magnitude bands.
YAW_DEADBAND_RAD = math.radians(3.0)
LEFT_WZ_MIN, LEFT_WZ_MAX = 0.60, 1.00
RIGHT_WZ_MIN, RIGHT_WZ_MAX = 0.18, 0.32
YAW_GAIN = 1.25
TURN_VX = 0.24

# The stock policy has a sharp gait-onset threshold, so tiny continuous
# velocity corrections are counterproductive: phases command discrete speeds.
FORWARD_CMD = (0.24, 0.0, 0.0)
BACKWARD_CMD = (-0.32, 0.0, 0.20)
IDLE_CMD = (0.0, 0.0, 0.0)

TURN_PHASES = ("LEFT TURN", "RIGHT TURN")


@dataclass(frozen=True)
class PersonState:
    """Scripted leader pose at one instant."""

    phase: str
    pos: np.ndarray
    yaw: float
    velocity: np.ndarray
    yaw_rate: float
    moving: bool
    progress: float


@dataclass(frozen=True)
class TrailState:
    """A point recovered from the leader's recorded world-space path."""

    phase: str
    pos: np.ndarray
    yaw: float
    path_s: float
    leader_path_s: float
    moving: bool


def wrap(angle: float) -> float:
    """Wrap an angle to the half-open interval [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _forward(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)


def person_trajectory(t: float) -> PersonState:
    """Script a true left arc followed by a true right arc.

    With the actor initially facing +X, positive world yaw is its LEFT turn and
    negative world yaw is its RIGHT turn. The two curves therefore have
    opposite signed angular velocities and trace an S-shaped route, rather than
    two same-signed curves that merely look different on camera.
    """
    start = np.array([0.65, 0.0], dtype=np.float64)

    if t < READY_END:
        return PersonState("READY", start, 0.0, np.zeros(2), 0.0, False,
                           t / READY_END)
    if t < FORWARD_END:
        u = t - READY_END
        pos = start + np.array([FORWARD_SPEED * u, 0.0])
        return PersonState("FORWARD", pos, 0.0,
                           np.array([FORWARD_SPEED, 0.0]), 0.0, True,
                           u / (FORWARD_END - READY_END))

    left_start = start + np.array([FORWARD_SPEED * (FORWARD_END - READY_END), 0.0])
    omega_left = (math.pi / 2.0) / TURN_DURATION
    radius = TURN_SPEED / omega_left
    if t < LEFT_TURN_END:
        u = t - FORWARD_END
        yaw = omega_left * u
        pos = left_start + np.array([
            radius * math.sin(yaw),
            radius * (1.0 - math.cos(yaw)),
        ])
        return PersonState("LEFT TURN", pos, yaw, TURN_SPEED * _forward(yaw),
                           omega_left, True, u / TURN_DURATION)

    left_end = left_start + np.array([radius, radius])
    left_yaw = math.pi / 2.0
    if t < STOP_END:
        return PersonState("STOP", left_end, left_yaw, np.zeros(2), 0.0, False,
                           (t - LEFT_TURN_END) / (STOP_END - LEFT_TURN_END))

    # Turn right from +Y back to +X: the opposite signed curvature.
    omega_right = -(math.pi / 2.0) / TURN_DURATION
    if t < RIGHT_ARC_END:
        u = t - STOP_END
        yaw = left_yaw + omega_right * u
        pos = left_end + np.array([
            (TURN_SPEED / omega_right) * (math.sin(yaw) - 1.0),
            -(TURN_SPEED / omega_right) * math.cos(yaw),
        ])
        return PersonState("RIGHT TURN", pos, yaw, TURN_SPEED * _forward(yaw),
                           omega_right, True, u / TURN_DURATION)

    right_end = left_end + np.array([radius, radius])
    if t < RIGHT_EXIT_END:
        # Straight exit: gives the delayed duck room to finish the same curve.
        u = t - RIGHT_ARC_END
        velocity = np.array([EXIT_SPEED, 0.0])
        return PersonState("RIGHT TURN", right_end + velocity * u, 0.0,
                           velocity, 0.0, True,
                           (TURN_DURATION + u) / (RIGHT_EXIT_END - STOP_END))

    exit_end = right_end + np.array([EXIT_SPEED * (RIGHT_EXIT_END - RIGHT_ARC_END), 0.0])
    if t < BACKWARD_END:
        u = t - RIGHT_EXIT_END
        velocity = np.array([-BACK_SPEED, 0.0])
        return PersonState("BACKWARD", exit_end + velocity * u, 0.0, velocity,
                           0.0, True, u / (BACKWARD_END - RIGHT_EXIT_END))

    back_end = exit_end + np.array([-BACK_SPEED * (BACKWARD_END - RIGHT_EXIT_END), 0.0])
    return PersonState("DONE", back_end, 0.0, np.zeros(2), 0.0, False,
                       min((t - BACKWARD_END) / (DEMO_SECONDS - BACKWARD_END), 1.0))


class FootstepTrail:
    """Return the world-space point the leader walked ``gap`` metres earlier.

    This is the core of the behavior. An earlier iteration selected the duck's
    command from the leader's CURRENT pose; when the leader began turning, the
    duck turned immediately from its own coordinates and cut across the inside
    of the corner instead of arriving at it.

    Here the leader's accumulated path is recorded in world coordinates at
    50 Hz and the duck pursues the interpolated point one ``gap`` of PATH
    LENGTH behind. A corner therefore stays in the queue: the duck keeps
    walking straight and turns where the leader actually turned.
    """

    def __init__(self, initial: PersonState, gap: float = TRAIL_DISTANCE):
        self.gap = gap
        self.path_s = 0.0
        self.previous_pos = initial.pos.copy()
        # Seed with a virtual sample one gap BEHIND the leader's start so the
        # queue is well-defined before any distance has accumulated.
        virtual_start = initial.pos - gap * _forward(initial.yaw)
        self.distances = [-gap, 0.0]
        self.samples = [
            TrailState("FORWARD", virtual_start, initial.yaw, -gap, 0.0, False),
            TrailState("FORWARD", initial.pos.copy(), initial.yaw, 0.0, 0.0, False),
        ]

    def update(self, leader: PersonState) -> TrailState:
        delta = float(np.linalg.norm(leader.pos - self.previous_pos))
        if delta > 1e-8:
            # Only real displacement advances the queue, so a stopped leader
            # freezes it rather than filling it with duplicate samples.
            self.path_s += delta
            self.distances.append(self.path_s)
            self.samples.append(TrailState(
                leader.phase, leader.pos.copy(), leader.yaw,
                self.path_s, self.path_s, leader.moving))
        self.previous_pos = leader.pos.copy()

        target_s = self.path_s - self.gap
        upper = min(max(bisect_right(self.distances, target_s), 1),
                    len(self.distances) - 1)
        lower = upper - 1
        a, b = self.samples[lower], self.samples[upper]
        span = self.distances[upper] - self.distances[lower]
        fraction = 0.0 if span <= 1e-10 else (
            target_s - self.distances[lower]) / span
        pos = a.pos + fraction * (b.pos - a.pos)
        yaw = wrap(a.yaw + fraction * wrap(b.yaw - a.yaw))
        return TrailState(
            phase=b.phase,
            pos=pos,
            yaw=yaw,
            path_s=target_s,
            leader_path_s=self.path_s,
            moving=leader.moving,
        )


class FollowController:
    """Replay the motion stored at the delayed world-space trail point."""

    def __init__(self, hz: float = CTRL_HZ):
        self.dt = 1.0 / hz
        self.command = np.zeros(3, dtype=np.float32)

    @staticmethod
    def turn_rate(yaw_error: float) -> float:
        """Measured asymmetric yaw command for a heading error, in rad/s.

        The stock policy's positive-yaw (left) authority is much weaker than
        its negative-yaw (right) authority, so the two directions get separate
        measured magnitude bands. Inside the deadband the command is exactly
        zero: small continuous corrections sit below the policy's gait-onset
        threshold and only produce shuffling.
        """
        if abs(yaw_error) < YAW_DEADBAND_RAD:
            return 0.0
        if yaw_error > 0.0:
            magnitude = min(LEFT_WZ_MAX, max(LEFT_WZ_MIN, YAW_GAIN * yaw_error))
        else:
            magnitude = min(RIGHT_WZ_MAX, max(RIGHT_WZ_MIN, YAW_GAIN * -yaw_error))
        return math.copysign(magnitude, yaw_error)

    @staticmethod
    def replay_phase(leader: PersonState, trail: TrailState) -> str:
        """Phase the duck replays: the queued one, except when reversing.

        Reversal is the single safety exception to trail following. If the duck
        waited for the leader's reverse footprint to arrive, the leader would
        walk into it; so the duck backs off immediately. Turning and lateral
        motion still come exclusively from the recorded world-space trail.
        """
        return "BACKWARD" if leader.phase == "BACKWARD" else trail.phase

    def update(self, leader: PersonState, trail: TrailState,
               duck_pos: np.ndarray, duck_yaw: float) -> tuple[np.ndarray, dict]:
        error_world = trail.pos - np.asarray(duck_pos[:2], dtype=np.float64)
        yaw_error = wrap(trail.yaw - duck_yaw)
        replay = self.replay_phase(leader, trail)

        if not leader.moving:
            raw_command = IDLE_CMD
        elif replay in TURN_PHASES:
            # Close the loop on the measured trunk world yaw so each turn uses
            # the empirically verified sign and stops at its target heading
            # instead of inheriting the previous one.
            raw_command = (TURN_VX, 0.0, self.turn_rate(yaw_error))
        elif replay == "FORWARD":
            raw_command = FORWARD_CMD
        elif replay == "BACKWARD":
            raw_command = BACKWARD_CMD
        else:
            raw_command = IDLE_CMD

        target_cmd = np.array(raw_command, dtype=np.float32)
        alpha = min(1.0, self.dt / CMD_TAU)
        self.command += alpha * (target_cmd - self.command)
        metrics = {
            "target_pos": trail.pos,
            "error_world": error_world,
            "error": float(np.linalg.norm(error_world)),
            "yaw_error": yaw_error,
            "target_cmd": target_cmd,
            "trail_phase": trail.phase,
            "replay_phase": replay,
            "trail_path_s": trail.path_s,
            "leader_path_s": trail.leader_path_s,
            "spatial_lag": trail.leader_path_s - trail.path_s,
        }
        return self.command.copy(), metrics


def animate_person(model, data, person: PersonState, t: float) -> None:
    """Apply the leader root pose and a speed-dependent opposing limb cycle."""
    body_id = model.body("person").id
    mocap_id = int(model.body_mocapid[body_id])
    data.mocap_pos[mocap_id, :2] = person.pos
    data.mocap_pos[mocap_id, 2] = 0.36 + (
        0.006 * abs(math.sin(2 * math.pi * t * 1.3)) if person.moving else 0.0)
    data.mocap_quat[mocap_id] = np.array([
        math.cos(person.yaw / 2.0), 0.0, 0.0, math.sin(person.yaw / 2.0)
    ])

    amplitude = math.radians(24.0) if person.moving else 0.0
    stride = amplitude * math.sin(2.0 * math.pi * 1.3 * t)
    values = {
        "person_hip_l": stride,
        "person_hip_r": -stride,
        "person_shoulder_l": -0.65 * stride,
        "person_shoulder_r": 0.65 * stride,
    }
    for name, value in values.items():
        joint_id = model.joint(name).id
        data.qpos[int(model.jnt_qposadr[joint_id])] = value
        data.qvel[int(model.jnt_dofadr[joint_id])] = 0.0
