"""Pedestrian routes, footstep queues and the SEARCH/FOUND/FOLLOW/STOP machine.

This module is deliberately free of any ``mujoco`` import: everything here is
plain geometry and bookkeeping, so it runs (and is unit-tested) on CPU without
a model, a policy or a renderer. ``animate_crowd`` touches MuJoCo structures but
only through duck-typed attribute access, which keeps it testable with stubs.
"""

import math
from bisect import bisect_right
from dataclasses import dataclass

import numpy as np

# Control rate of the exported policies (50 Hz, see AGENTS.md).
CTRL_HZ = 50.0

# Arc length by which the robot trails the selected pedestrian. The robot walks
# toward where that person *was* 0.55 m ago along their own path, not toward
# their current pose, which is what makes the motion read as "following".
TRAIL_DISTANCE = 0.55

COLORS = ("BLUE", "GREEN", "RED", "YELLOW", "PURPLE")

# BLUE/GREEN/RED can be requested; YELLOW/PURPLE only ever act as moving
# distractors, so a demo that locks onto them is a bug, not a variation.
SELECTABLE_COLORS = ("BLUE", "GREEN", "RED")
DISTRACTOR_COLORS = ("YELLOW", "PURPLE")

# The requested demo sequence. BLUE appears twice on purpose: re-acquiring a
# color that was already followed once exercises the full search path again
# instead of letting the controller keep a latched target.
TARGET_SEQUENCE = ("BLUE", "GREEN", "RED", "BLUE")


def wrap(angle: float) -> float:
    """Wrap an angle to the half-open interval [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class PersonState:
    """World-space pose of one pedestrian at one instant."""

    color: str
    pos: np.ndarray
    yaw: float
    velocity: np.ndarray
    moving: bool = True


@dataclass(frozen=True)
class TrailState:
    """A point queued a fixed arc length behind one pedestrian."""

    pos: np.ndarray
    yaw: float
    path_s: float
    leader_path_s: float


def _ellipse(
    color: str,
    t: float,
    *,
    center: tuple[float, float],
    radii: tuple[float, float],
    omega: float,
    phase: float,
    wobble: float = 0.0,
) -> PersonState:
    """A smooth closed pedestrian route with a small independent wobble.

    The wobble frequencies (0.17 / 0.11 Hz) are intentionally unrelated to the
    lap rate, so the five routes never phase-lock into a formation that would
    make color acquisition artificially easy.
    """
    angle = phase + omega * t
    rx, ry = radii
    pos = np.array(
        [
            center[0] + rx * math.cos(angle) + wobble * math.sin(0.17 * t + phase),
            center[1] + ry * math.sin(angle) + 0.5 * wobble * math.sin(0.11 * t),
        ],
        dtype=np.float64,
    )
    velocity = np.array(
        [
            -rx * omega * math.sin(angle) + 0.17 * wobble * math.cos(0.17 * t + phase),
            ry * omega * math.cos(angle) + 0.055 * wobble * math.cos(0.11 * t),
        ],
        dtype=np.float64,
    )
    yaw = math.atan2(float(velocity[1]), float(velocity[0]))
    return PersonState(color, pos, yaw, velocity, True)


def crowd_trajectory(t: float) -> dict[str, PersonState]:
    """Five logical but deliberately unsynchronised walking routes.

    Every person keeps walking for the whole rollout: nobody freezes, waits or
    teleports to help the robot acquire them.
    """
    return {
        "BLUE": _ellipse(
            "BLUE",
            t,
            center=(1.80, 0.75),
            radii=(0.52, 0.25),
            omega=0.080,
            phase=0.10,
            wobble=0.025,
        ),
        "GREEN": _ellipse(
            "GREEN",
            t,
            center=(0.92, 0.52),
            radii=(0.46, 0.24),
            omega=-0.073,
            phase=1.75,
            wobble=0.030,
        ),
        "RED": _ellipse(
            "RED",
            t,
            center=(1.20, 1.15),
            radii=(0.56, 0.24),
            omega=0.068,
            phase=3.75,
            wobble=0.020,
        ),
        "YELLOW": _ellipse(
            "YELLOW",
            t,
            center=(0.82, 1.00),
            radii=(0.62, 0.20),
            omega=-0.092,
            phase=5.05,
            wobble=0.035,
        ),
        "PURPLE": _ellipse(
            "PURPLE",
            t,
            center=(0.90, -1.00),
            radii=(0.52, 0.22),
            omega=0.087,
            phase=2.75,
            wobble=0.028,
        ),
    }


class FootstepTrail:
    """Interpolate a point a fixed arc length behind one pedestrian.

    Samples are appended in monotonically increasing arc length, so the lookup
    for ``path_s - gap`` is a binary search plus a linear interpolation. Before
    the person has walked ``gap`` metres, the queue extrapolates backwards along
    their initial heading rather than clamping onto their start position.
    """

    def __init__(self, initial: PersonState, gap: float = TRAIL_DISTANCE):
        self.gap = gap
        self.path_s = 0.0
        self.previous_pos = initial.pos.copy()
        virtual_start = initial.pos - gap * np.array(
            [math.cos(initial.yaw), math.sin(initial.yaw)]
        )
        self.distances = [-gap, 0.0]
        self.positions = [virtual_start, initial.pos.copy()]
        self.yaws = [initial.yaw, initial.yaw]

    def update(self, person: PersonState) -> TrailState:
        delta = float(np.linalg.norm(person.pos - self.previous_pos))
        if delta > 1e-8:
            self.path_s += delta
            self.distances.append(self.path_s)
            self.positions.append(person.pos.copy())
            self.yaws.append(person.yaw)
        self.previous_pos = person.pos.copy()

        target_s = self.path_s - self.gap
        upper = min(
            max(bisect_right(self.distances, target_s), 1), len(self.distances) - 1
        )
        lower = upper - 1
        span = self.distances[upper] - self.distances[lower]
        fraction = 0.0 if span <= 1e-10 else (target_s - self.distances[lower]) / span
        pos = self.positions[lower] + fraction * (
            self.positions[upper] - self.positions[lower]
        )
        yaw = wrap(
            self.yaws[lower] + fraction * wrap(self.yaws[upper] - self.yaws[lower])
        )
        return TrailState(pos, yaw, target_s, self.path_s)


class SearchFollowStateMachine:
    """Enforce SEARCH -> FOUND -> FOLLOW -> STOP for each requested target.

    The machine owns *when* the robot may follow; it never looks at pedestrian
    ground truth itself. The only way out of ``SEARCH`` is a camera report in
    which the requested color is inside the field of view and within
    ``FOUND_CONE`` of the crosshair, which is what makes a distractor crossing
    the view harmless.
    """

    FOUND_SECONDS = 1.0
    FOLLOW_SECONDS = 9.0
    STOP_SECONDS = 1.5

    # A minimum dwell keeps the first frame of a sweep from counting as a
    # search; the maximum turns "never acquired" into a loud failure instead of
    # a rollout that silently stands still.
    MIN_SEARCH_SECONDS = 0.4
    MAX_SEARCH_SECONDS = 8.0

    FOUND_CONE = math.radians(8.0)

    def __init__(self, sequence: tuple[str, ...] = TARGET_SEQUENCE):
        if not sequence:
            raise ValueError("target sequence must not be empty")
        unknown = [color for color in sequence if color not in SELECTABLE_COLORS]
        if unknown:
            raise ValueError(f"targets must be selectable colors, got {unknown}")
        self.sequence = tuple(sequence)
        self.index = 0
        self.state = "SEARCH"
        self.state_since = 0.0
        self.cycles: list[dict] = []
        self.current = {
            "selection": 1,
            "target": self.sequence[0],
            "search_start_s": 0.0,
        }

    @property
    def target(self) -> str:
        return self.sequence[min(self.index, len(self.sequence) - 1)]

    @property
    def done(self) -> bool:
        return self.index >= len(self.sequence)

    @property
    def follows_now(self) -> bool:
        """True only while locomotion is allowed."""
        return self.state == "FOLLOW" and not self.done

    def update(self, t: float, camera: dict) -> tuple[str, str, bool]:
        """Advance one control step. Returns (state, target, changed)."""
        if self.done:
            return "DONE", self.sequence[-1], False

        elapsed = t - self.state_since
        changed = False

        if self.state == "SEARCH":
            seen = camera.get("target_visible", False)
            centered = camera.get("target_off_axis", math.pi) < self.FOUND_CONE
            if elapsed >= self.MIN_SEARCH_SECONDS and seen and centered:
                self.state = "FOUND"
                self.current["found_s"] = t
                self.current["search_duration_s"] = elapsed
                changed = True
            elif elapsed >= self.MAX_SEARCH_SECONDS:
                raise RuntimeError(
                    f"camera failed to find {self.target} in {elapsed:.2f}s"
                )
        elif self.state == "FOUND" and elapsed >= self.FOUND_SECONDS:
            self.state = "FOLLOW"
            self.current["follow_start_s"] = t
            changed = True
        elif self.state == "FOLLOW" and elapsed >= self.FOLLOW_SECONDS:
            self.state = "STOP"
            self.current["stop_s"] = t
            changed = True
        elif self.state == "STOP" and elapsed >= self.STOP_SECONDS:
            self.current["cycle_end_s"] = t
            self.cycles.append(dict(self.current))
            self.index += 1
            if self.done:
                return "DONE", self.sequence[-1], True
            self.state = "SEARCH"
            self.current = {
                "selection": self.index + 1,
                "target": self.target,
                "search_start_s": t,
            }
            changed = True

        if changed:
            self.state_since = t
        return self.state, self.target, changed


class CrowdFollowController:
    """Walk toward the selected pedestrian's queued world-space footprint.

    Emits a twist command for the stock walking policy; it does not touch
    joints. Outside ``FOLLOW`` the command is exactly zero, which is what the
    ``stationary_state_command_max`` gate measures.
    """

    # Measured gait-onset command for the stock walking policy. The policy has
    # a hard onset threshold: a smaller vx produces a valid ONNX action but no
    # visible locomotion, which is exactly how the RED hand-off first failed.
    # Keep the walking command during large heading changes instead of slowing
    # down to turn, so every target switch actually initiates gait.
    WALK_VX = 0.24

    # Inside this radius the robot is already on the queued footprint; pushing
    # forward would overshoot into the person.
    ARRIVED_RADIUS = 0.14

    # Heading deadband: below this the robot is aimed well enough to just walk.
    YAW_DEADBAND = math.radians(4.0)

    # Empirically tuned, deliberately asymmetric turn gains: this robot's stock
    # gait yaws faster to the right than to the left, so equal gains produced
    # unequal turns. These are the values used for the published metrics —
    # changing them changes the measured follow segments.
    YAW_GAIN = 1.25
    LEFT_WZ_MIN, LEFT_WZ_MAX = 0.60, 1.00
    RIGHT_WZ_MIN, RIGHT_WZ_MAX = 0.18, 0.32

    def __init__(self):
        self.command = np.zeros(3, dtype=np.float32)

    def update(
        self,
        active: bool,
        trail: TrailState,
        duck_pos: np.ndarray,
        duck_yaw: float,
    ) -> tuple[np.ndarray, dict]:
        error = trail.pos - np.asarray(duck_pos[:2], dtype=np.float64)
        distance = float(np.linalg.norm(error))
        # Far away, steer at the queued footprint; on top of it, adopt the
        # person's heading instead of chasing a numerically noisy bearing.
        desired_yaw = (
            math.atan2(float(error[1]), float(error[0]))
            if distance > 0.10
            else trail.yaw
        )
        yaw_error = wrap(desired_yaw - duck_yaw)

        if not active:
            raw = (0.0, 0.0, 0.0)
        else:
            vx = 0.0 if distance < self.ARRIVED_RADIUS else self.WALK_VX
            if abs(yaw_error) < self.YAW_DEADBAND:
                wz = 0.0
            elif yaw_error > 0.0:
                wz = min(
                    self.LEFT_WZ_MAX,
                    max(self.LEFT_WZ_MIN, self.YAW_GAIN * yaw_error),
                )
            else:
                wz = -min(
                    self.RIGHT_WZ_MAX,
                    max(self.RIGHT_WZ_MIN, self.YAW_GAIN * -yaw_error),
                )
            raw = (vx, 0.0, wz)

        # Applied unfiltered: upstream policies are trained without an action
        # low-pass, and adding smoothing on only one side breaks transfer
        # (see AGENTS.md, "Policies are UNFILTERED").
        self.command = np.asarray(raw, dtype=np.float32)
        return self.command.copy(), {
            "target_pos": trail.pos,
            "error": distance,
            "desired_yaw": desired_yaw,
            "yaw_error": yaw_error,
            "spatial_lag": trail.leader_path_s - trail.path_s,
        }


def animate_crowd(model, data, crowd: dict[str, PersonState], t: float) -> None:
    """Pose all pedestrians and animate independent gait phases.

    The people are mocap bodies with collision-free geoms, so they are scenery:
    they never push the robot, and posing them cannot perturb the walking state.
    """
    for order, (color, person) in enumerate(crowd.items()):
        prefix = color.lower()
        body = model.body(f"person_{prefix}")
        mocap = int(model.body_mocapid[body.id])
        # Offsetting each person's gait phase keeps the crowd from marching in
        # lockstep, which would look like one rigid object to the camera.
        phase_t = t + 0.73 * order
        data.mocap_pos[mocap, :2] = person.pos
        data.mocap_pos[mocap, 2] = 0.36 + 0.006 * abs(
            math.sin(2.0 * math.pi * 1.15 * phase_t)
        )
        data.mocap_quat[mocap] = np.array(
            [math.cos(person.yaw / 2.0), 0.0, 0.0, math.sin(person.yaw / 2.0)]
        )
        stride = math.radians(24.0) * math.sin(2.0 * math.pi * 1.15 * phase_t)
        values = {
            f"{prefix}_hip_l": stride,
            f"{prefix}_hip_r": -stride,
            f"{prefix}_shoulder_l": -0.65 * stride,
            f"{prefix}_shoulder_r": 0.65 * stride,
        }
        for name, value in values.items():
            joint = model.joint(name).id
            data.qpos[int(model.jnt_qposadr[joint])] = value
            data.qvel[int(model.jnt_dofadr[joint])] = 0.0
