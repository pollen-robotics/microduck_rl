"""Color-selective search sweep and target tracking from the head camera.

Scope, stated plainly: this is a **semantic proxy for color recognition**, not a
vision model. The simulator supplies each actor's identity and world pose, and
this module tests known shirt/head sample points against the camera frustum.
What is genuinely simulated is the *geometry* — where the head camera
physically sits, where it is pointing, whether the requested person is inside
the field of view, and how far off the crosshair they are. Replacing the
identity lookup with an onboard RGB classifier is a separate piece of work;
nothing here trains or evaluates one.

The visibility test is deliberately frustum-only: it does NOT ray-cast for
mutual occlusion between pedestrians, so a person standing directly behind
another still counts as visible. This is the gate the published metrics were
measured with. Occlusion rejection is a reasonable extension, but it changes
acquisition timing and would need the whole sequence re-measured, so it is
called out as a limitation instead of being switched on untested.
"""

import math

import mujoco
import numpy as np

from .crowd import COLORS, wrap


class CrowdCameraSearch:
    """Search for one shirt color, then keep that pedestrian centered.

    Locomotion physics stays authoritative in the caller's ``data``. Head pose
    and the stabilized optical view live in an isolated ``MjData`` used only for
    perception and rendering, so gazing around can never perturb the walking
    state that the policy is controlling.
    """

    # Actuator indices for the head chain (see AGENTS.md joint layout: 5-8 are
    # neck_pitch, head_pitch, head_yaw, head_roll). Resolved to joint ids via
    # the model rather than assumed, so this stays correct on models where
    # passive joints interleave.
    HEAD_PITCH_ACT = 6
    HEAD_YAW_ACT = 7
    HEAD_ROLL_ACT = 8

    # One full panoramic sweep per this many seconds during SEARCH.
    SWEEP_SECONDS = 4.5

    # Vertical sample offsets relative to the torso centre: torso, upper torso,
    # head. A crowded scene cannot be judged by a single torso-centre ray —
    # another pedestrian may cover exactly that pixel while the colored shirt
    # and head remain plainly visible. Any clear sample is sufficient.
    SAMPLE_OFFSETS_Z = (-0.06, 0.08, 0.20)

    # Maximum yaw slew per control step while tracking, so the head turns
    # toward the target instead of teleporting onto it.
    MAX_AIM_YAW_STEP = math.radians(5.0)
    MAX_AIM_PITCH_STEP = math.radians(3.0)

    def __init__(self, model, data, qpos_idx, trunk_id, pip_size=(260, 190)):
        self.model = model
        self.gaze_data = mujoco.MjData(model)
        self.qpos_idx = qpos_idx
        self.trunk_id = trunk_id
        self.pip_w, self.pip_h = pip_size

        self.head_pitch_joint = int(model.actuator_trnid[self.HEAD_PITCH_ACT, 0])
        self.head_yaw_joint = int(model.actuator_trnid[self.HEAD_YAW_ACT, 0])
        self.head_cam = model.camera("head_camera").id
        self.follow_cam = model.camera("follow_camera").id
        rig = model.body("follow_camera_rig")
        self.follow_mocap = int(model.body_mocapid[rig.id])

        self.people = {
            color: int(model.body_mocapid[model.body(f"person_{color.lower()}").id])
            for color in COLORS
        }

        # MuJoCo cameras look along -Z. The upstream head camera is exported
        # with its own optical convention, so align the frame before copying
        # its physical position onto the stabilized rig.
        model.cam_quat[self.head_cam] = np.array(
            [math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)], dtype=np.float64
        )
        mujoco.mj_forward(model, data)

        self.gaze_pitch = float(data.qpos[qpos_idx[self.HEAD_PITCH_ACT]])
        self.gaze_yaw = float(data.qpos[qpos_idx[self.HEAD_YAW_ACT]])
        self.view_yaw = 0.0
        self.view_pitch = math.radians(8.0)

        vertical_half = math.radians(float(model.cam_fovy[self.follow_cam])) * 0.5
        self.tan_v = math.tan(vertical_half)
        self.tan_h = (self.pip_w / self.pip_h) * self.tan_v

        self.samples = 0
        self.target_visible_steps = 0
        self.search_steps = 0
        self.search_target_visible_steps = 0
        self.max_target_off_axis = 0.0

    def _pose_gaze(self, data, duck_yaw: float) -> None:
        """Copy authoritative physics into the gaze clone and aim the head."""
        mujoco.mj_copyData(self.gaze_data, self.model, data)
        yaw_lo, yaw_hi = self.model.jnt_range[self.head_yaw_joint]
        pitch_lo, pitch_hi = self.model.jnt_range[self.head_pitch_joint]
        # The demanded view direction is absolute; the servo can only deliver
        # the part of it that fits inside the real neck range.
        relative = wrap(self.view_yaw - duck_yaw)
        self.gaze_yaw = float(np.clip(relative, yaw_lo + 0.03, yaw_hi - 0.03))
        self.gaze_pitch = float(
            np.clip(-self.view_pitch, pitch_lo + 0.04, pitch_hi - 0.04)
        )
        self.gaze_data.qpos[self.qpos_idx[self.HEAD_YAW_ACT]] = self.gaze_yaw
        self.gaze_data.qpos[self.qpos_idx[self.HEAD_PITCH_ACT]] = self.gaze_pitch
        self.gaze_data.qpos[self.qpos_idx[self.HEAD_ROLL_ACT]] = 0.0
        mujoco.mj_forward(self.model, self.gaze_data)

    def _orient_rig(self) -> None:
        """Place the stabilized rig at the physical eye, aimed at the view."""
        eye = self.gaze_data.cam_xpos[self.head_cam].copy()
        cp = math.cos(self.view_pitch)
        forward = np.array(
            [
                cp * math.cos(self.view_yaw),
                cp * math.sin(self.view_yaw),
                math.sin(self.view_pitch),
            ],
            dtype=np.float64,
        )
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        right /= max(float(np.linalg.norm(right)), 1e-9)
        up = np.cross(right, forward)
        up /= max(float(np.linalg.norm(up)), 1e-9)
        rotation = np.column_stack((right, up, -forward))
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, rotation.ravel())
        self.gaze_data.mocap_pos[self.follow_mocap] = eye
        self.gaze_data.mocap_quat[self.follow_mocap] = quaternion
        mujoco.mj_forward(self.model, self.gaze_data)

    def _person_target(self, color: str) -> np.ndarray:
        target = self.gaze_data.mocap_pos[self.people[color]].copy()
        target[2] += 0.03
        return target

    def _aim_toward(self, target, max_yaw_step=None) -> None:
        """Slew the view toward a point at a bounded rate (no teleporting)."""
        if max_yaw_step is None:
            max_yaw_step = self.MAX_AIM_YAW_STEP
        eye = self.gaze_data.cam_xpos[self.head_cam]
        delta = target - eye
        desired_yaw = math.atan2(float(delta[1]), float(delta[0]))
        desired_pitch = math.atan2(float(delta[2]), float(np.linalg.norm(delta[:2])))
        self.view_yaw = wrap(
            self.view_yaw
            + float(
                np.clip(wrap(desired_yaw - self.view_yaw), -max_yaw_step, max_yaw_step)
            )
        )
        self.view_pitch += float(
            np.clip(
                desired_pitch - self.view_pitch,
                -self.MAX_AIM_PITCH_STEP,
                self.MAX_AIM_PITCH_STEP,
            )
        )

    def _visibility(self, color: str) -> tuple[bool, float, float]:
        """Frustum test for one person. Returns (visible, off_axis, range).

        No occlusion rejection: see the module docstring.
        """
        center = self._person_target(color)
        eye = self.gaze_data.cam_xpos[self.follow_cam].copy()
        rotation = self.gaze_data.cam_xmat[self.follow_cam].reshape(3, 3)
        right, up, forward = rotation[:, 0], rotation[:, 1], -rotation[:, 2]
        visible = False
        best_off_axis = math.pi
        center_distance = float(np.linalg.norm(center - eye))
        for z_offset in self.SAMPLE_OFFSETS_Z:
            target = center + np.array([0.0, 0.0, z_offset])
            delta = target - eye
            distance = float(np.linalg.norm(delta))
            unit = delta / max(distance, 1e-9)
            depth = float(np.dot(delta, forward))
            image_x = float(np.dot(delta, right))
            image_y = float(np.dot(delta, up))
            in_fov = (
                depth > 0.0
                and abs(image_x) <= depth * self.tan_h
                and abs(image_y) <= depth * self.tan_v
            )
            off_axis = math.acos(float(np.clip(np.dot(unit, forward), -1.0, 1.0)))
            best_off_axis = min(best_off_axis, off_axis)
            if in_fov:
                visible = True
        return visible, best_off_axis, center_distance

    def update(self, data, *, target_color, mode, mode_elapsed, duck_yaw) -> dict:
        """Advance perception one control step and report what is visible."""
        self._pose_gaze(data, duck_yaw)
        target = self._person_target(target_color)

        if mode == "SEARCH":
            # One complete panoramic sweep, starting behind the robot. The
            # sweep is open-loop: it does not steer toward the answer, so the
            # measured search durations reflect where the person actually was.
            self.view_yaw = wrap(
                duck_yaw - math.pi + 2.0 * math.pi * mode_elapsed / self.SWEEP_SECONDS
            )
            self.view_pitch = math.radians(9.0)
            self.search_steps += 1
        else:
            self._aim_toward(target)

        self._pose_gaze(data, duck_yaw)
        self._orient_rig()

        visible_colors = []
        target_visible = False
        target_off_axis = math.pi
        target_distance = float("nan")
        # Every color is evaluated, including distractors, so the demo can
        # report "the camera saw yellow and did not lock onto it".
        for color in COLORS:
            visible, off_axis, distance = self._visibility(color)
            if visible:
                visible_colors.append(color)
            if color == target_color:
                target_visible = visible
                target_off_axis = off_axis
                target_distance = distance

        self.samples += 1
        if target_visible:
            self.target_visible_steps += 1
            if mode == "SEARCH":
                self.search_target_visible_steps += 1
        self.max_target_off_axis = max(
            self.max_target_off_axis,
            target_off_axis if math.isfinite(target_off_axis) else 0.0,
        )
        return {
            "target_color": target_color,
            "mode": mode,
            "target_visible": target_visible,
            "target_off_axis": target_off_axis,
            "target_distance": target_distance,
            "visible_colors": visible_colors,
            "view_yaw": self.view_yaw,
            "view_pitch": self.view_pitch,
            "gaze_yaw": self.gaze_yaw,
        }

    @property
    def camera_id(self) -> int:
        return self.follow_cam
