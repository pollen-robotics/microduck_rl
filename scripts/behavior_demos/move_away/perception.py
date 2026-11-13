"""Head-camera perception for the move-away demo.

Perception is a RAY CAST from the head camera to the person, not segmentation
rendering. Segmentation is unusable here: the camera sits INSIDE the robot's own
jaw geometry, so every pixel/ray hits ``jaw_soft`` first and the person is never
visible. The ray cast skips the robot's own bodies and answers the question that
actually matters: is the person inside the field of view AND unoccluded?
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .gaze import bearing_elevation, camera_axes, in_frustum

# Small step so the first ray starts off the camera site itself.
SELF_SKIP = 0.02
# Maximum self-geometry hits to walk through before giving up.
MAX_RAY_STEPS = 8


def body_subtree(model: mujoco.MjModel, root: int) -> set[int]:
    """All body ids in the subtree rooted at ``root`` (inclusive)."""
    subtree = {root}
    for body in range(model.nbody):
        parent = body
        while parent > 0:
            if parent == root:
                subtree.add(body)
                break
            parent = model.body_parentid[parent]
    return subtree


def optical_frame_quat() -> np.ndarray:
    """The -90 deg local-Z rotation that fixes the exported head camera.

    The ``head_camera`` quaternion exported by the upstream MJCF is
    ``[0 0 -1 0]``, which points the optical ``-Z`` axis along world ``-X`` —
    backwards, into the robot's own CAD — while the robot walks towards ``+X``.
    Rendering from it shows internal geometry instead of the scene ahead.

    This correction makes optical ``-Z`` follow the head's forward axis and
    ``+Y`` follow world up. It is applied to the demo's in-memory ``MjModel``
    only: it changes rendering and perception, never physics, and it does not
    modify the committed MJCF.
    """
    return np.array([math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)], dtype=np.float64)


class PersonTracker:
    """Field-of-view + occlusion test for the scripted person prop."""

    def __init__(
        self,
        model: mujoco.MjModel,
        camera_name: str = "head_camera",
        person_body: str = "person",
        robot_root_body: str = "trunk_base",
        aspect: float = 1.0,
    ) -> None:
        self.model = model
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.camera_id < 0:
            raise ValueError(f"camera {camera_name!r} not found in the model")
        person_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, person_body)
        if person_id < 0:
            raise ValueError(f"body {person_body!r} not found in the model")
        root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_root_body)
        if root_id < 0:
            raise ValueError(f"body {robot_root_body!r} not found in the model")

        self.person_body_id = person_id
        self.person_mocap_id = int(model.body_mocapid[person_id])
        if self.person_mocap_id < 0:
            raise ValueError(f"body {person_body!r} is not a mocap body")
        self.person_tree = body_subtree(model, person_id)
        self.self_bodies = body_subtree(model, root_id)

        half_v = math.radians(float(model.cam_fovy[self.camera_id])) * 0.5
        self.tan_half_v = math.tan(half_v)
        self.tan_half_h = aspect * self.tan_half_v

    def look(self, data: mujoco.MjData) -> dict:
        """Evaluate visibility of the person from the current camera pose."""
        eye = data.cam_xpos[self.camera_id].copy()
        forward, right, up = camera_axes(data.cam_xmat[self.camera_id])
        to_person = data.mocap_pos[self.person_mocap_id].copy() - eye
        distance = float(np.linalg.norm(to_person))
        unit = to_person / max(distance, 1e-9)

        visible_in_fov = in_frustum(
            to_person, forward, right, up, self.tan_half_h, self.tan_half_v
        )
        bearing, elevation = bearing_elevation(to_person, forward, right, up)
        off_axis = math.acos(float(np.clip(np.dot(unit, forward), -1.0, 1.0)))

        occluded = False
        if visible_in_fov:
            occluded = self._occluded(data, eye, unit, distance)

        return {
            "visible": visible_in_fov and not occluded,
            "in_fov": visible_in_fov,
            "occluded": occluded,
            "distance": distance,
            "bearing": bearing,
            "elevation": elevation,
            "off_axis": off_axis,
        }

    def _occluded(
        self, data: mujoco.MjData, eye: np.ndarray, unit: np.ndarray, distance: float
    ) -> bool:
        geom_id = np.zeros(1, dtype=np.int32)
        travelled = SELF_SKIP
        for _ in range(MAX_RAY_STEPS):
            origin = eye + unit * travelled
            hit = mujoco.mj_ray(self.model, data, origin, unit, None, 1, -1, geom_id)
            if geom_id[0] < 0 or hit < 0.0:
                return False  # clear line of sight
            body = int(self.model.geom_bodyid[int(geom_id[0])])
            if body in self.person_tree:
                return False  # we can see them
            if body in self.self_bodies:
                travelled += hit + 0.005  # our own head: step past it
                if travelled >= distance:
                    return False
                continue
            return travelled + hit < distance - 0.02
        return False
