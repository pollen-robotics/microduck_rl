"""Run a FootstepNet planner exported by footstepnet/export_onnx.py (numpy + onnxruntime only).

Replicates gym_footsteps_planning's FootstepsPlanningEnv around the network: observation (support → goal, goal-foot
flag, support → obstacle, with the left-support mirror), the action's mirror / box clip / ellipsoid clip, the step
from the NEUTRAL pose, and the reached test. Poses are planar (x, y, yaw) sole centres in the world; feet are
"left" / "right". `plan()` rolls the planner out to the goal: a closed-loop user replans from the actual support
foot after every touchdown and takes the first step.
"""

import json
import math

import numpy as np
import onnxruntime as ort


def _frame(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def _pose(T):
    return np.array([T[0, 2], T[1, 2], math.atan2(T[1, 0], T[0, 0])])


class FootstepPlanner:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.options = json.loads(meta["options"])
        o = self.options
        self.feet_spacing = o["feet_spacing"]
        self.high = np.array([o["max_dx_forward"], o["max_dy"], o["max_dtheta"]])
        self.max_dx_backward = o["max_dx_backward"]

    def neutral(self, support_pose, support_foot):
        """World pose where the swing foot lands for a zero step (parallel, feet_spacing beside the support)."""
        sign = 1.0 if support_foot == "right" else -1.0
        return _frame(*support_pose) @ _frame(0.0, sign * self.feet_spacing, 0.0)

    def observation(self, support_pose, support_foot, goal_pose, goal_foot, obstacle=None):
        """FootstepsPlanningEnv.get_observation(); obstacle = (x, y, radius) in the world or None (radius 0)."""
        T_inv = np.linalg.inv(_frame(*support_pose))
        T_goal = T_inv @ _frame(*goal_pose)
        target = [T_goal[0, 2], T_goal[1, 2], T_goal[0, 0], T_goal[1, 0]]
        ox, oy, r = obstacle if obstacle is not None else (goal_pose[0], goal_pose[1], 0.0)
        obs_xy = (T_inv @ [ox, oy, 1.0])[:2]
        if support_foot == "left":
            target[1], target[3] = -target[1], -target[3]
            obs_xy[1] = -obs_xy[1]
        return np.array([*target, float(support_foot == goal_foot), *obs_xy, r], dtype=np.float32)

    def step(self, obs, support_foot):
        """Network → neutral-relative step in the support foot's own convention (mirrored back, clipped)."""
        a = self.session.run(None, {"obs": obs[None]})[0][0].astype(np.float64)
        if support_foot == "left":
            a[1], a[2] = -a[1], -a[2]
        a = np.clip(a, -self.high, self.high)
        factor = np.array([self.high[0] if a[0] >= 0 else self.max_dx_backward, self.high[1], self.high[2]])
        norm = np.linalg.norm(a / factor)
        return a / norm if norm > 1 else a

    def reached(self, support_pose, support_foot, goal_pose, goal_foot):
        T = np.linalg.inv(_frame(*support_pose)) @ _frame(*goal_pose)
        return (support_foot == goal_foot and math.hypot(T[0, 2], T[1, 2]) < self.options["tolerance_distance"]
                and abs(math.atan2(T[1, 0], T[0, 0])) < self.options["tolerance_angle"])

    def plan(self, support_pose, support_foot, goal_pose, goal_foot, obstacle=None, max_steps=50):
        """Roll out to the goal → ([(world pose, foot)] of every planned landing, reached?). The first entry is the
        swing foot's next target."""
        steps = []
        pose, foot = np.asarray(support_pose, dtype=np.float64), support_foot
        for _ in range(max_steps):
            if self.reached(pose, foot, goal_pose, goal_foot):
                return steps, True
            d = self.step(self.observation(pose, foot, goal_pose, goal_foot, obstacle), foot)
            pose = _pose(self.neutral(pose, foot) @ _frame(*d))
            foot = "left" if foot == "right" else "right"
            steps.append((pose, foot))
        return steps, self.reached(pose, foot, goal_pose, goal_foot)
