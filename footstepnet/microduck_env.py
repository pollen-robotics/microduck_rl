"""Register FootstepNet planning envs with microduck's footstep contract (src/mjlab_microduck/footsteps.py).

The contract module is loaded BY PATH (no torch/mjlab in this venv). gym_footsteps_planning hard-codes a few
Sigmaban-scale values outside its options dict; they are rescaled here:
  - start/goal poses sampled in ±AREA m (Sigmaban ±2 m with 8 cm steps ≈ 25+ steps; microduck steps are ~2x
    shorter, so ±1 m keeps episodes well under the 50-step cap)
  - the obstacle sits OBSTACLE_X ahead of the target (Sigmaban 0.3 m, the ball it approaches)
"""

import importlib.util
import math
from pathlib import Path

import gymnasium
import numpy as np
from gym_footsteps_planning.envs.footsteps_planning_env import FootstepsPlanningEnv

_spec = importlib.util.spec_from_file_location(
    "microduck_footsteps", Path(__file__).resolve().parents[1] / "src" / "mjlab_microduck" / "footsteps.py")
contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract)

AREA = 1.0  # [m]
OBSTACLE_X = 0.15  # [m]
MAX_EPISODE_STEPS = 50
ENV_ID = "microduck-footsteps-any-obstacle-multigoal-v0"


def set_limits(fwd: float, bwd: float, dy: float, dtheta_deg: float) -> None:
    """Override the contract's step limits for THIS process (provisional runs); the source of truth stays footsteps.py."""
    contract.MAX_DX_FORWARD, contract.MAX_DX_BACKWARD, contract.MAX_DY = fwd, bwd, dy
    contract.MAX_DTHETA = math.radians(dtheta_deg)


class MicroduckFootstepsEnv(FootstepsPlanningEnv):
    def __init__(self, train=False, visualize=False, render_mode="none", options=None):
        opts = contract.footstepnet_options()
        opts.update(has_obstacle=True, multi_goal=True, shaped=True, foot="any",
                    obstacle_position=np.array([OBSTACLE_X, 0], dtype=np.float32))
        opts.update(options or {})
        super().__init__(options=opts, train=train, visualize=visualize, render_mode=render_mode)

    def reset(self, seed=None, options=None):
        options = dict(options or {})
        if seed is not None:
            super(FootstepsPlanningEnv, self).reset(seed=seed)  # seed np_random before sampling below
        lo, hi = [-AREA, -AREA, -math.pi], [AREA, AREA, math.pi]
        options.setdefault("start_foot_pose", self.np_random.uniform(lo, hi))
        options.setdefault("target_foot_pose", self.np_random.uniform(lo, hi))
        return super().reset(seed=None, options=options)


def register() -> None:
    if ENV_ID not in gymnasium.registry:
        gymnasium.register(id=ENV_ID, entry_point=MicroduckFootstepsEnv, max_episode_steps=MAX_EPISODE_STEPS)


register()
