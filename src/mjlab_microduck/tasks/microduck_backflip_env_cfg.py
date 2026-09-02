"""Experimental staged backflip task for MicroDuck.

The first version intentionally exposes a reverse-curriculum landing task:
some resets begin late in a tucked backward rotation, while the rest begin
standing.  This makes the landing controller learnable before PPO has found a
full takeoff.  A trajectory-optimization or motion-reference stage should be
added before expecting a reliable sim-to-real aerial backflip.
"""

import math
from copy import deepcopy

from mjlab.managers import EventTermCfg, RewardTermCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    MicroduckRouladeRlCfg,
    make_microduck_roulade_env_cfg,
)


BACKFLIP_STAND_HEIGHT = 0.115
BACKFLIP_TUCK_OVERRIDES = {
    2: -1.25,
    3: 1.35,
    4: 1.05,
    5: -0.8,
    6: 0.8,
    11: 1.25,
    12: -1.35,
    13: -1.05,
}
BACKFLIP_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]


def make_microduck_backflip_env_cfg(play: bool = False):
    """Create the staged backward aerial rotation environment."""
    cfg = deepcopy(make_microduck_roulade_env_cfg(play=play))
    cfg.episode_length_s = 5.0

    cfg.rewards.clear()
    cfg.rewards["backflip_progress"] = RewardTermCfg(
        func=microduck_mdp.backflip_progress,
        weight=8.0,
        params={"target_angle": 2 * math.pi, "max_paid_rate": 10.0},
    )
    cfg.rewards["backflip_landing"] = RewardTermCfg(
        func=microduck_mdp.backflip_landing,
        weight=6.0,
        params={
            "target_height": BACKFLIP_STAND_HEIGHT,
            "height_std": 0.04,
            "upright_std": 0.4,
            "pose_std": 0.4,
            "joint_indices": BACKFLIP_LEG_JOINTS,
            "gate_lo": math.radians(300.0),
            "gate_hi": math.radians(355.0),
        },
    )
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.05,
        params={"sensor_name": "self_collision"},
    )

    # Do not terminate at the inverted or airborne states that make the task
    # interesting.  NaN and timeout protection remain active.
    cfg.terminations.pop("fell_over", None)
    cfg.curriculum.clear()

    cfg.events.pop("set_roulade_state", None)
    cfg.events["set_backflip_state"] = EventTermCfg(
        func=microduck_mdp.reset_backflip_state,
        mode="reset",
        params={
            "late_spawn_prob": 0.25,
            "late_angle_range": (math.radians(25.0), math.radians(320.0)),
            "late_height_range": (0.10, 0.18),
            "late_omega_range": (0.0, 4.0),
            "standing_height_range": (0.11, 0.12),
            "tuck_overrides": BACKFLIP_TUCK_OVERRIDES,
            "joint_noise_std": 0.02,
        },
    )
    return cfg


MicroduckBackflipRlCfg = deepcopy(MicroduckRouladeRlCfg)
MicroduckBackflipRlCfg.experiment_name = "microduck_backflip"
MicroduckBackflipRlCfg.run_name = "microduck_backflip"
MicroduckBackflipRlCfg.max_iterations = 10_000
