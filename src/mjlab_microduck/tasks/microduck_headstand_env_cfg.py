"""Experimental head-supported balance task for MicroDuck.

This task trains a head-supported freeze from a measured tucked pose.  It is
deliberately separate from walking and from the roulade: the head must be the
only terrain contact, its flat top must point down, and the body must remain
still.  A successful result is a controlled headstand, not a face-plant or a
three-point tripod.
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


HEADSTAND_HEIGHT = 0.124
HEADSTAND_TUCK_OVERRIDES = {
    2: -1.15,
    3: 1.25,
    4: 1.05,
    5: -1.0,
    6: 1.0,
    11: 1.15,
    12: -1.25,
    13: -1.05,
}


def make_microduck_headstand_env_cfg(play: bool = False):
    """Create the headstand environment using roulade's full-collision model."""
    cfg = deepcopy(make_microduck_roulade_env_cfg(play=play))
    cfg.episode_length_s = 4.0

    # The headstand objective is a sustained state, so walking/rolling rewards
    # and the roulade curriculum must not compete with it.
    cfg.rewards.clear()
    cfg.rewards["headstand_contact"] = RewardTermCfg(
        func=microduck_mdp.headstand_contact,
        weight=1.5,
        params={
            "head_sensor_name": "head_ground_contact",
            "feet_sensor_name": "feet_ground_contact",
        },
    )
    cfg.rewards["headstand_hold"] = RewardTermCfg(
        func=microduck_mdp.headstand_hold,
        weight=8.0,
        params={
            "head_sensor_name": "head_ground_contact",
            "feet_sensor_name": "feet_ground_contact",
            "lin_vel_std": 0.04,
            "ang_vel_std": 0.45,
        },
    )
    cfg.rewards["headstand_height"] = RewardTermCfg(
        func=microduck_mdp.headstand_height,
        weight=1.5,
        params={"target_height": HEADSTAND_HEIGHT, "std": 0.025},
    )
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.15)
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.05,
        params={"sensor_name": "self_collision"},
    )

    # A fall is not a terminal state here.  The fixed-duration episode gives
    # PPO a gradient after an imperfect reset instead of ending immediately.
    cfg.terminations.pop("fell_over", None)
    cfg.curriculum.clear()

    # Start in the measured tucked, head-down pose.  The task can later be
    # extended with a standing-to-headstand phase, but learning the freeze
    # itself is the safer first curriculum stage for this small robot.
    cfg.events.pop("set_roulade_state", None)
    cfg.events["set_headstand_state"] = EventTermCfg(
        func=microduck_mdp.reset_roulade_state,
        mode="reset",
        params={
            "standing_prob": 0.0,
            "midroll_prob": 1.0,
            "midroll_pitch_min": math.radians(90.0),
            "midroll_pitch_max": math.radians(90.0),
            "midroll_z_min": 0.125,
            "midroll_z_max": 0.135,
            "midroll_omega_range": (0.0, 0.0),
            "tuck_overrides": HEADSTAND_TUCK_OVERRIDES,
            "tuck_factor_range": (1.0, 1.0),
            "joint_noise_std": 0.01,
        },
    )
    return cfg


MicroduckHeadstandRlCfg = deepcopy(MicroduckRouladeRlCfg)
MicroduckHeadstandRlCfg.experiment_name = "microduck_headstand"
MicroduckHeadstandRlCfg.run_name = "microduck_headstand"
MicroduckHeadstandRlCfg.max_iterations = 3_000
