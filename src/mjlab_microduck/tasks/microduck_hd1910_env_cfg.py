"""Microduck geometry with LuwuDynamics' public 1910 M6 actuator baseline.

Parameters and provenance: robot/hd1910/. This does not replace the robot's
measured mass/inertia, calibration, or runtime gain/IMU integration.
"""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab_microduck.actuator.feetech_bam import FeetechBamActuatorCfg
from .microduck_velocity_env_cfg import MicroduckRlCfg, make_microduck_velocity_env_cfg


HD1910_M6_JSON = Path(__file__).resolve().parents[1] / "robot" / "hd1910" / "1910_m6.json"


def make_microduck_hd1910_velocity_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
    # The base factory uses a module-level EntityCfg shared by other tasks.
    # Copy before replacing actuators, including nested home-pose dictionaries.
    robot = deepcopy(cfg.scene.entities["robot"])
    assert robot.articulation is not None
    robot.articulation.actuators = (
        FeetechBamActuatorCfg(
            json_path=str(HD1910_M6_JSON),
            target_names_expr=(r"^(?!passive_).*",),
            kp_fw=5.0,
            vin_range=(7.4, 8.0),
            vin_drop_gain_range=(0.0, 0.2),
            vin_min=7.0,
            delay_min_lag=3,
            delay_max_lag=6,
        ),
    )
    cfg.scene.entities["robot"] = robot
    return cfg


MicroduckHD1910RlCfg = replace(
    deepcopy(MicroduckRlCfg),
    experiment_name="microduck_hd1910_velocity",
    run_name="hd1910_m6_luwu",
    logger="tensorboard",
)
