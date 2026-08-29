"""Forward-roll task for the footed, 16-servo Growbot."""

from copy import deepcopy

from mjlab_microduck.robot.growbot_constants import GROWBOT_ROBOT_CFG
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    MicroduckRouladeRlCfg,
    make_microduck_roulade_env_cfg,
)


def make_growbot_roulade_env_cfg(play: bool = False):
    """Derive the proven footed Roulade task and swap in Growbot physics."""
    cfg = make_microduck_roulade_env_cfg(play=play)
    cfg.scene.entities = {"robot": GROWBOT_ROBOT_CFG}
    # The framework default is framed for a much larger scene.  Keep recorded
    # evaluation videos close enough to inspect this 25 cm robot and its arms.
    cfg.viewer.distance = 0.65
    cfg.viewer.elevation = -12.0
    cfg.viewer.azimuth = 135.0
    return cfg


GrowbotRouladeRlCfg = deepcopy(MicroduckRouladeRlCfg)
GrowbotRouladeRlCfg.experiment_name = "growbot_roulade"
GrowbotRouladeRlCfg.run_name = "growbot_footed_16dof"
