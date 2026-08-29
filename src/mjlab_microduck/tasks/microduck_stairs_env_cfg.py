"""Dedicated low-rise stair task for MicroDuck.

The existing rough task mixes stairs, random grids, and slopes.  This task
isolates stair ascent so the curriculum has a clear signal and can be tested
against printable 5-15 mm risers before rough terrain is mixed back in.
"""

from copy import deepcopy

import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    _soften_terrain_contacts,
    make_microduck_velocity_env_cfg,
)


MICRODUCK_STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=8,
    num_cols=8,
    curriculum=True,
    sub_terrains={
        "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.35),
        "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
            proportion=0.65,
            # Difficulty interpolates this range across terrain rows.
            # The 15 mm ceiling matches the current MicroDuck foot-lift budget.
            step_height_range=(0.0, 0.015),
            step_width=0.15,
            platform_width=2.0,
            border_width=1.0,
        ),
    },
    add_lights=False,
)


def make_microduck_stairs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create a stairs-only velocity-tracking environment."""
    cfg = make_microduck_velocity_env_cfg(rough=True)
    cfg.scene.terrain.terrain_generator = deepcopy(MICRODUCK_STAIRS_TERRAINS_CFG)
    # A transferred walking policy should meet the stair curriculum at its
    # easiest row. Successful forward travel promotes environments one level
    # at a time; random initialization across all rows skips that progression.
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.scene.spec_fn = _soften_terrain_contacts
    cfg.sim.nconmax = 200
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50

    if play:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.max_init_terrain_level = None
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.num_cols = 5

    # Stairs are a locomotion task, so keep the shared 61D policy contract and
    # the standard velocity curriculum.  This only changes the terrain mix.
    return cfg


MicroduckStairsRlCfg = deepcopy(MicroduckRlCfg)
MicroduckStairsRlCfg.experiment_name = "microduck_stairs"
MicroduckStairsRlCfg.run_name = "microduck_stairs"
