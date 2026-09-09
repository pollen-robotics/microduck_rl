"""Procedural lower-floor -> square-edged upper-floor terrain for Microduck."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeometry,
    TerrainOutput,
)

# Start on a flush floor so every environment can generate successful
# endpoint examples before it is asked to negotiate a vertical edge.  The
# monotonic curriculum then introduces the edge in ~2.8 mm increments.
STEP_HEIGHT_MIN = 0.0
STEP_HEIGHT_MAX = 0.025


def step_height_by_difficulty(
    difficulty: float,
    height_min: float = STEP_HEIGHT_MIN,
    height_max: float = STEP_HEIGHT_MAX,
) -> float:
    """Linearly map curriculum difficulty to step height in metres."""
    d = float(np.clip(difficulty, 0.0, 1.0))
    return height_min + d * (height_max - height_min)


@dataclass(kw_only=True)
class UpStepTerrainCfg(SubTerrainCfg):
    """A lower floor followed by one continuous, vertical-edged upper floor.

    The spawn origin is on the lower floor, a randomized distance before the
    edge. This represents the Mac/camera switching to the dedicated skill with
    imperfect distance estimation. The actor itself remains proprioceptive.
    """

    lower_length: float = 1.0
    approach_distance_range: tuple[float, float] = (0.20, 0.40)
    height_min: float = STEP_HEIGHT_MIN
    height_max: float = STEP_HEIGHT_MAX
    thickness: float = 0.10

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng) -> TerrainOutput:
        assert 0.0 < self.lower_length < self.size[0]
        approach_lo, approach_hi = self.approach_distance_range
        assert 0.0 < approach_lo <= approach_hi < self.lower_length

        body = spec.body("terrain")
        tile_length, tile_width = self.size
        upper_length = tile_length - self.lower_length
        height = step_height_by_difficulty(difficulty, self.height_min, self.height_max)
        t = self.thickness

        lower = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(self.lower_length / 2.0, tile_width / 2.0, t / 2.0),
            pos=(self.lower_length / 2.0, tile_width / 2.0, -t / 2.0),
        )
        upper = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(upper_length / 2.0, tile_width / 2.0, t / 2.0),
            pos=(
                self.lower_length + upper_length / 2.0,
                tile_width / 2.0,
                height - t / 2.0,
            ),
        )

        approach_distance = float(rng.uniform(approach_lo, approach_hi))
        # Terrain-generator world positions are tile CORNERS. Put both the
        # geometry and spawn at the tile's lateral center, not at its boundary.
        origin = np.array(
            [self.lower_length - approach_distance, tile_width / 2.0, 0.0]
        )
        return TerrainOutput(
            origin=origin,
            geometries=[
                TerrainGeometry(geom=lower, color=(0.45, 0.45, 0.45, 1.0)),
                TerrainGeometry(geom=upper, color=(0.58, 0.36, 0.17, 1.0)),
            ],
        )
