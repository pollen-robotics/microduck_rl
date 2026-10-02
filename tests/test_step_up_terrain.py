import math

import mujoco
import numpy as np

from mjlab_microduck.tasks.step_up_terrain import (
    STEP_HEIGHT_MAX,
    STEP_HEIGHT_MIN,
    UpStepTerrainCfg,
    step_height_by_difficulty,
)


def _empty_terrain_spec():
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    return spec


def test_step_height_curriculum_endpoints_and_clamping():
    assert STEP_HEIGHT_MIN == 0.0
    assert math.isclose(step_height_by_difficulty(0.0), STEP_HEIGHT_MIN)
    assert math.isclose(step_height_by_difficulty(1.0), STEP_HEIGHT_MAX)
    assert math.isclose(step_height_by_difficulty(-1.0), STEP_HEIGHT_MIN)
    assert math.isclose(step_height_by_difficulty(2.0), STEP_HEIGHT_MAX)


def test_up_step_is_continuous_upper_floor_with_square_edge():
    cfg = UpStepTerrainCfg(lower_length=1.0, approach_distance_range=(0.3, 0.3))
    cfg.size = (3.0, 1.2)
    out = cfg.function(1.0, _empty_terrain_spec(), np.random.default_rng(0))
    assert len(out.geometries) == 2
    lower, upper = (g.geom for g in out.geometries)
    lower_end = lower.pos[0] + lower.size[0]
    upper_start = upper.pos[0] - upper.size[0]
    assert math.isclose(lower_end, upper_start, abs_tol=1e-9)
    assert math.isclose(upper.pos[2] + upper.size[2], 0.025, abs_tol=1e-9)
    assert math.isclose(out.origin[0], 0.7, abs_tol=1e-9)
    assert math.isclose(out.origin[1], 0.6, abs_tol=1e-9)
    assert math.isclose(lower.pos[1], 0.6, abs_tol=1e-9)
    assert math.isclose(upper.pos[1], 0.6, abs_tol=1e-9)


def test_spawn_distance_is_randomized_within_camera_trigger_range():
    cfg = UpStepTerrainCfg(approach_distance_range=(0.2, 0.4))
    cfg.size = (3.0, 1.2)
    for seed in range(20):
        out = cfg.function(0.5, _empty_terrain_spec(), np.random.default_rng(seed))
        distance = cfg.lower_length - out.origin[0]
        assert 0.2 <= distance <= 0.4
