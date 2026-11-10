"""Regression test for issue #76: ToF beams into the apartment stairwell.

Before the fix, the stairwell (x ∈ [-0.40, 0.00], y ∈ [-1.40, -0.70]) had
steps but no bottom or walls, and the surrounding floor is a 1 cm slab. A
shallow-angle downward ray entering the hole passed under the slab and hit
nothing — mj_ray returns -1 and sim/tof.py reports STATUS_NO_TARGET (255),
where a real VL53L8CX would measure the well wall/bottom.

Fix adds five boxes (well_bottom + well_n/s/w/e) lining the hole. The test
fires the exact beams from the issue report and asserts:
  - 14.1° and 19.7° now hit the south wall (~1.19 m / ~1.22 m),
  - 8.4° is unchanged (legitimately clears the hole and sees far floor).
Pure CPU (mujoco only), no GPU or mjlab dependency.
"""

import math

import mujoco
import numpy as np
import pytest

_SCENE = "src/mjlab_microduck/robot/microduck/scene_apartment.xml"
_SENSOR_POS = np.array([-0.2, -0.25, 0.246])


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(_SCENE)


def _beam(model, elevation_deg):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    e = math.radians(elevation_deg)
    v = np.array([0.0, -math.cos(e), -math.sin(e)])
    geom_ids = np.zeros(1, dtype=np.int32)
    return mujoco.mj_ray(model, data, _SENSOR_POS, v, None, 1, -1, geom_ids)


def test_stairwell_beams_hit_well(model):
    south = _beam(model, 14.1)
    deeper = _beam(model, 19.7)
    assert south > 1.0, f"14.1° beam should hit the well south wall, got {south}"
    assert deeper > 1.0, f"19.7° beam should hit the well south wall, got {deeper}"
    # Both walls belong to the well shell (issue predicted 1.186 / 1.221).
    assert south == pytest.approx(1.186, abs=0.05)
    assert deeper == pytest.approx(1.221, abs=0.05)


def test_shallow_beam_still_sees_far_floor(model):
    shallow = _beam(model, 8.4)
    # Unchanged behaviour: the shallow beam legitimately clears the hole and
    # hits the floor beyond it.
    assert shallow == pytest.approx(1.6566, abs=0.01)
