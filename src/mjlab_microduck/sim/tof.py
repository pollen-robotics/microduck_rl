"""A simulated VL53L5CX, so the duck can see a wall.

8x8 zones over a 45-degree square field of view, out to 4 m, at 15 Hz — the real sensor's shape,
because `tofd` publishes frames of exactly that and `maploc` reprojects them assuming it.

Modelled on `~/MISC/microduck_maploc`'s `sim/tof_sensor.py`, which had these numbers from the
datasheet and from the sensor on a desk: noise that grows with distance (millimetres up close,
centimetres out near the limit), and a status per zone rather than a distance alone.

**The status byte matters as much as the distance.** A real sensor distinguishes "nothing out there"
from "could not measure", and `maploc` treats them differently — a zone with no target is empty
space to clear on the map, and a zone that failed is no information at all. A simulator reporting
only distances would let a bug through that hardware finds.
"""

from __future__ import annotations

import mujoco
import numpy as np

# The sensor, as `tof/src/lib.rs` publishes it.
ROWS = 8
COLS = 8
ZONES = ROWS * COLS
STATUS_VALID = 5
STATUS_NO_TARGET = 255

# VL53L5CX: 45 degrees per axis (63 on the diagonal), 4 m of range.
FOV_DEG = 45.0
MAX_RANGE = 4.0


class Tof:
    """The 8x8 sensor on one duck's `tof` site.

    Rays are cast in the site's frame — +x forward, +y left, +z up — so a head that turns takes the
    sensor with it, which is the whole point of `robot.look` scanning a room.
    """

    def __init__(self, model: mujoco.MjModel, site: int, seed: int = 0):
        self.model = model
        self.site = site
        self.random = np.random.default_rng(seed)

        # Zone centres, once. Azimuth about +z (positive toward +y, the duck's left) and elevation
        # up, both spanning the field of view — row 0 is the top of the frame, and column 0 the
        # sensor's LEFT, as `kinematics::tof` reads the real sensor's buffer. With column 0 on the
        # right every frame reached the mapper mirrored: oblique walls were inked at their mirror
        # image across the head's axis, and no loop ever closed on the twin.
        half = np.radians(FOV_DEG) / 2.0
        edges = np.linspace(-half, half, COLS + 1)
        centres = (edges[:-1] + edges[1:]) / 2.0
        self.directions = np.zeros((ZONES, 3))
        for row in range(ROWS):
            elevation = -centres[row]
            for col in range(COLS):
                azimuth = -centres[col]
                self.directions[row * COLS + col] = [
                    np.cos(elevation) * np.cos(azimuth),
                    np.cos(elevation) * np.sin(azimuth),
                    np.sin(elevation),
                ]

    def frame(self, data: mujoco.MjData) -> tuple[list[int], list[int]]:
        """One capture: distances in millimetres and a status per zone.

        Self-hits are reported, not filtered. A real sensor sees the duck's own beak when the beak is
        in front of it, and a simulator that quietly skipped its own geometry would hide exactly the
        kind of mounting problem this is here to catch.
        """
        origin = data.site_xpos[self.site].copy()
        rotation = data.site_xmat[self.site].reshape(3, 3)
        world = rotation @ self.directions.T  # (3, ZONES)

        distance_mm = [0] * ZONES
        status = [STATUS_NO_TARGET] * ZONES
        geom = np.zeros(1, dtype=np.int32)
        for zone in range(ZONES):
            hit = mujoco.mj_ray(
                self.model, data, origin, np.ascontiguousarray(world[:, zone]),
                None, 1, -1, geom,
            )
            if hit < 0 or hit > MAX_RANGE:
                continue
            # Noise that grows with range, as the datasheet has it: a few millimetres up close, a
            # couple of centimetres at the far end. Without it a simulated map is suspiciously
            # crisp, and every filter downstream goes untested.
            sigma = 0.003 + 0.02 * (hit / MAX_RANGE)
            measured = max(0.0, hit + self.random.normal(0.0, sigma))
            distance_mm[zone] = int(measured * 1000.0)
            status[zone] = STATUS_VALID
        return distance_mm, status
