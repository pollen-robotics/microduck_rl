"""Move-away: a CPU MuJoCo behavior demo built on a stock walking policy."""

from .controller import MoveAwayController, wrap_angle
from .gaze import GazeState, bearing_elevation, camera_axes, in_frustum
from .perception import PersonTracker, optical_frame_quat
from .runtime import (
    CTRL_HZ,
    DEFAULT_POSE,
    GYRO_SENSOR,
    SCENE_XML,
    build_observation,
    sensor_address,
)

__all__ = [
    "CTRL_HZ",
    "DEFAULT_POSE",
    "GYRO_SENSOR",
    "SCENE_XML",
    "GazeState",
    "MoveAwayController",
    "PersonTracker",
    "bearing_elevation",
    "build_observation",
    "camera_axes",
    "in_frustum",
    "optical_frame_quat",
    "sensor_address",
    "wrap_angle",
]
