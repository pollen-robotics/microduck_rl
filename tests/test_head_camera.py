"""The MJCF `head_camera` looks where the duck looks, on its side like the real one.

The camera element used to carry `quat="0 0 -1 0"`, which looks along the duck's -x: backwards,
into the head shell. `sim/camera.py` turned it at load time, so only the simulated duck saw a
sensible image and anyone else rendering from the model got grey.

The camera is deliberately NOT upright. The real head camera is mounted a quarter turn off and
`mediad --rotate 90` undoes it, so a simulated duck has to arrive on its side too. An upright
camera here would be turned a second time by the console.
"""

from pathlib import Path

import mujoco
import numpy as np
import pytest

ROBOT_DIR = Path(__file__).resolve().parents[1] / "src" / "mjlab_microduck" / "robot" / "microduck"
MODELS = sorted(ROBOT_DIR.glob("robot_*.xml"))
CONFIGS = sorted(ROBOT_DIR.glob("config_mjcf_*.json"))

# The `quat` that the config post-processing writes, and the one the generated MJCF carries.
CAMERA_QUAT = (0.0, 0.0, 0.0, 1.0)


def _camera_axes(path: Path):
    """The camera's right, up and view directions in the world, and the ToF site's forward."""
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    tof = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tof")
    assert cam >= 0 and tof >= 0
    axes = data.cam_xmat[cam].reshape(3, 3)  # columns: right, up, backwards
    return axes[:, 0], axes[:, 1], -axes[:, 2], data.site_xmat[tof].reshape(3, 3)[:, 0]


def test_the_robot_models_are_found():
    # A glob that matches nothing would make every other test here pass without checking anything.
    assert len(MODELS) == 8
    assert len(CONFIGS) == 7


@pytest.mark.parametrize("path", MODELS, ids=lambda p: p.name)
def test_head_camera_looks_along_the_duck_forward_axis(path):
    _, _, view, forward = _camera_axes(path)
    assert float(view @ forward) > 0.999, f"{path.name}: the head camera does not look forward"


@pytest.mark.parametrize("path", MODELS, ids=lambda p: p.name)
def test_head_camera_is_mounted_a_quarter_turn_off(path):
    right, up, _, _ = _camera_axes(path)
    world_down = np.array([0.0, 0.0, -1.0])
    assert float(right @ world_down) > 0.9, f"{path.name}: the frame is not on its side"
    assert abs(float(up[2])) < 0.2, f"{path.name}: the frame is upright, mediad would turn it twice"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_the_config_writes_the_camera_the_model_carries(path):
    # The MJCF is generated from the config, so regenerating must not bring the old camera back.
    # Read as text: onshape-to-robot accepts these files, but some are not strict JSON.
    steps = [line for line in path.read_text().splitlines() if "head_camera" in line]
    assert len(steps) == 1
    wanted = 'quat=\\"{} {} {} {}\\"'.format(*(int(v) for v in CAMERA_QUAT))
    assert wanted in steps[0], f"{path.name}: {steps[0].strip()}"
