"""``com_height_target`` must bracket the height the robot actually stands at.

AGENTS.md rule 2: target heights are MEASURED on the model in use, never carried
across model revisions. The band that shipped with the roller env came from the
FOOTED recipe (stand z ~0.115) and was never re-measured when the roller model
arrived. Its ceiling sat 15 mm BELOW the roller standing height, so a robot
standing normally on its wheels scored 0 and could only collect the +2/step by
crouching — the term silently rewarded a flexed posture nobody asked for.

These tests lock the RULE, not the numbers. Whatever the models and the band
become, two things must hold for every env that uses the term:

  * the standing height measured ON THAT MODEL lies inside the band;
  * the whole ``reset_base`` spawn range lies inside the band, so an episode
    never starts out-of-band.

Re-export the CAD and change the stance by a centimetre and these fail, instead
of leaving the reward quietly pushing the policy into a crouch.

Envs are discovered, not hardcoded: any variant that builds on the roller recipe
and reuses ``com_height_target`` is covered as soon as its module exists. The
ice-skate variant inherits the band verbatim (it calls the roller factory and
never redefines the term), so it is the one that most needs this check — it is
listed here and skips cleanly while it is still out of tree.
"""

import importlib
import re

import mujoco
import numpy as np
import pytest

from mjlab_microduck.robot import microduck_constants as C

_SCENE_DIR = C.MICRODUCK_GROUNDCONTACT_ROLLERS_XML.parent

# (module, factory, scene). The scenes carry the terrain; the robot XMLs alone
# have no floor to stand on, and com_height_target reads a WORLD z, not the
# trunk-minus-foot stance height a kinematic check would give.
ENVS = {
    "rollers": (
        "mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg",
        "make_microduck_velocity_rollers_env_cfg",
        "scene_rollers.xml",
    ),
    "ice_skate": (
        "mjlab_microduck.tasks.microduck_velocity_ice_skate_env_cfg",
        "make_microduck_velocity_ice_skate_env_cfg",
        "scene_ice_skate.xml",
    ),
}

# Standing on four free wheels (or two blades) with a PD to HOME is not a stable
# equilibrium — the robot topples in ~0.6 s. Measure inside the upright window
# and assert the tilt, per AGENTS.md: a settle test that only records z reports
# fallen states as "resting fine".
_SETTLE_START, _SETTLE_END = 25, 100
_MAX_TILT_DEG = 5.0


def _load(name):
    """(cfg, scene path) for an env, skipping one that is not in the tree yet."""
    module_name, factory_name, scene = ENVS[name]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        pytest.skip(f"{name}: {module_name} is not in this tree")
    scene_path = _SCENE_DIR / scene
    if not scene_path.exists():
        pytest.skip(f"{name}: {scene_path.name} is not in this tree")
    return getattr(module, factory_name)(), scene_path


def _standing_trunk_z(scene_xml) -> float:
    """World z of trunk_base with ctrl held at HOME — what com_height_target reads."""
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    home = C.HOME_FRAME.joint_pos

    def home_value(joint_name: str | None) -> float | None:
        if joint_name is None:
            return None
        for pattern, value in home.items():
            if re.fullmatch(pattern, joint_name):
                return value
        return None

    for jnt in range(model.njnt):
        if model.jnt_type[jnt] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        value = home_value(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt))
        if value is not None:
            data.qpos[model.jnt_qposadr[jnt]] = value
    for act in range(model.nu):
        joint = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[act, 0]
        )
        value = home_value(joint)
        if value is not None:
            data.ctrl[act] = value

    heights, tilts = [], []
    for step in range(_SETTLE_END):
        mujoco.mj_step(model, data)
        if step + 1 >= _SETTLE_START:
            heights.append(data.xpos[trunk][2])
            z_axis = data.xmat[trunk].reshape(3, 3)[:, 2]
            tilts.append(np.degrees(np.arccos(np.clip(z_axis[2], -1.0, 1.0))))

    assert max(tilts) < _MAX_TILT_DEG, (
        f"{scene_xml.name}: robot tilted {max(tilts):.1f} deg during the settle "
        "window — the height below is a falling robot's, not a standing one's"
    )
    return float(np.median(heights))


def _band(cfg) -> tuple[float, float]:
    params = cfg.rewards["com_height_target"].params
    return params["target_height_min"], params["target_height_max"]


@pytest.mark.parametrize("name", sorted(ENVS))
def test_standing_height_is_inside_the_band(name):
    cfg, scene_xml = _load(name)
    low, high = _band(cfg)
    stand_z = _standing_trunk_z(scene_xml)
    assert low < stand_z < high, (
        f"{name}: measured standing height {stand_z:.4f} m is outside the "
        f"com_height_target band [{low}, {high}] — a robot standing normally "
        "scores 0 and the term pays only for leaving that posture"
    )


@pytest.mark.parametrize("name", sorted(ENVS))
def test_whole_spawn_range_is_inside_the_band(name):
    cfg, _ = _load(name)
    low, high = _band(cfg)
    spawn_min, spawn_max = cfg.events["reset_base"].params["pose_range"]["z"]
    assert low <= spawn_min and spawn_max <= high, (
        f"{name}: reset_base z range [{spawn_min}, {spawn_max}] is not contained "
        f"in the com_height_target band [{low}, {high}] — episodes would start "
        "out-of-band, so the term pays the policy to leave its spawn posture"
    )


@pytest.mark.parametrize("name", sorted(ENVS))
def test_band_leaves_room_to_flex_but_not_to_collapse(name):
    """The floor is a design choice, not an accident: it must sit BELOW the
    stance (a skating crouch stays paid) and well ABOVE the ground (collapsing
    must not)."""
    cfg, scene_xml = _load(name)
    low, _ = _band(cfg)
    flex_margin = _standing_trunk_z(scene_xml) - low
    assert 0.005 < flex_margin < 0.030, (
        f"{name}: the band floor allows {flex_margin * 1000:.0f} mm of flexion "
        "below the stance — under ~5 mm taxes the gait's own knee bend, over "
        "~30 mm lets the robot sit on the floor and still collect the reward"
    )
