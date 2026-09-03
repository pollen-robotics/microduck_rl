import mujoco

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_LAUNCHER_CFG,
    get_launcher_spec,
)


def test_launcher_spec_compiles_with_one_free_body():
    model = get_launcher_spec().compile()
    assert model.nbody == 2  # world + plate
    assert model.njnt == 1
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 1)
    assert name == "plate"


def test_plate_is_heavy_enough_to_ignore_the_duck():
    # The plate is re-prescribed every control step; mass only has to keep the
    # intra-step deviation negligible against an ~800 g duck pushing off.
    model = get_launcher_spec().compile()
    assert model.body_mass[1] >= 20.0


def test_plate_geom_is_a_box_wide_enough_to_stand_on():
    model = get_launcher_spec().compile()
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "plate_geom")
    assert gid >= 0
    assert model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX
    half_x, half_y, half_z = model.geom_size[gid]
    assert half_x >= 0.08 and half_y >= 0.08   # >= 16 cm wide: both feet fit
    assert half_z <= 0.02


def test_launcher_entity_cfg_has_no_articulation():
    # A prop, not a robot: no actuators, no articulation info (cf. the ball).
    assert MICRODUCK_LAUNCHER_CFG.articulation is None
