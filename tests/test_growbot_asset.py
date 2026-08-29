import mujoco

from mjlab_microduck.robot.growbot_constants import get_growbot_spec


EXPECTED_MICRODUCK_JOINTS = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]


def _joint_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    ]


def _articulated_joint_names(model: mujoco.MjModel) -> list[str]:
    return [
        name
        for index, name in enumerate(_joint_names(model))
        if model.jnt_type[index] != mujoco.mjtJoint.mjJNT_FREE
    ]


def test_growbot_joint_order_and_count():
    model = get_growbot_spec().compile()
    names = _articulated_joint_names(model)

    assert names[:14] == EXPECTED_MICRODUCK_JOINTS
    assert names[14:] == ["left_elbow_pitch", "right_elbow_pitch"]
    assert model.njnt == 17
    assert model.nu == 16


def test_growbot_is_footed_and_has_arm_collisions():
    model = get_growbot_spec().compile()

    assert all("wheel" not in name for name in _joint_names(model))
    for name in (
        "left_upper_arm_collision",
        "left_forearm_collision",
        "left_hand_collision",
        "right_upper_arm_collision",
        "right_forearm_collision",
        "right_hand_collision",
        "left_foot_collision",
        "right_foot_collision",
    ):
        assert model.geom(name).id >= 0


def test_growbot_uses_exported_custom_head_and_arm_visuals():
    model = get_growbot_spec().compile()

    for name in (
        "growbot_head_visual",
        "growbot_eyes_visual",
        "growbot_mesh_0_001_visual",
        "growbot_arm_lower_001_visual",
        "growbot_hand_r_001_visual",
        "growbot_mesh_0_004_visual",
        "growbot_arm_lower_visual",
        "growbot_hand_r_visual",
    ):
        geom = model.geom(name)
        assert geom.id >= 0
        assert model.geom_contype[geom.id] == 0
        assert model.geom_conaffinity[geom.id] == 0

    # Training still uses simple collision hulls, but they are not rendered on
    # top of the approved Blender meshes.
    collision = model.geom("left_forearm_collision")
    assert model.geom_rgba[collision.id, 3] == 0.0
    assert model.geom_contype[collision.id] != 0

    replaced_head_body_ids = {
        model.body(name).id
        for name in ("neck", "neck_pitch", "yaw_roll_motion", "jaw_soft")
    }
    stock_head_mesh_names = {
        model.mesh(model.geom_dataid[index]).name
        for index in range(model.ngeom)
        if model.geom_bodyid[index] in replaced_head_body_ids
        and model.geom_type[index] == mujoco.mjtGeom.mjGEOM_MESH
        and not model.mesh(model.geom_dataid[index]).name.startswith("growbot_")
    }
    assert stock_head_mesh_names <= {"top_head_shell", "jaw", "bottom_head_shell"}


def test_growbot_mass_stays_inside_initial_design_range():
    model = get_growbot_spec().compile()
    assert 0.50 <= float(model.body_mass.sum()) <= 0.80
