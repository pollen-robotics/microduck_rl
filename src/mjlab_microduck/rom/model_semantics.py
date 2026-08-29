"""Code-owned physical capability checks shared by bundle and runtime preflight."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class _FootContact:
    ankle_joint: str
    ankle_body: str
    geom_name: str
    mesh_name: str


@dataclass(frozen=True)
class _WheelContact:
    ankle_joint: str
    ankle_body: str
    wheel_body: str


# These are the exported contact identities in robot_walk/allcollisions and their
# backlash variants. A differently named or typed ankle descendant is not a foot.
WALK_FOOT_CONTACTS = {
    "left": _FootContact(
        "left_ankle", "ankle_left", "left_foot_collision", "sole_left"
    ),
    "right": _FootContact(
        "right_ankle", "ankle_right", "right_foot_collision", "sole_right"
    ),
}

# These are the complete passive wheel topology exported by the checked-in roller
# models. Each wheel body directly owns its one named hinge and one active tire
# collision mesh; visual tire/rim meshes on the same body remain non-colliding.
ROLLER_WHEEL_CONTACTS = {
    "passive_LF_wheel": _WheelContact("left_ankle", "ankle_l_v1", "tire"),
    "passive_LR_wheel": _WheelContact("left_ankle", "ankle_l_v1", "tire_2"),
    "passive_RF_wheel": _WheelContact("right_ankle", "ankle_r_v1", "tire_3"),
    "passive_RR_wheel": _WheelContact("right_ankle", "ankle_r_v1", "tire_4"),
}


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, object_type, index) or ""


def _collision_masks_pair(model: mujoco.MjModel, first: int, second: int) -> bool:
    return bool(
        (int(model.geom_contype[first]) & int(model.geom_conaffinity[second]))
        or (int(model.geom_contype[second]) & int(model.geom_conaffinity[first]))
    )


def _mesh_name(model: mujoco.MjModel, geom_id: int) -> str:
    if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
        return ""
    return _name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[geom_id]))


def _walk_contact_geoms(model: mujoco.MjModel) -> tuple[tuple[int, ...], ...]:
    groups: list[tuple[int, ...]] = []
    for contact in WALK_FOOT_CONTACTS.values():
        ankle_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, contact.ankle_joint
        )
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, contact.ankle_body)
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom_name)
        if min(ankle_id, body_id, geom_id) < 0:
            return ()
        if (
            int(model.jnt_bodyid[ankle_id]) != body_id
            or int(model.geom_bodyid[geom_id]) != body_id
            or _mesh_name(model, geom_id) != contact.mesh_name
        ):
            return ()
        groups.append((geom_id,))
    return tuple(groups)


def _roller_contact_geoms(model: mujoco.MjModel) -> dict[str, int] | None:
    contacts: dict[str, int] = {}
    for wheel_joint, contact in ROLLER_WHEEL_CONTACTS.items():
        wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, wheel_joint)
        wheel_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, contact.wheel_body
        )
        ankle_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, contact.ankle_joint
        )
        ankle_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, contact.ankle_body
        )
        if min(wheel_id, wheel_body_id, ankle_id, ankle_body_id) < 0:
            return None
        if (
            model.jnt_type[wheel_id] != mujoco.mjtJoint.mjJNT_HINGE
            or int(model.jnt_bodyid[wheel_id]) != wheel_body_id
            or int(model.jnt_bodyid[ankle_id]) != ankle_body_id
            or int(model.body_parentid[wheel_body_id]) != ankle_body_id
            or int(model.body_jntnum[wheel_body_id]) != 1
            or int(model.body_jntadr[wheel_body_id]) != wheel_id
            or not np.allclose(
                model.jnt_axis[wheel_id],
                np.array([0.0, 0.0, 1.0]),
                atol=1e-7,
                rtol=0.0,
            )
        ):
            return None
        collision_geoms = [
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) == wheel_body_id
            and bool(model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
        ]
        if len(collision_geoms) != 1 or _mesh_name(model, collision_geoms[0]) != "tire":
            return None
        contacts[wheel_joint] = collision_geoms[0]
    return contacts


def _intended_contact_geom_groups(
    model: mujoco.MjModel,
) -> tuple[tuple[int, ...], ...]:
    walk = _walk_contact_geoms(model)
    if walk:
        return walk
    rollers = _roller_contact_geoms(model)
    if rollers is None:
        return ()
    # Every wheel, not merely one wheel per side, must be able to reach the floor.
    return tuple((rollers[name],) for name in ROLLER_WHEEL_CONTACTS)


def _mesh_minimum_world_z(
    model: mujoco.MjModel, data: mujoco.MjData, geom_id: int
) -> float:
    mesh_id = int(model.geom_dataid[geom_id])
    vertex_start = int(model.mesh_vertadr[mesh_id])
    vertex_count = int(model.mesh_vertnum[mesh_id])
    vertices = model.mesh_vert[vertex_start : vertex_start + vertex_count]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    world_vertices = data.geom_xpos[geom_id] + vertices @ rotation.T
    return float(np.min(world_vertices[:, 2]))


def _contacts_have_plausible_reset_reach(
    model: mujoco.MjModel,
    floor_id: int,
    contact_groups: tuple[tuple[int, ...], ...],
) -> bool:
    """Use exact reset kinematics to bound sole/tire reach instead of trunk gap."""
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    if trunk_id < 0:
        return False
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    floor_height = float(data.geom_xpos[floor_id, 2])
    trunk_height = float(data.xpos[trunk_id, 2])
    bottoms: list[float] = []
    for group in contact_groups:
        group_bottoms = [
            _mesh_minimum_world_z(model, data, geom_id) for geom_id in group
        ]
        bottom = min(group_bottoms)
        if (
            not np.isfinite([bottom, floor_height, trunk_height]).all()
            or bottom < floor_height - 0.03
            or bottom > floor_height + 0.03
            or min(float(data.geom_xpos[item, 2]) for item in group)
            > trunk_height - 0.04
        ):
            return False
        bottoms.append(bottom)
    return max(bottoms) - min(bottoms) <= 0.02


def _usable_flat_floor_ids(model: mujoco.MjModel) -> tuple[int, ...]:
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if (
        floor_id < 0
        or model.geom_type[floor_id] != mujoco.mjtGeom.mjGEOM_PLANE
        or int(model.geom_bodyid[floor_id]) != 0
    ):
        return ()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    normal_z = float(data.geom_xmat[floor_id].reshape(3, 3)[2, 2])
    if not np.isfinite(normal_z) or normal_z < 0.999:
        return ()
    contact_groups = _intended_contact_geom_groups(model)
    if (
        not contact_groups
        or not all(
            any(_collision_masks_pair(model, floor_id, geom_id) for geom_id in group)
            for group in contact_groups
        )
        or not _contacts_have_plausible_reset_reach(model, floor_id, contact_groups)
    ):
        return ()
    return (floor_id,)


def has_flat_world_floor(model: mujoco.MjModel) -> bool:
    """Prove the exact bilateral soles/tires plausibly reach the named world floor."""
    return bool(_usable_flat_floor_ids(model))


def has_exact_passive_roller_topology(model: mujoco.MjModel) -> bool:
    """Require the complete checked-in four-wheel topology and compatible floor."""
    floor_ids = _usable_flat_floor_ids(model)
    wheel_geoms = _roller_contact_geoms(model)
    if not floor_ids or wheel_geoms is None:
        return False
    passive_wheel_names = {
        name
        for joint_id in range(model.njnt)
        if (name := _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)).startswith(
            "passive_"
        )
        and "wheel" in name
    }
    if passive_wheel_names != set(ROLLER_WHEEL_CONTACTS):
        return False
    for wheel_name, geom_id in wheel_geoms.items():
        wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, wheel_name)
        if any(
            model.actuator_trnid[index, 0] == wheel_id for index in range(model.nu)
        ) or not any(
            _collision_masks_pair(model, geom_id, floor_id) for floor_id in floor_ids
        ):
            return False
    return True


def has_exact_position_actuator_topology(
    model: mujoco.MjModel, joint_names: tuple[str, ...]
) -> bool:
    if model.nu != len(joint_names):
        return False
    controlled_ids: set[int] = set()
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            return False
        actuator_ids = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
        if actuator_ids.size != 1:
            return False
        actuator_id = int(actuator_ids[0])
        if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
            return False
        gear = model.actuator_gear[actuator_id]
        if not np.isfinite(gear).all() or not np.allclose(
            gear,
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            atol=1e-12,
        ):
            return False
        if model.actuator_gaintype[actuator_id] != mujoco.mjtGain.mjGAIN_FIXED:
            return False
        if model.actuator_dyntype[actuator_id] != mujoco.mjtDyn.mjDYN_NONE:
            return False
        if model.actuator_biastype[actuator_id] != mujoco.mjtBias.mjBIAS_AFFINE:
            return False
        gain_parameters = model.actuator_gainprm[actuator_id]
        bias = model.actuator_biasprm[actuator_id]
        gain = gain_parameters[0]
        if (
            not np.isfinite(gain_parameters).all()
            or not np.isfinite(bias).all()
            or gain <= 0
            or not np.isclose(bias[0], 0.0, atol=1e-12, rtol=0.0)
            or not np.isclose(bias[1], -gain, atol=1e-12, rtol=0.0)
            or not np.isclose(bias[2], 0.0, atol=1e-12, rtol=0.0)
        ):
            return False
        if not model.actuator_ctrllimited[actuator_id]:
            return False
        low, high = model.actuator_ctrlrange[actuator_id]
        if not np.isfinite([low, high]).all() or low >= high:
            return False
        controlled_ids.add(joint_id)
    return {int(item) for item in model.actuator_trnid[:, 0]} == controlled_ids


def has_exact_deployment_frames(model: mujoco.MjModel) -> bool:
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    root_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
    )
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    if min(trunk_id, root_id, sensor_id) < 0:
        return False
    if (
        model.jnt_type[root_id] != mujoco.mjtJoint.mjJNT_FREE
        or model.jnt_bodyid[root_id] != trunk_id
        or model.sensor_type[sensor_id] != mujoco.mjtSensor.mjSENS_GYRO
        or model.sensor_dim[sensor_id] != 3
        or model.sensor_objtype[sensor_id] != mujoco.mjtObj.mjOBJ_SITE
    ):
        return False
    site_id = int(model.sensor_objid[sensor_id])
    return bool(
        model.site_bodyid[site_id] == trunk_id
        and np.allclose(
            model.site_quat[site_id],
            np.array([1.0, 0.0, 0.0, 0.0]),
            atol=1e-7,
        )
    )
