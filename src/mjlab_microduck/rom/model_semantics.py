"""Code-owned physical capability checks shared by bundle and runtime preflight."""

from __future__ import annotations

import mujoco
import numpy as np

ROLLER_WHEEL_TO_ANKLE = {
    "passive_LF_wheel": "left_ankle",
    "passive_LR_wheel": "left_ankle",
    "passive_RF_wheel": "right_ankle",
    "passive_RR_wheel": "right_ankle",
}


def _collision_masks_pair(model: mujoco.MjModel, first: int, second: int) -> bool:
    return bool(
        (int(model.geom_contype[first]) & int(model.geom_conaffinity[second]))
        or (int(model.geom_contype[second]) & int(model.geom_conaffinity[first]))
    )


def _body_descends_from(model: mujoco.MjModel, body_id: int, ancestor_id: int) -> bool:
    current = body_id
    while current > 0 and current != ancestor_id:
        current = int(model.body_parentid[current])
    return current == ancestor_id


def _usable_flat_floor_ids(model: mujoco.MjModel) -> tuple[int, ...]:
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    root_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
    )
    if (
        trunk_id < 0
        or root_id < 0
        or model.jnt_type[root_id] != mujoco.mjtJoint.mjJNT_FREE
    ):
        return ()
    root_qpos = int(model.jnt_qposadr[root_id])
    trunk_height = float(model.qpos0[root_qpos + 2])
    contact_geom_groups: list[tuple[int, ...]] = []
    for ankle_name in ("left_ankle", "right_ankle"):
        ankle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ankle_name)
        if ankle_id < 0:
            return ()
        ankle_body = int(model.jnt_bodyid[ankle_id])
        contact_geoms = tuple(
            index
            for index in range(model.ngeom)
            if _body_descends_from(model, int(model.geom_bodyid[index]), ankle_body)
            and bool(model.geom_contype[index] or model.geom_conaffinity[index])
        )
        if not contact_geoms:
            return ()
        contact_geom_groups.append(contact_geoms)
    floors: list[int] = []
    for index in range(model.ngeom):
        if (
            model.geom_type[index] != mujoco.mjtGeom.mjGEOM_PLANE
            or model.geom_bodyid[index] != 0
        ):
            continue
        quaternion = model.geom_quat[index]
        normal_z = 1.0 - 2.0 * (quaternion[1] ** 2 + quaternion[2] ** 2)
        floor_height = float(model.geom_pos[index, 2])
        gap = trunk_height - floor_height
        if (
            not np.isfinite([normal_z, floor_height, gap]).all()
            or normal_z < 0.999
            or gap < 0.01
            or gap > 0.5
            or not all(
                any(
                    _collision_masks_pair(model, index, contact_geom)
                    for contact_geom in contact_geoms
                )
                for contact_geoms in contact_geom_groups
            )
        ):
            continue
        floors.append(index)
    return tuple(floors)


def has_flat_world_floor(model: mujoco.MjModel) -> bool:
    """Prove a reachable horizontal world plane can collide with the robot."""
    return bool(_usable_flat_floor_ids(model))


def has_exact_passive_roller_topology(model: mujoco.MjModel) -> bool:
    """Require four unactuated hinge wheels below the correct ankle with collision."""
    floor_ids = _usable_flat_floor_ids(model)
    if not floor_ids:
        return False
    for wheel_name, ankle_name in ROLLER_WHEEL_TO_ANKLE.items():
        wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, wheel_name)
        ankle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ankle_name)
        if wheel_id < 0 or ankle_id < 0:
            return False
        if model.jnt_type[wheel_id] != mujoco.mjtJoint.mjJNT_HINGE:
            return False
        if any(model.actuator_trnid[index, 0] == wheel_id for index in range(model.nu)):
            return False
        ankle_body = int(model.jnt_bodyid[ankle_id])
        wheel_body = int(model.jnt_bodyid[wheel_id])
        ancestor = wheel_body
        while ancestor > 0 and ancestor != ankle_body:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor != ankle_body:
            return False
        wheel_collision = any(
            model.geom_bodyid[index] == wheel_body
            and model.geom_type[index]
            in {mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_MESH}
            and any(
                _collision_masks_pair(model, index, floor_id) for floor_id in floor_ids
            )
            for index in range(model.ngeom)
        )
        if not wheel_collision:
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
        if not np.allclose(
            model.actuator_gear[actuator_id],
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
        gain = model.actuator_gainprm[actuator_id, 0]
        bias = model.actuator_biasprm[actuator_id]
        if (
            gain <= 0
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
