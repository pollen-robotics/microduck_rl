"""Physics asset for the 25 cm, footed, 16-servo Growbot."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mjlab_microduck.robot.microduck_constants import (
    FULL_COLLISION,
    HOME_FRAME,
    MICRODUCK_ALLCOLLISIONS_XML,
    actuators,
)


GROWBOT_TRUNK_MASS_KG = 0.155
GROWBOT_HEAD_MASS_KG = 0.120
GROWBOT_VISUAL_DIR = Path(__file__).parent / "growbot" / "assets"

# Reference poses are expressed relative to trunk_base at the stock zero pose.
_VISUAL_BODY_REFERENCE = {
    "jaw_soft": ((0.0081, 0.0, 0.11561), (0.70710678, 0.0, 0.70710678, 0.0)),
    "left_upper_arm": ((0.0, 0.052, 0.035), (1.0, 0.0, 0.0, 0.0)),
    "left_forearm": ((0.0, 0.052, 0.010), (1.0, 0.0, 0.0, 0.0)),
    "right_upper_arm": ((0.0, -0.052, 0.035), (1.0, 0.0, 0.0, 0.0)),
    "right_forearm": ((0.0, -0.052, 0.010), (1.0, 0.0, 0.0, 0.0)),
}


def _scale_explicit_body_mass(body: mujoco.MjsBody, target_mass: float) -> None:
    """Change an explicit body's mass while preserving its inertia shape."""
    ratio = target_mass / float(body.mass)
    body.mass = target_mass
    body.fullinertia = [float(value) * ratio for value in body.fullinertia]


def _add_arm(spec: mujoco.MjSpec, side: str, lateral_pos: float) -> None:
    """Attach a fixed upper arm and a single actuated elbow to the trunk."""
    trunk = spec.body("trunk_base")
    upper = trunk.add_body(
        name=f"{side}_upper_arm",
        pos=(0.0, lateral_pos, 0.035),
        childclass="microduck",
    )
    upper.add_geom(
        name=f"{side}_upper_arm_collision",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=(0.0, 0.0, 0.0, 0.0, 0.0, -0.025),
        size=(0.010,),
        mass=0.018,
        group=3,
        rgba=(0.86, 0.86, 0.86, 1.0),
    )

    forearm = upper.add_body(
        name=f"{side}_forearm",
        pos=(0.0, 0.0, -0.025),
        childclass="microduck",
    )
    forearm.add_joint(
        name=f"{side}_elbow_pitch",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=(0.0, 1.0, 0.0),
        limited=True,
        range=(0.0, 2.35),
        damping=0.041,
        frictionloss=0.0048,
        armature=0.0018,
    )
    forearm.add_geom(
        name=f"{side}_forearm_collision",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=(0.0, 0.0, 0.0, 0.0, 0.0, -0.027),
        size=(0.009,),
        mass=0.016,
        group=3,
        rgba=(0.88, 0.88, 0.88, 1.0),
    )
    forearm.add_geom(
        name=f"{side}_hand_collision",
        type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        pos=(0.0, 0.0, -0.030),
        size=(0.013, 0.009, 0.016),
        mass=0.010,
        group=3,
        rgba=(0.72, 0.76, 0.80, 1.0),
    )
    spec.add_actuator(
        default=spec.find_default("chosen_actuator"),
        name=f"{side}_elbow_pitch",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        target=f"{side}_elbow_pitch",
    )


def _hide_replaced_geometry(spec: mujoco.MjSpec) -> None:
    """Keep the stock collision hulls active while hiding replaced visuals."""
    # Delete the stock Duck head visuals outright.  Alpha-only hiding is not
    # reliable in every native-viewer transparency mode and allowed the round
    # camera lens to remain visible over the Growbot face.
    for body_name in ("neck", "neck_pitch", "yaw_roll_motion", "jaw_soft"):
        for geom in tuple(spec.body(body_name).geoms):
            if geom.group == 2:
                spec.delete(geom)
            else:
                geom.rgba = (*geom.rgba[:3], 0.0)
    for body_name in (
        "left_upper_arm",
        "left_forearm",
        "right_upper_arm",
        "right_forearm",
    ):
        for geom in spec.body(body_name).geoms:
            geom.rgba = (*geom.rgba[:3], 0.0)


def _visual_pose(body_name: str, reference_pos: list[float]):
    body_pos, body_quat = _VISUAL_BODY_REFERENCE[body_name]
    dx = reference_pos[0] - body_pos[0]
    dy = reference_pos[1] - body_pos[1]
    dz = reference_pos[2] - body_pos[2]
    if body_name == "jaw_soft":
        # jaw_soft is rotated -90 degrees about Y in the stock zero pose.
        return (dz, dy, -dx), body_quat
    return (dx, dy, dz), body_quat


def _add_custom_visuals(spec: mujoco.MjSpec) -> None:
    """Bind the approved Blender head and compact arms to articulated bodies."""
    manifest = json.loads((GROWBOT_VISUAL_DIR / "manifest.json").read_text())
    for visual_name, entry in manifest.items():
        mesh_name = f"growbot_{visual_name}_mesh"
        spec.add_mesh(
            name=mesh_name,
            file=str(GROWBOT_VISUAL_DIR / entry["file"]),
        )
        pos, quat = _visual_pose(entry["body"], entry["reference_pos"])
        spec.body(entry["body"]).add_geom(
            name=f"growbot_{visual_name}_visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            pos=pos,
            quat=quat,
            rgba=entry["rgba"],
            contype=0,
            conaffinity=0,
            mass=0.0,
            group=2,
        )


def get_growbot_spec() -> mujoco.MjSpec:
    """Return the stock footed model with a lightweight 2-DOF arm system."""
    spec = mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_XML))
    spec.modelname = "growbot_footed_16dof"

    _scale_explicit_body_mass(spec.body("trunk_base"), GROWBOT_TRUNK_MASS_KG)
    _scale_explicit_body_mass(spec.body("jaw_soft"), GROWBOT_HEAD_MASS_KG)

    _add_arm(spec, "left", 0.052)
    _add_arm(spec, "right", -0.052)
    _hide_replaced_geometry(spec)
    _add_custom_visuals(spec)
    return spec


GROWBOT_HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={
        **HOME_FRAME.joint_pos,
        r"left_elbow_pitch": 0.35,
        r"right_elbow_pitch": 0.35,
    },
    joint_vel={".*": 0.0},
)


GROWBOT_ROBOT_CFG = EntityCfg(
    spec_fn=get_growbot_spec,
    init_state=GROWBOT_HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)
