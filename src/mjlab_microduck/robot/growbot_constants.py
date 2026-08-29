"""Physics asset for the 25 cm, footed, 16-servo Growbot."""

from __future__ import annotations

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
        pos=(0.0, lateral_pos, 0.018),
        childclass="microduck",
    )
    upper.add_geom(
        name=f"{side}_upper_arm_collision",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=(0.0, 0.0, 0.0, 0.0, 0.0, -0.040),
        size=(0.010,),
        mass=0.018,
        group=3,
        rgba=(0.86, 0.86, 0.86, 1.0),
    )

    forearm = upper.add_body(
        name=f"{side}_forearm",
        pos=(0.0, 0.0, -0.040),
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
        fromto=(0.0, 0.0, 0.0, 0.0, 0.0, -0.038),
        size=(0.009,),
        mass=0.016,
        group=3,
        rgba=(0.88, 0.88, 0.88, 1.0),
    )
    forearm.add_geom(
        name=f"{side}_hand_collision",
        type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        pos=(0.0, 0.0, -0.047),
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


def get_growbot_spec() -> mujoco.MjSpec:
    """Return the stock footed model with a lightweight 2-DOF arm system."""
    spec = mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_XML))
    spec.modelname = "growbot_footed_16dof"

    _scale_explicit_body_mass(spec.body("trunk_base"), GROWBOT_TRUNK_MASS_KG)
    _scale_explicit_body_mass(spec.body("jaw_soft"), GROWBOT_HEAD_MASS_KG)

    _add_arm(spec, "left", 0.055)
    _add_arm(spec, "right", -0.055)
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
