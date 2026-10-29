"""Body-protocol integration of training-matched CPU motors, without a policy."""

import subprocess
import sys

import mujoco
import numpy as np
import pytest

from mjlab_microduck.sim.body_server import (
    HOME_POSE,
    HOME_TRUNK_Z,
    JOINT_NAMES,
    MOUTH_INDEX,
    SCENES,
    SPACING,
    Body,
    World,
)
from mjlab_microduck.sim.visualization.scenes import VELSTAND_SCENE, resolve_scene

SCENE = VELSTAND_SCENE
WIRED_MOTORS = [index for index in range(len(JOINT_NAMES)) if index != MOUTH_INDEX]


def make_world(count=1, *, limp=False):
    world = World(SCENE, count=count, actuator_model="bam")
    for index in range(count):
        body = Body(world, index, limp=limp)
        body.place(None, HOME_TRUNK_Z, offset_y=index * SPACING)
        world.bodies.append(body)
    mujoco.mj_forward(world.model, world.data)
    return world


@pytest.fixture(scope="module")
def training_model():
    # Training Entity is the oracle only in tests. Production body construction
    # must not import Torch/mjlab to edit the robot's physical specification.
    from mjlab.entity import Entity
    from mjlab.sim import MujocoCfg

    from mjlab_microduck.robot.microduck_constants import (
        MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG,
    )

    model = Entity(MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG).spec.compile()
    MujocoCfg(timestep=0.005, iterations=30, ls_iterations=50).apply(model)
    return model


def test_bam_scene_matches_training_robot_geometry_and_motor_settings(training_model):
    world = make_world()
    model = world.model
    oracle = training_model
    assert (model.nq, model.nv, model.nu) == (oracle.nq, oracle.nv, oracle.nu)
    assert model.ngeom == oracle.ngeom + 1  # The separate scene adds its floor.
    robot = np.flatnonzero(model.geom_bodyid != 0)
    oracle_robot = np.flatnonzero(oracle.geom_bodyid != 0)
    assert len(robot) == len(oracle_robot) == 140
    for field in (
        "geom_type",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_contype",
        "geom_conaffinity",
        "geom_condim",
        "geom_priority",
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "geom_margin",
        "geom_gap",
    ):
        np.testing.assert_allclose(
            getattr(model, field)[robot],
            getattr(oracle, field)[oracle_robot],
            err_msg=field,
        )
    for field in (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "jnt_axis",
        "jnt_pos",
        "jnt_range",
        "dof_armature",
        "dof_damping",
        "dof_frictionloss",
        "dof_solref",
        "dof_solimp",
        "actuator_gainprm",
        "actuator_gaintype",
        "actuator_biastype",
        "actuator_gear",
        "actuator_forcerange",
        "actuator_forcelimited",
    ):
        np.testing.assert_allclose(
            getattr(model, field), getattr(oracle, field), err_msg=field
        )
    for field in (
        "timestep",
        "integrator",
        "solver",
        "cone",
        "jacobian",
        "iterations",
        "ls_iterations",
        "tolerance",
        "ls_tolerance",
        "ccd_iterations",
        "impratio",
        "gravity",
    ):
        np.testing.assert_allclose(
            getattr(model.opt, field),
            getattr(oracle.opt, field),
            err_msg=f"opt.{field}",
        )


def test_bam_sensors_use_output_encoder_and_keep_the_15_joint_wire_order():
    world = make_world()
    body = world.bodies[0]
    servo = np.linspace(-0.1, 0.1, 14)
    passive = np.linspace(0.001, 0.014, 14)
    world.data.qpos[body.bam.servo_qpos] = servo
    world.data.qpos[body.bam.backlash_qpos] = passive
    world.data.actuator_force[body.actuator_slice] = np.arange(14) / 10
    state = body.sensors()
    assert (
        len(state["positions"])
        == len(state["velocities"])
        == len(state["currents_ma"])
        == 15
    )
    np.testing.assert_allclose(
        np.array(state["positions"])[WIRED_MOTORS], servo + passive
    )
    np.testing.assert_allclose(
        np.array(state["currents_ma"])[WIRED_MOTORS], np.arange(14) * 10
    )
    assert state["positions"][MOUTH_INDEX] == 0
    assert state["velocities"][MOUTH_INDEX] == 0
    assert state["currents_ma"][MOUTH_INDEX] == 0
    targets = np.linspace(-0.2, 0.2, 15)
    targets[MOUTH_INDEX] = 100  # No actuator exists for this protocol-only slot.
    body.set_targets(targets.tolist())
    np.testing.assert_allclose(body.bam.desired, targets[WIRED_MOTORS])
    np.testing.assert_array_equal(
        world.data.ctrl, 0
    )  # Angles were not written as motor Nm.


def test_sensor_velocity_delay_advances_only_on_physics_control_ticks():
    world = make_world(limp=True)
    body = world.bodies[0]
    world.model.opt.gravity[:] = 0
    world.data.qpos[body.trunk + 2] = 1  # Exclude floor contact from this clock test.
    world.data.qvel[body.bam.servo_dof] = 3
    world.data.qvel[body.bam.backlash_dof] = 0.1
    world.step(3)
    np.testing.assert_array_equal(body.sensors()["velocities"], 0)
    world.step(1)
    first_sample = body.bam.encoder_velocities().copy()
    assert np.linalg.norm(first_sample) > 0.01
    # Even many Studio/robotd reads cannot consume a delay sample.
    for _ in range(10):
        np.testing.assert_array_equal(body.sensors()["velocities"], 0)
    world.step(3)
    np.testing.assert_array_equal(body.sensors()["velocities"], 0)
    world.step(1)
    second_sample = body.bam.encoder_velocities().copy()
    for _ in range(10):
        np.testing.assert_allclose(
            np.array(body.sensors()["velocities"])[WIRED_MOTORS], first_sample
        )
    world.step(4)
    np.testing.assert_allclose(
        np.array(body.sensors()["velocities"])[WIRED_MOTORS], second_sample
    )


def test_placement_clears_backlash_and_old_velocity_samples():
    world = make_world(limp=True)
    body = world.bodies[0]
    world.model.opt.gravity[:] = 0
    world.data.qpos[body.trunk + 2] = 1
    world.data.qvel[body.bam.servo_dof] = 3
    world.data.qvel[body.bam.backlash_dof] = 0.1
    world.step(8)
    assert np.linalg.norm(body.sensors()["velocities"]) > 0.01
    world.data.qpos[body.bam.backlash_qpos] = 0.01
    world.data.qvel[body.bam.backlash_dof] = 1
    body.place(None, HOME_TRUNK_Z, offset_y=0)
    np.testing.assert_array_equal(world.data.qpos[body.bam.backlash_qpos], 0)
    np.testing.assert_array_equal(world.data.qvel[body.bam.backlash_dof], 0)
    np.testing.assert_array_equal(body.sensors()["velocities"], 0)
    np.testing.assert_allclose(body.sensors()["positions"], HOME_POSE)
    np.testing.assert_allclose(body.bam.desired, np.array(HOME_POSE)[WIRED_MOTORS])
    np.testing.assert_array_equal(world.data.ctrl, 0)


def test_held_body_restores_passive_joints_as_well_as_the_servo_pose():
    world = make_world()
    body = world.bodies[0]
    assert not body.released
    world.data.qpos[body.bam.backlash_qpos] = 0.01
    world.data.qvel[body.bam.backlash_dof] = 2
    world.data.qpos[body.qpos_adr] += 0.02
    world.data.qpos[body.trunk] += 0.03
    world.step(4)
    np.testing.assert_array_equal(world.data.qpos[body.bam.backlash_qpos], 0)
    np.testing.assert_array_equal(world.data.qvel[body.bam.backlash_dof], 0)
    np.testing.assert_allclose(body.sensors()["positions"], HOME_POSE)
    assert body.sensors()["trunk"] == [0, 0, HOME_TRUNK_Z]


def test_enable_clears_stale_targets_even_for_the_initial_held_body():
    world = make_world()
    body = world.bodies[0]
    gainprm = world.model.actuator_gainprm.copy()
    assert body.torque_on and not body.released
    body.set_targets([0.8] * 15)
    body.set_torque(True)
    assert body.released
    np.testing.assert_allclose(body.bam.desired, np.array(HOME_POSE)[WIRED_MOTORS])
    np.testing.assert_array_equal(world.data.ctrl, 0)
    body.set_gain(123)
    assert body.bam.bam_model.actuator.kp == 123
    np.testing.assert_array_equal(world.model.actuator_gainprm, gainprm)
    body.set_torque(False)
    body.set_targets([0.8] * 15)
    np.testing.assert_array_equal(body.bam.update(), 0)
    body.set_torque(True)
    np.testing.assert_allclose(body.bam.desired, np.array(HOME_POSE)[WIRED_MOTORS])
    np.testing.assert_array_equal(world.data.ctrl, 0)


def test_two_ducks_keep_encoder_targets_gains_and_torque_separate():
    world = make_world(count=2)
    left, right = world.bodies
    for field in (
        "actuator_ids",
        "servo_qpos",
        "servo_dof",
        "backlash_qpos",
        "backlash_dof",
    ):
        assert not set(getattr(left.bam, field)) & set(getattr(right.bam, field))
    assert all(name.startswith("d1_") for name in right.bam.actuator_names)
    a, b = np.linspace(-0.2, 0.2, 15), np.linspace(0.3, -0.3, 15)
    left.set_targets(a.tolist())
    right.set_targets(b.tolist())
    np.testing.assert_allclose(left.bam.desired, a[WIRED_MOTORS])
    np.testing.assert_allclose(right.bam.desired, b[WIRED_MOTORS])
    left.set_gain(111)
    assert left.bam.bam_model.actuator.kp == 111
    assert right.bam.bam_model.actuator.kp == 200
    left.set_torque(False)
    assert not left.bam.torque_enabled and right.bam.torque_enabled
    before = right.sensors()
    world.data.qpos[left.bam.servo_qpos] += 0.03
    world.data.qpos[left.bam.backlash_qpos] = 0.01
    np.testing.assert_array_equal(right.sensors()["positions"], before["positions"])
    np.testing.assert_allclose(left.sensors()["trunk"], [0, 0, HOME_TRUNK_Z])
    np.testing.assert_allclose(right.sensors()["trunk"], [0, SPACING, HOME_TRUNK_Z])


def test_default_world_keeps_legacy_xml_pd_behavior():
    world = World(SCENES / "scene.xml")
    body = Body(world, 0)
    body.place(None, HOME_TRUNK_Z, offset_y=0)
    world.bodies.append(body)
    assert world.actuator_model == "xml" and body.bam is None
    assert world.model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
    assert world.model.opt.iterations == 100
    np.testing.assert_allclose(
        world.data.ctrl[body.actuator_slice], np.array(HOME_POSE)[WIRED_MOTORS]
    )
    world.data.qvel[body.qvel_adr] = np.arange(14) / 10
    np.testing.assert_allclose(
        np.array(body.sensors()["velocities"])[WIRED_MOTORS], np.arange(14) / 10
    )


def test_extension_scene_compiles_with_shared_robot_assets_and_keyframes():
    from mjlab_microduck.sim.body_server import build_world, pose_table

    # The moved scene references the existing model/meshes instead of copying
    # them into the extension or leaving an XML wrapper in the robot directory.
    model = build_world(VELSTAND_SCENE, 1)
    assert (model.nq, model.nv, model.nu) == (35, 34, 14)
    assert model.camera("head_camera").id >= 0
    pose, trunk_z = pose_table(VELSTAND_SCENE, "STAND")
    assert trunk_z == pytest.approx(0.12)
    assert set(pose) == {JOINT_NAMES[index] for index in WIRED_MOTORS}
    np.testing.assert_allclose(
        [pose[JOINT_NAMES[index]] for index in WIRED_MOTORS],
        np.array(HOME_POSE)[WIRED_MOTORS],
        atol=1e-4,
    )


def test_extension_scene_resolution_preserves_names_and_explicit_paths(tmp_path):
    assert resolve_scene(SCENES / "scene_velstand.xml") == VELSTAND_SCENE
    assert resolve_scene("scene_velstand.xml") == VELSTAND_SCENE
    custom = tmp_path / "scene_velstand.xml"
    custom.write_text("<mujoco/>")
    assert resolve_scene(custom) == custom


def test_default_xml_body_does_not_load_optional_physics_or_training_stack():
    # Use a clean process: this module's training-model oracle imports Torch.
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from mjlab_microduck.sim.body_server import Body, World, SCENES; "
                "world = World(SCENES / 'scene.xml'); "
                "body = Body(world, 0); "
                "body.place(None, 0.125, 0); "
                "world.bodies.append(body); "
                "world.step(4); "
                "assert len(body.sensors()['positions']) == 15; "
                "assert body.bam is None; "
                "assert all(name not in sys.modules for name in "
                "('torch', 'mjlab', 'bam', 'mjlab_microduck.sim.visualization.bam', "
                "'mjlab_microduck.sim.visualization.physics'))"
            ),
        ],
        check=True,
        timeout=15,
    )


def test_bam_conversion_preserves_non_robot_scene_contacts(monkeypatch):
    original = mujoco.MjSpec.from_file

    class ExtendedSpec:
        @staticmethod
        def from_file(path):
            spec = original(path)
            spec.worldbody.add_geom(
                name="obstacle_collision",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[1, 0, 0.1],
                size=[0.1, 0.1, 0.1],
                condim=4,
                friction=[0.7, 0.02, 0.003],
            )
            return spec

    monkeypatch.setattr(mujoco, "MjSpec", ExtendedSpec)
    from mjlab_microduck.sim.body_server import build_world

    model = build_world(SCENE, 1, actuator_model="bam")
    prop = model.geom("obstacle_collision")
    assert int(prop.condim[0]) == 4
    assert prop.friction == pytest.approx([0.7, 0.02, 0.003])


def test_bam_conversion_preserves_motorized_scenery_actuators(monkeypatch):
    original = mujoco.MjSpec.from_file

    def scene_with_motorized_prop(path):
        spec = original(str(path))
        prop = spec.worldbody.add_body(name="prop", pos=[1, 0, 0.2])
        prop.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1], mass=0.2)
        prop.add_joint(
            name="prop_hinge",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            armature=0.012,
            damping=0.04,
            frictionloss=0.13,
        )
        servo = spec.add_actuator(
            name="prop_servo", target="prop_hinge", trntype=mujoco.mjtTrn.mjTRN_JOINT
        )
        servo.set_to_position(kp=8.0, kv=0.4)
        servo.ctrllimited = True
        servo.ctrlrange = [-0.5, 0.5]
        servo.forcelimited = True
        servo.forcerange = [-1.25, 1.25]
        return spec

    oracle = scene_with_motorized_prop(SCENE).compile()

    class ExtendedSpec:
        from_file = staticmethod(scene_with_motorized_prop)

    monkeypatch.setattr(mujoco, "MjSpec", ExtendedSpec)
    from mjlab_microduck.sim.body_server import build_world

    model = build_world(SCENE, 1, actuator_model="bam")
    prop, expected = model.actuator("prop_servo"), oracle.actuator("prop_servo")
    for field in (
        "gainprm",
        "biasprm",
        "gear",
        "gaintype",
        "biastype",
        "ctrllimited",
        "ctrlrange",
        "forcelimited",
        "forcerange",
    ):
        np.testing.assert_allclose(getattr(prop, field), getattr(expected, field))
    dof, expected_dof = (
        model.joint("prop_hinge").dofadr,
        oracle.joint("prop_hinge").dofadr,
    )
    for field in ("dof_armature", "dof_damping", "dof_frictionloss"):
        np.testing.assert_allclose(
            getattr(model, field)[dof], getattr(oracle, field)[expected_dof]
        )
    # The prop remains a position servo, while robot joints become BAM motors.
    assert prop.biastype[0] == mujoco.mjtBias.mjBIAS_AFFINE
    duck = model.actuator("left_hip_yaw")
    assert duck.gainprm[0] == 1
    assert duck.biastype[0] == mujoco.mjtBias.mjBIAS_NONE
