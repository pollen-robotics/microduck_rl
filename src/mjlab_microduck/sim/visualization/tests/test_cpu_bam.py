"""CPU motor correctness, independent of locomotion-policy performance."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from mjlab_microduck.sim.visualization.bam import (
    STIFF_SOLIMP_FRICTION,
    STIFF_SOLREF_FRICTION,
    CpuBamController,
    convert_to_motors,
    dof_friction_force,
    friction_budget,
)

XML = """
<mujoco>
  <option timestep=".005" gravity="0 0 0"/>
  <default><geom contype="0" conaffinity="0"/></default>
  <worldbody>
    <body name="root">
      <freejoint/>
      <geom type="sphere" size=".03" mass="1"/>
      <body name="rotor">
        <joint name="servo" type="hinge" frictionloss=".1"/>
        <geom type="sphere" size=".02" mass=".1"/>
        <body name="link">
          <joint name="passive_servo_backlash" type="hinge" frictionloss=".1"
                 range="-1 1"/>
          <geom type="sphere" size=".01" mass=".01"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator><position name="motor" joint="servo" kp=".55"/></actuator>
</mujoco>
"""


def make_sim(*, delay_steps=(0, 0), **kwargs):
    spec = mujoco.MjSpec.from_string(XML)
    convert_to_motors(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    controller = CpuBamController(
        model, data, ("motor",), delay_steps=delay_steps, **kwargs
    )
    mujoco.mj_forward(model, data)
    return model, data, controller


@pytest.mark.parametrize("variant", ["m1", "m2", "m3", "m4", "m5", "m6"])
def test_friction_matches_training_oracle_for_all_load_signs(variant):
    import torch
    from bam.mjlab import BamActuator
    from bam.model import load_model

    bam = load_model(motor_name="xl330", model=variant)
    # Includes equal magnitudes, same-sign loads and zero-speed Stribeck,
    # where the original CPU M6 quadratic expression diverges from training.
    motor = np.array([0, 0.4, 0.4, -0.4, -0.4, 0.2, -0.2, 0.4], dtype=float)
    external = np.array([0, 0.2, -0.2, 0.2, -0.2, 0.2, -0.2, 0.7], dtype=float)
    velocity = np.array([0, 0, 0, 0, 0, 0.1, -0.1, 3], dtype=float)
    tensor = lambda values: torch.as_tensor(values, dtype=torch.float64)
    speed = tensor(velocity)
    stribeck = (
        torch.exp(-((speed.abs() / bam.dtheta_stribeck.value) ** bam.alpha.value))
        if bam.stribeck
        else torch.zeros_like(speed)
    )
    oracle = BamActuator._compute_friction_budget(
        SimpleNamespace(_bam_model=bam), tensor(motor), tensor(external), stribeck
    )
    np.testing.assert_allclose(
        friction_budget(bam, motor, external, velocity), oracle.numpy(), rtol=1e-12
    )


def test_dof_friction_is_not_indexed_as_a_joint():
    model, data, controller = make_sim()
    servo = int(controller.servo_dof[0])
    passive = int(controller.backlash_dof[0])
    assert servo != model.joint("servo").id
    model.dof_frictionloss[[servo, passive]] = 0.1
    mujoco.mj_forward(model, data)
    # Use real MuJoCo constraint rows, with controlled previous-solve forces.
    mask = data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)
    rows = np.flatnonzero(mask)
    assert set(data.efc_id[rows]) == {servo, passive}
    data.efc_force[rows] = [0.11, -0.03]
    force = dof_friction_force(model, data)
    assert force[servo] == pytest.approx(0.11)
    assert force[passive] == pytest.approx(-0.03)
    assert np.count_nonzero(force) == 2


def test_motor_conversion_preserves_training_limits_and_constraint_settings():
    model, _, controller = make_sim()
    bam = controller.bam_model
    dof = controller.servo_dof
    assert model.actuator_gainprm[0, 0] == 1
    assert model.actuator_biastype[0] == mujoco.mjtBias.mjBIAS_NONE
    np.testing.assert_allclose(
        model.actuator_forcerange[0],
        [-8.2 * bam.kt.value / bam.R.value, 8.2 * bam.kt.value / bam.R.value],
    )
    np.testing.assert_allclose(
        model.dof_armature[dof], bam.actuator.get_extra_inertia()
    )
    np.testing.assert_allclose(model.dof_solref[dof][0], STIFF_SOLREF_FRICTION)
    np.testing.assert_allclose(model.dof_solimp[dof][0], STIFF_SOLIMP_FRICTION)
    assert bam.actuator.max_current == pytest.approx(1.75)


@pytest.mark.parametrize("target", [-2.0, 2.0])
@pytest.mark.parametrize("velocity", [-30.0, -5.0, 0.0, 5.0, 30.0])
def test_firmware_current_limiter_matches_training_during_drive_and_braking(
    target, velocity
):
    import torch
    from bam.actuator import TorchBackend

    from mjlab_microduck.actuator.friction_dr_bam import BacklashEncoderBamActuator
    from mjlab_microduck.robot.microduck_constants import backlash_actuators

    # The actual training actuator loads its own motor model. Compare its shared
    # Torch voltage-control path, including cases where back-EMF exceeds the supply;
    # an unconditional torque clamp would incorrectly change those cases.
    oracle = BacklashEncoderBamActuator(
        backlash_actuators, SimpleNamespace(), [0], ["servo"]
    )._bam_model.actuator
    assert oracle.max_current == pytest.approx(1.75)
    oracle.backend = TorchBackend()
    oracle.kp, oracle.vin = 200, 7.4
    _, data, controller = make_sim(voltage_drop_gain=0)
    controller.set_targets([target])
    data.qvel[controller.servo_dof] = velocity
    tensor = lambda value: torch.tensor([value], dtype=torch.float64)
    expected_voltage = oracle.compute_control(
        tensor(target), tensor(0), tensor(velocity), 0.005
    )
    expected_torque = oracle.compute_torque(
        expected_voltage, True, tensor(0), tensor(velocity)
    )
    np.testing.assert_allclose(controller.update(), expected_torque.numpy(), rtol=1e-12)


def test_custom_motor_firmware_current_limit_is_preserved():
    from bam.model import load_model

    bam = load_model(motor_name="xl330", model="m6")
    bam.actuator.max_current = 1.0
    _, _, controller = make_sim(bam_model=bam)
    controller.set_targets([2.0])
    assert controller.bam_model.actuator.max_current == 1.0
    np.testing.assert_allclose(controller.update(), [bam.kt.value])


def test_encoder_feedback_includes_backlash_but_motor_velocity_does_not():
    model, data, controller = make_sim(voltage_drop_gain=0)
    data.qpos[controller.servo_qpos] = 0.1
    data.qpos[controller.backlash_qpos] = 0.017
    data.qvel[controller.servo_dof] = 0.4
    data.qvel[controller.backlash_dof] = 1.2
    controller.reset()
    controller.set_target("motor", 0.127)
    actuator = controller.bam_model.actuator
    signal = actuator.compute_control(
        np.array([0.127]), np.array([0.117]), np.array([0.4]), 0.005
    )
    expected = actuator.compute_torque(signal, True, np.array([0.117]), np.array([0.4]))
    np.testing.assert_allclose(controller.encoder_positions(), [0.117])
    np.testing.assert_allclose(controller.encoder_velocities(), [1.6])
    np.testing.assert_allclose(controller.update(), expected)
    np.testing.assert_allclose(data.ctrl[controller.actuator_ids], expected)
    assert model.actuator_gainprm[0, 0] == 1


def test_target_delay_is_counted_in_physics_steps_and_reset_discards_old_targets():
    model, data, controller = make_sim(delay_steps=(3, 3), voltage_drop_gain=0)
    controller.set_targets([0.01])
    for _ in range(3):
        np.testing.assert_allclose(controller.update(), 0, atol=1e-12)
        mujoco.mj_step(model, data)
    assert controller.update()[0] > 0
    mujoco.mj_resetData(model, data)
    controller.reset()
    assert controller.last_time == 0
    for _ in range(7):
        np.testing.assert_allclose(controller.update(), 0, atol=1e-12)
        mujoco.mj_step(model, data)


def test_torque_reenable_discards_stale_goal_and_gain_never_rescales_motors():
    model, data, controller = make_sim(voltage_drop_gain=0)
    gainprm = model.actuator_gainprm.copy()
    controller.set_targets([0.01])
    full = controller.update()[0]
    controller.set_gain(100)
    half = controller.update()[0]
    assert half == pytest.approx(full / 2)
    controller.set_torque(False)
    np.testing.assert_array_equal(controller.update(), [0])
    controller.set_targets([0.5])
    controller.set_torque(True)
    np.testing.assert_allclose(controller.update(), 0, atol=1e-15)
    np.testing.assert_array_equal(model.actuator_gainprm, gainprm)
    np.testing.assert_array_equal(data.ctrl[controller.actuator_ids], [0])


def test_voltage_drop_uses_previous_torque_and_reset_clears_the_load():
    model, data, controller = make_sim()
    controller.set_targets([0.01])
    initial = controller.update()[0]
    loaded = controller.update()[0]
    assert loaded == pytest.approx(initial * (7.4 - 0.1 * abs(initial)) / 7.4)
    assert controller.bam_model.actuator.vin == 7.4
    mujoco.mj_resetData(model, data)
    controller.reset()
    controller.set_targets([0.01])
    assert controller.update()[0] == pytest.approx(initial)


def test_prefixed_duck_finds_its_own_backlash_joint():
    xml = (
        XML.replace('name="servo"', 'name="d1_servo"')
        .replace('joint="servo"', 'joint="d1_servo"')
        .replace('name="passive_servo_backlash"', 'name="d1_passive_servo_backlash"')
        .replace('name="motor"', 'name="d1_motor"')
    )
    spec = mujoco.MjSpec.from_string(xml)
    convert_to_motors(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    controller = CpuBamController(model, data, ("d1_motor",), prefix="d1_")
    data.qpos[controller.servo_qpos] = 0.12
    data.qpos[controller.backlash_qpos] = 0.01
    np.testing.assert_allclose(controller.encoder_positions(), [0.13])


def test_velocity_without_a_backlash_joint_is_just_motor_velocity():
    spec = mujoco.MjSpec.from_string(XML.replace("passive_servo_backlash", "free_play"))
    convert_to_motors(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    controller = CpuBamController(model, data, ("motor",))
    data.qpos[controller.servo_qpos] = 0.12
    data.qvel[controller.servo_dof] = 0.7
    data.qpos[model.joint("free_play").qposadr] = 0.017
    np.testing.assert_allclose(controller.encoder_positions(), [0.12])
    np.testing.assert_allclose(controller.encoder_velocities(), [0.7])


def test_runtime_import_does_not_load_torch():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import mjlab_microduck.sim.visualization.bam; "
                "assert 'torch' not in sys.modules and 'mjlab' not in sys.modules"
            ),
        ],
        cwd=Path(__file__).resolve().parents[5],
        check=True,
    )


def test_cpu_controller_is_owned_by_the_visualization_extension():
    from mjlab_microduck.sim.visualization import bam as extension

    assert CpuBamController is extension.CpuBamController
    assert CpuBamController.__module__ == "mjlab_microduck.sim.visualization.bam"
    _, _, controller = make_sim()
    assert isinstance(controller, extension.CpuBamController)
    assert controller.bam_model.actuator.max_current == pytest.approx(1.75)


@pytest.mark.parametrize("values", [[float("nan")], [0, 1], [float("inf")]])
def test_nonfinite_or_wrong_size_targets_are_rejected(values):
    _, _, controller = make_sim()
    with pytest.raises(ValueError, match="finite angle"):
        controller.set_targets(values)


def test_position_actuators_must_be_explicitly_converted():
    model = mujoco.MjModel.from_xml_string(XML)
    with pytest.raises(ValueError, match="convert position actuators"):
        CpuBamController(model, mujoco.MjData(model), ("motor",))
