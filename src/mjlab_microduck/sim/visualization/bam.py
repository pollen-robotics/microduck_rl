"""NumPy BAM actuators with the same friction and encoder contract as training.

The pinned BAM CPU controller treats ``FRICTION_DOF.efc_id`` as a joint ID;
MuJoCo supplies a DOF ID. A floating root and serial backlash joints make
those indices different. Its M6 quadratic budget also differs from the
training implementation. Keep those corrections here rather than changing
site-packages. This module imports neither Torch nor mjlab.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bam.model import Model

STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)


def load_m6(gain: float = 200.0, voltage: float = 7.4) -> Model:
    """Load the training XL330 model, including its firmware current limiter."""
    from bam.model import load_model

    bam = load_model(motor_name="xl330", model="m6")
    bam.actuator.kp = gain
    bam.actuator.vin = voltage
    return bam


def friction_budget(bam: Model, motor, external, velocity) -> np.ndarray:
    """NumPy counterpart of ``bam.mjlab.BamActuator._compute_friction_budget``.

    These BAM training equations (Rhoban BAM, Apache-2.0) intentionally apply
    the M6 quadratic term for same-sign loads too, and use the backdrive
    branch when both load magnitudes are equal. The old CPU Model differs.
    """
    motor, external, velocity = np.broadcast_arrays(motor, external, velocity)
    result = np.full(motor.shape, bam.friction_base.value, dtype=float)
    stribeck = np.zeros_like(result)
    if bam.stribeck:
        stribeck = np.exp(
            -((np.abs(velocity) / bam.dtheta_stribeck.value) ** bam.alpha.value)
        )
        result += stribeck * bam.friction_stribeck.value
    if bam.load_dependent:
        if bam.directional:
            result += np.abs(
                external * bam.load_friction_external.value
                - motor * bam.load_friction_motor.value
            )
            if bam.stribeck:
                result += stribeck * np.abs(
                    external * bam.load_friction_external_stribeck.value
                    - motor * bam.load_friction_motor_stribeck.value
                )
                if bam.quadratic:
                    result += stribeck * np.where(
                        np.abs(motor) > np.abs(external),
                        bam.load_friction_external_quad.value * np.abs(external) ** 2,
                        bam.load_friction_motor_quad.value * np.abs(motor) ** 2,
                    )
        else:
            load = np.abs(external - motor)
            result += bam.load_friction_base.value * load
            if bam.stribeck:
                result += stribeck * bam.load_friction_stribeck.value * load
    return result


def dof_friction_force(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Scatter only DOF-friction constraints onto their actual DOFs."""
    result = np.zeros(model.nv)
    mask = data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)
    # efc_id indexes DOFs for these rows; joint IDs are not interchangeable.
    np.add.at(result, data.efc_id[mask], data.efc_force[mask])
    return result


def convert_to_motors(
    spec: mujoco.MjSpec,
    names: Sequence[str] | None = None,
    *,
    bam_model: Model | None = None,
    voltage_ceiling: float = 8.2,
) -> tuple[str, ...]:
    """Convert selected XML position actuators to the training BAM motor form.

    Training sets the force ceiling from the top of its voltage range, while
    the controller below uses the selected nominal voltage and battery sag.
    Geometry and collision configuration remain the caller's responsibility.
    """
    bam = load_m6() if bam_model is None else bam_model
    selected = None if names is None else set(names)
    found = []
    for actuator in spec.actuators:
        if (selected is not None and actuator.name not in selected) or (
            selected is None and actuator.target.startswith("passive_")
        ):
            continue
        joint = spec.joint(actuator.target)
        actuator.set_to_motor()
        actuator.gear = [1, 0, 0, 0, 0, 0]
        actuator.ctrllimited = False
        actuator.forcelimited = True
        limit = voltage_ceiling * bam.kt.value / bam.R.value
        actuator.forcerange = [-limit, limit]
        joint.armature = float(bam.actuator.get_extra_inertia())
        joint.damping = np.zeros((3, 1))
        joint.frictionloss = 0.0
        joint.solref_friction = STIFF_SOLREF_FRICTION
        joint.solimp_friction = STIFF_SOLIMP_FRICTION
        found.append(actuator.name)
    if selected is not None and selected != set(found):
        raise ValueError(f"missing actuators: {sorted(selected - set(found))}")
    return tuple(found)


class CpuBamController:
    """One duck's named BAM motors; ``update`` runs once per physics step.

    ``desired`` contains output-side encoder targets in ``actuator_names``
    order. Command delay is counted in physics steps, shared by this duck's
    motors and resampled each step, as in training. Sensor observation delay
    is deliberately separate and belongs to the simulated body's clock.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        actuator_names: Sequence[str],
        *,
        bam_model: Model | None = None,
        gain: float = 200.0,
        voltage: float = 7.4,
        voltage_drop_gain: float = 0.1,
        voltage_min: float = 6.0,
        delay_steps: tuple[int, int] = (3, 6),
        seed: int | None = 0,
        prefix: str = "",
    ):
        actuator_names = tuple(actuator_names)
        if not actuator_names or len(set(actuator_names)) != len(actuator_names):
            raise ValueError("actuator names must be nonempty and unique")
        if not 0 <= delay_steps[0] <= delay_steps[1] or any(
            int(value) != value for value in delay_steps
        ):
            raise ValueError("command delay requires nonnegative integer min/max steps")
        if not np.isfinite(gain) or gain < 0:
            raise ValueError("firmware gain must be finite and nonnegative")
        if not all(
            np.isfinite(value) for value in (voltage, voltage_min, voltage_drop_gain)
        ) or not (voltage >= voltage_min > 0 and voltage_drop_gain >= 0):
            raise ValueError(
                "voltage and its floor must be positive; drop gain must be nonnegative"
            )
        self.model, self.data = model, data
        self.actuator_names = tuple(actuator_names)
        self.actuator_ids = np.array(
            [model.actuator(name).id for name in actuator_names]
        )
        joints = model.actuator_trnid[self.actuator_ids, 0]
        if not np.all(model.jnt_type[joints] == mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("BAM motors must actuate hinge joints")
        if not (
            np.all(model.actuator_gainprm[self.actuator_ids, 0] == 1)
            and np.all(
                model.actuator_biastype[self.actuator_ids] == mujoco.mjtBias.mjBIAS_NONE
            )
        ):
            raise ValueError(
                "convert position actuators to torque motors before using BAM"
            )
        self.servo_qpos = model.jnt_qposadr[joints].copy()
        self.servo_dof = model.jnt_dofadr[joints].copy()
        joint_names = [model.joint(int(joint)).name for joint in joints]
        passive = [
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{prefix}passive_{name.removeprefix(prefix)}_backlash",
            )
            for name in joint_names
        ]
        self._backlash_mask = np.array(passive) >= 0
        safe_ids = np.maximum(passive, 0)
        self.backlash_qpos = model.jnt_qposadr[safe_ids].copy()
        self.backlash_dof = model.jnt_dofadr[safe_ids].copy()
        self.bam_model = load_m6(gain, voltage) if bam_model is None else bam_model
        self.bam_model.actuator.kp = gain
        self.bam_model.actuator.vin = voltage
        # Training retains the loaded motor's PWM current limiter (XL330: 1.75 A).
        # The XML force ceiling is only a supply-voltage bound and does not replace
        # that firmware behavior. Preserve custom motor limits here as well.
        self.voltage_drop_gain = voltage_drop_gain
        self.voltage_min = voltage_min
        self.delay_steps = tuple(int(value) for value in delay_steps)
        self.rng = np.random.default_rng(seed)
        self.torque_enabled = True
        self.desired = np.zeros(len(actuator_names))
        self._previous_motor_torque = np.zeros_like(self.desired)
        self._targets = deque(maxlen=self.delay_steps[1] + 1)
        self.reset()

    def encoder_positions(self) -> np.ndarray:
        return self.data.qpos[self.servo_qpos] + (
            self.data.qpos[self.backlash_qpos] * self._backlash_mask
        )

    def encoder_velocities(self) -> np.ndarray:
        return self.data.qvel[self.servo_dof] + (
            self.data.qvel[self.backlash_dof] * self._backlash_mask
        )

    def set_targets(self, values) -> None:
        values = np.asarray(values, dtype=float)
        if values.shape != self.desired.shape or not np.isfinite(values).all():
            raise ValueError("targets must contain one finite angle per named motor")
        self.desired[:] = values

    def set_target(self, name: str, value: float) -> None:
        if not np.isfinite(value):
            raise ValueError("target angle must be finite")
        self.desired[self.actuator_names.index(name)] = value

    def set_gain(self, gain: float) -> None:
        if not np.isfinite(gain) or gain < 0:
            raise ValueError("firmware gain must be finite and nonnegative")
        self.bam_model.actuator.kp = gain

    def set_torque(self, enabled: bool) -> None:
        if self.torque_enabled != bool(enabled):
            self.reset()
        self.torque_enabled = bool(enabled)

    def reset(self) -> None:
        """Forget delayed goals and load sag after placement or a world reset."""
        self.bam_model.reset()
        self.desired[:] = self.encoder_positions()
        self._targets.clear()
        self._targets.extend(self.desired.copy() for _ in range(self._targets.maxlen))
        self._previous_motor_torque[:] = 0
        self.data.ctrl[self.actuator_ids] = 0
        self.last_time = float(self.data.time)

    def update(self) -> np.ndarray:
        """Write motor torque and the training friction budget before ``mj_step``."""
        self._targets.append(self.desired.copy())
        lag = int(self.rng.integers(self.delay_steps[0], self.delay_steps[1] + 1))
        target = self._targets[-1 - lag]
        encoder = self.encoder_positions()
        # Back-EMF and friction see rotor velocity, not output-side backlash velocity.
        velocity = self.data.qvel[self.servo_dof]
        actuator = self.bam_model.actuator
        voltage = actuator.vin
        actuator.vin = max(
            voltage
            - self.voltage_drop_gain * np.abs(self._previous_motor_torque).sum(),
            self.voltage_min,
        )
        try:
            signal = actuator.compute_control(
                target, encoder, velocity, self.model.opt.timestep
            )
            torque = np.asarray(
                actuator.compute_torque(signal, self.torque_enabled, encoder, velocity)
            )
        finally:
            actuator.vin = voltage
        self.data.ctrl[self.actuator_ids] = torque
        self._previous_motor_torque[:] = torque
        external = (
            -self.data.qfrc_bias[self.servo_dof]
            + self.data.qfrc_constraint[self.servo_dof]
            - dof_friction_force(self.model, self.data)[self.servo_dof]
        )
        self.model.dof_frictionloss[self.servo_dof] = friction_budget(
            self.bam_model, self.data.qfrc_actuator[self.servo_dof], external, velocity
        )
        self.model.dof_damping[self.servo_dof] = self.bam_model.friction_viscous.value
        self.last_time = float(self.data.time)
        return torque.copy()
