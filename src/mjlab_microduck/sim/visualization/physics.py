"""Optional BAM scene and body adapters for Studio's authoritative world.

The body server owns locking and protocol state. These helpers never acquire
its non-reentrant lock, change the wire contract, or import the training stack.
XML bodies do not construct an adapter and retain their original PD behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from mjlab_microduck.sim.visualization.bam import (
    CpuBamController,
    convert_to_motors,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mjlab_microduck.sim.body_server import Body


def configure_bam_spec(
    spec: mujoco.MjSpec, prefixes: Sequence[str], joint_names: Sequence[str]
) -> None:
    """Match robot collision/motor settings, preserving non-robot scene props."""
    counts: dict[str, int] = {}
    roots = {f"{prefix}trunk_base" for prefix in prefixes}
    for geom in spec.geoms:
        owner = geom.parent
        while owner.name not in roots and owner.name != "world":
            owner = owner.parent
        if owner.name not in roots:
            continue
        if geom.meshname.endswith("xl330") and (geom.contype or geom.conaffinity):
            body_name = geom.parent.name
            number = counts.get(body_name, 0)
            counts[body_name] = number + 1
            geom.name = f"{body_name}_{number}_servo_collision"
        if geom.name.endswith("_collision"):
            foot = geom.name.endswith(("left_foot_collision", "right_foot_collision"))
            geom.condim = 3 if foot else 1
            if foot:
                geom.priority = 1
                geom.friction[0] = 1.0
    # Motorized scenery must retain its own position loop and input units.
    names = {f"{prefix}{name}" for prefix in prefixes for name in joint_names}
    convert_to_motors(
        spec,
        tuple(actuator.name for actuator in spec.actuators if actuator.name in names),
    )


def configure_bam_solver(model: mujoco.MjModel, timestep: float) -> None:
    """The v8 training solver profile, independent of rendering backend."""
    model.opt.timestep = timestep
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_AUTO
    model.opt.iterations = 30
    model.opt.ls_iterations = 50
    model.opt.tolerance = 1e-8
    model.opt.ls_tolerance = 0.01
    model.opt.ccd_iterations = 50
    model.opt.impratio = 1.0
    model.opt.gravity[:] = [0, 0, -9.81]


class BamBodyAdapter:
    """Own delayed encoder/placement state for one opt-in BAM body.

    Callers hold the world lock for reads, writes, placement and physics ticks.
    Controller state remains accessible through ``Body.bam`` for older probes.
    """

    def __init__(self, body: Body, gain: float, torque_on: bool):
        self.body = body
        self.model, self.data = body.world.model, body.world.data
        names = [self.model.actuator(index).name for index in body.actuators]
        self.controller = CpuBamController(
            self.model,
            self.data,
            names,
            gain=gain,
            prefix=body.prefix,
            seed=body.index,
        )
        self.controller.set_torque(torque_on)
        self.last_velocity = np.zeros(len(body.actuators))
        self.sensor_velocity = self.last_velocity.copy()
        self.held_backlash = None

    def placed(self) -> None:
        """Clear passive play, delayed feedback and queued goals after placement."""
        motor = self.controller
        mask = motor._backlash_mask
        self.data.qpos[motor.backlash_qpos[mask]] = 0.0
        self.data.qvel[motor.backlash_dof[mask]] = 0.0
        self.last_velocity[:] = self.sensor_velocity[:] = 0.0
        motor.reset()

    def remember(self) -> None:
        self.held_backlash = self.data.qpos[self.controller.backlash_qpos].copy()

    def restore(self) -> None:
        motor = self.controller
        mask = motor._backlash_mask
        self.data.qpos[motor.backlash_qpos[mask]] = self.held_backlash[mask]
        self.data.qvel[motor.backlash_dof[mask]] = 0.0

    def sample_velocity(self) -> None:
        self.sensor_velocity = self.last_velocity.copy()
        self.last_velocity = self.controller.encoder_velocities().copy()

    def positions(self) -> np.ndarray:
        return self.controller.encoder_positions()

    def velocities(self) -> np.ndarray:
        return self.sensor_velocity

    def set_targets(self, wire_targets: Sequence[float]) -> None:
        self.controller.set_targets(
            [wire_targets[index] for index in self.body.to_wire]
        )

    def set_torque(self, on: bool) -> None:
        if on:
            self.controller.reset()
        self.controller.set_torque(on)


class BamWorldAdapter:
    """Clock BAM updates and delayed encoders on physics ticks, never reads."""

    def __init__(self):
        self.steps = 0

    def before_step(self, bodies: Sequence[Body]) -> None:
        for body in bodies:
            body.bam.update()

    def after_step(
        self, model: mujoco.MjModel, data: mujoco.MjData, bodies: Sequence[Body]
    ) -> None:
        self.steps += 1
        if self.steps % 4 == 0:
            # Training refreshes derived/constraint forces after decimation;
            # the next friction update consumes those force fields as well.
            mujoco.mj_forward(model, data)
            for body in bodies:
                body.sample_velocity()
