"""BAM actuator with per-env friction-magnitude domain randomization.

The canonical ``bam.mjlab.BamActuator`` exposes per-env gain scaling (kp/kd) but
no friction hook, and under BAM MuJoCo's ``dof_frictionloss`` is zeroed in
``edit_spec`` (BAM computes friction itself in ``compute()``). So the stock
``dr.dof_frictionloss`` is a no-op here.

This thin subclass adds a per-env ``friction_scale`` that multiplies BAM's
velocity-INDEPENDENT friction budget (Coulomb + Stribeck + load-dependent) inside
``_compute_friction_budget`` — the term that carries the dominant sim2real
friction uncertainty (stiction / gearbox). The viscous (velocity-proportional)
term is left at nominal; scale it too by overriding ``compute`` if ever needed.

Non-accumulating: ``friction_scale`` is reset to 1.0 then set to a fresh sample
each episode by the ``randomize_bam_friction`` event (see tasks/mdp.py).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
from bam.mjlab import BamActuator, BamActuatorCfg
from mjlab.actuator.actuator import ActuatorCmd


class FrictionDRBamActuator(BamActuator):
    """BamActuator + per-env friction_scale on the BAM friction budget."""

    def initialize(self, mj_model, model, data, device) -> None:
        super().initialize(mj_model, model, data, device)
        # kp_scale is (num_envs, 1); mirror it for a per-env friction multiplier.
        self.friction_scale = torch.ones_like(self.kp_scale)
        self.default_friction_scale = self.friction_scale.clone()

    def _compute_friction_budget(
        self,
        motor_torque: torch.Tensor,
        external_torque: torch.Tensor,
        stribeck_coeff: torch.Tensor,
    ) -> torch.Tensor:
        base = super()._compute_friction_budget(
            motor_torque, external_torque, stribeck_coeff
        )
        fs = getattr(self, "friction_scale", None)
        return base if fs is None else base * fs  # (N, J) * (N, 1)

    def set_friction_scale(self, env_ids, friction_scale: torch.Tensor) -> None:
        self.friction_scale[env_ids] = friction_scale

    def reset_friction_scale(self, env_ids) -> None:
        self.friction_scale[env_ids] = self.default_friction_scale[env_ids]


@dataclass(kw_only=True)
class FrictionDRBamActuatorCfg(BamActuatorCfg):
    """Drop-in for BamActuatorCfg that builds a friction-DR-capable actuator."""

    def build(self, entity, target_ids, target_names) -> FrictionDRBamActuator:
        return FrictionDRBamActuator(self, entity, target_ids, target_names)


class BacklashEncoderBamActuator(FrictionDRBamActuator):
    """FrictionDRBamActuator whose firmware PD reads the encoder THROUGH backlash.

    Backlash models (robot_groundcontact_backlash.xml) put an unactuated
    ``passive_<joint>_backlash`` hinge in series with each servo joint: the
    servo joint is the motor output, the backlash joint is the play between it
    and the link, and the link angle is their sum.

    On the real servo the magnetic encoder sits on the OUTPUT side of that
    play, so the firmware position loop closes on main+backlash — while the
    servo winds through the dead zone the measured position (and hence the PD
    error) doesn't change. This subclass reproduces that: ``cmd.pos`` fed to
    BAM's voltage control law becomes qpos[main] + qpos[backlash].

    ``cmd.vel`` is left motor-side on purpose: in BAM it drives back-EMF and
    friction, which are rotor physics, not an encoder-derived firmware signal.

    Degrades to a plain FrictionDRBamActuator on models without backlash
    joints (per-joint mask), so it is safe to use on any microduck model.
    """

    def initialize(self, mj_model, model, data, device) -> None:
        super().initialize(mj_model, model, data, device)
        name_to_local = {n: i for i, n in enumerate(self.entity.joint_names)}
        ids, mask = [], []
        for name in self._target_names:
            bl_id = name_to_local.get(f"passive_{name}_backlash")
            ids.append(0 if bl_id is None else bl_id)
            mask.append(0.0 if bl_id is None else 1.0)
        self._backlash_joint_ids = torch.tensor(ids, dtype=torch.long, device=device)
        self._backlash_mask = torch.tensor(mask, dtype=torch.float32, device=device)
        n_backlash = int(self._backlash_mask.sum().item())
        print(
            f"[BacklashEncoderBamActuator] encoder-through-backlash feedback on "
            f"{n_backlash}/{len(mask)} joints"
        )

    def get_command(self, data) -> ActuatorCmd:
        cmd = super().get_command(data)
        pos = cmd.pos + data.joint_pos[:, self._backlash_joint_ids] * self._backlash_mask
        return dataclasses.replace(cmd, pos=pos)


@dataclass(kw_only=True)
class BacklashEncoderBamActuatorCfg(FrictionDRBamActuatorCfg):
    """FrictionDRBamActuatorCfg whose PD feedback reads through backlash joints."""

    def build(self, entity, target_ids, target_names) -> BacklashEncoderBamActuator:
        return BacklashEncoderBamActuator(self, entity, target_ids, target_names)
