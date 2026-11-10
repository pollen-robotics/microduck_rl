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

It also supports **per-joint servo models** (``per_joint_motors``), for robots
mixing servo variants (e.g. metal-gear XL330 on the knees / hip pitches / neck
pitch, plastic gears elsewhere). They stay ONE actuator group on purpose: a
BamActuator samples its own battery voltage per env and computes the voltage
sag from its own joints' torques, so splitting the joints into two groups would
give them two uncorrelated batteries each seeing only part of the load. Instead
every BAM parameter becomes a per-joint ``(J,)`` tensor, which the control law,
torque equation and friction budget already broadcast over ``(N, J)``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
from bam.mjlab import BamActuator, BamActuatorCfg
from mjlab.actuator.actuator import ActuatorCmd

from mjlab_microduck import bam_servos


class FrictionDRBamActuator(BamActuator):
    """BamActuator + per-env friction_scale on the BAM friction budget,
    + optional per-joint servo models (see ``FrictionDRBamActuatorCfg``)."""

    def __init__(self, cfg, entity, target_ids, target_names) -> None:
        super().__init__(cfg, entity, target_ids, target_names)
        self._joint_models = None
        if not cfg.per_joint_motors:
            return
        carrier, joint_models, motor_names = bam_servos.resolve_joint_models(
            target_names, cfg.motor_name, cfg.model, cfg.per_joint_motors,
            default_model=self._bam_model,
        )
        if cfg.kp_fw is not None:
            carrier.actuator.kp = cfg.kp_fw
        if cfg.vin is not None:
            carrier.actuator.vin = cfg.vin
        self._bam_model = carrier
        self._joint_models = joint_models
        self._joint_motor_names = motor_names

    def edit_spec(self, spec, target_names) -> None:
        super().edit_spec(spec, target_names)
        if self._joint_models is None:
            return
        # Per-joint armature, and a force ceiling covering every servo variant.
        vin = (
            max(self.cfg.vin_range)
            if self.cfg.vin_range is not None
            else self._bam_model.actuator.vin
        )
        force_limit = max(vin * m.kt.value / m.R.value for m in self._joint_models)
        for mjact in self._mjs_actuators:
            mjact.forcerange = (-force_limit, force_limit)
        by_name = dict(zip(target_names, self._joint_models))
        for joint in spec.joints:
            if joint.name in by_name:
                joint.armature = float(by_name[joint.name].actuator.get_extra_inertia())

    def initialize(self, mj_model, model, data, device) -> None:
        super().initialize(mj_model, model, data, device)
        # kp_scale is (num_envs, 1); mirror it for a per-env friction multiplier.
        self.friction_scale = torch.ones_like(self.kp_scale)
        self.default_friction_scale = self.friction_scale.clone()

        if self._joint_models is None:
            return
        # After super().initialize (which still reads scalars): every BAM
        # parameter becomes a (J,) tensor, in target-joint order.
        bam_servos.vectorize_parameters(
            self._bam_model,
            self._joint_models,
            lambda v: torch.tensor(v, dtype=torch.float32, device=device),
        )
        print(
            "[FrictionDRBamActuator] per-joint servo models: "
            + bam_servos.describe(self._target_names, self._joint_motor_names)
        )

    def _compute_friction_budget(
        self,
        motor_torque: torch.Tensor,
        external_torque: torch.Tensor,
        stribeck_coeff: torch.Tensor,
    ) -> torch.Tensor:
        if self._joint_models is None:
            base = super()._compute_friction_budget(
                motor_torque, external_torque, stribeck_coeff
            )
        else:
            # BamActuator seeds the budget with torch.full_like(..., friction_base),
            # which needs a scalar: run it with 0 and add the per-joint (J,) value.
            friction_base = self._bam_model.friction_base
            per_joint = friction_base.value
            friction_base.value = 0.0
            try:
                base = super()._compute_friction_budget(
                    motor_torque, external_torque, stribeck_coeff
                )
            finally:
                friction_base.value = per_joint
            base = base + per_joint
        fs = getattr(self, "friction_scale", None)
        return base if fs is None else base * fs  # (N, J) * (N, 1)

    def set_friction_scale(self, env_ids, friction_scale: torch.Tensor) -> None:
        self.friction_scale[env_ids] = friction_scale

    def reset_friction_scale(self, env_ids) -> None:
        self.friction_scale[env_ids] = self.default_friction_scale[env_ids]


@dataclass(kw_only=True)
class FrictionDRBamActuatorCfg(BamActuatorCfg):
    """Drop-in for BamActuatorCfg that builds a friction-DR-capable actuator.

    ``per_joint_motors``: ``((joint_name_regex, bundled_motor_name), ...)``.
    Target joints whose name fully matches a regex use that motor's parameters
    (same ``model`` variant, e.g. m6) instead of ``motor_name``'s; the last
    matching entry wins. Empty: every joint uses ``motor_name``.
    """

    per_joint_motors: tuple[tuple[str, str], ...] = ()

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
