"""Initialize and reset the stateful Feetech goal limiter in BAM's mjlab bridge.

The pinned BAM version initializes STS3215.q_target_smooth only in load_log(),
which mjlab never calls. Seed from measured position on each world's first
compute after reset, once reset events have set that world's new joint pose.
"""

from dataclasses import dataclass

import torch
from mjlab.actuator.actuator import ActuatorCmd

from .friction_dr_bam import FrictionDRBamActuator, FrictionDRBamActuatorCfg


class FeetechBamActuator(FrictionDRBamActuator):
    def initialize(self, mj_model, model, data, device) -> None:
        super().initialize(mj_model, model, data, device)
        self._bam_model.actuator.q_target_smooth = torch.zeros_like(self._prev_motor_torque)
        self._goal_reset_pending = torch.ones(
            (data.nworld, 1), dtype=torch.bool, device=device
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._goal_reset_pending.fill_(True)
        else:
            self._goal_reset_pending[env_ids] = True

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        actuator = self._bam_model.actuator
        # Use a tensor mask to avoid synchronizing the GPU and leave the other
        # worlds' goal histories intact during a partial reset.
        actuator.q_target_smooth = torch.where(
            self._goal_reset_pending, cmd.pos, actuator.q_target_smooth
        )
        self._goal_reset_pending.zero_()
        return super().compute(cmd)


@dataclass(kw_only=True)
class FeetechBamActuatorCfg(FrictionDRBamActuatorCfg):
    def build(self, entity, target_ids, target_names) -> FeetechBamActuator:
        return FeetechBamActuator(self, entity, target_ids, target_names)
