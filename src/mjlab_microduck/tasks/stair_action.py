"""Action term that preserves stair reset history after mjlab manager reset."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


class StairHistoryJointPositionAction(JointPositionAction):
    """Restore reset-provided action history after ``ActionManager.reset``."""

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        history = getattr(self._env, "_stair_reset_action_history", None)
        if history is None:
            return

        manager = self._env.action_manager
        current = history["current"][ids]
        manager._action[ids] = current
        manager._prev_action[ids] = history["previous"][ids]
        manager._prev_prev_action[ids] = history["previous_previous"][ids]
        self._raw_actions[ids] = current

        scale = self._scale[ids] if isinstance(self._scale, torch.Tensor) else self._scale
        offset = self._offset[ids] if isinstance(self._offset, torch.Tensor) else self._offset
        self._processed_actions[ids] = current * scale + offset


@dataclass(kw_only=True)
class StairHistoryJointPositionActionCfg(JointPositionActionCfg):
    """Joint position action config with post-manager stair history seeding."""

    def build(self, env) -> StairHistoryJointPositionAction:
        return StairHistoryJointPositionAction(self, env)


def with_stair_history_seed(
    cfg: JointPositionActionCfg,
) -> StairHistoryJointPositionActionCfg:
    """Copy a stock joint-position config into the stair-specific subtype."""
    values = {
        field.name: deepcopy(getattr(cfg, field.name))
        for field in fields(cfg)
        if field.init
    }
    return StairHistoryJointPositionActionCfg(**values)
