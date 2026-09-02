"""PPO specialization that preserves a verified backroll parent policy."""

from dataclasses import dataclass

import torch
from rsl_rl.algorithms import PPO

from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg


@dataclass
class AnchoredPpoCfg(PpoWithSymmetryCfg):
    """PPO config with an iteration-level proximal pull to a loaded actor."""

    class_name: str = "mjlab_microduck.tasks.backroll_ppo.AnchoredPPO"
    anchor_retention: float = 0.50
    refresh_anchor_on_load: bool = False


class AnchoredPPO(PPO):
    """Let PPO learn a residual without erasing the loaded actor in one burst.

    After each PPO update, retain ``anchor_retention`` of the actor's parameter
    displacement from the originally loaded parent. The actor normalizer and
    exploration distribution are intentionally excluded: the runner freezes
    the former, while PPO remains free to tune exploration independently.
    """

    def __init__(
        self,
        *args,
        anchor_retention: float = 0.50,
        refresh_anchor_on_load: bool = False,
        **kwargs,
    ):
        if not 0.0 <= anchor_retention <= 1.0:
            raise ValueError("anchor_retention must be between zero and one")
        self.anchor_retention = float(anchor_retention)
        self.refresh_anchor_on_load = bool(refresh_anchor_on_load)
        self._actor_anchor: dict[str, torch.Tensor] | None = None
        super().__init__(*args, **kwargs)

    def _anchored_actor_parameters(self):
        return (
            (name, parameter)
            for name, parameter in self.actor.named_parameters()
            if name.startswith("mlp.")
        )

    def _capture_actor_anchor(self) -> None:
        self._actor_anchor = {
            name: parameter.detach().clone()
            for name, parameter in self._anchored_actor_parameters()
        }

    def _apply_actor_anchor(self) -> float:
        if not self._actor_anchor:
            return 0.0
        delta_sq = torch.zeros((), device=self.device)
        anchor_sq = torch.zeros((), device=self.device)
        with torch.no_grad():
            for name, parameter in self._anchored_actor_parameters():
                anchor = self._actor_anchor[name].to(parameter.device)
                parameter.lerp_(anchor, 1.0 - self.anchor_retention)
                delta_sq += (parameter - anchor).square().sum()
                anchor_sq += anchor.square().sum()
        return float(torch.sqrt(delta_sq / anchor_sq.clamp_min(1.0e-12)).item())

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        loss_dict["actor_anchor_relative_l2"] = self._apply_actor_anchor()
        return loss_dict

    def save(self) -> dict:
        saved_dict = super().save()
        if self._actor_anchor:
            saved_dict["actor_anchor_state_dict"] = {
                name: value.detach().clone()
                for name, value in self._actor_anchor.items()
            }
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        loads_actor = load_cfg is None or load_cfg.get("actor", False)
        if loads_actor:
            stored_anchor = loaded_dict.get("actor_anchor_state_dict")
            if stored_anchor and not self.refresh_anchor_on_load:
                self._actor_anchor = {
                    name: value.detach().clone().to(self.device)
                    for name, value in stored_anchor.items()
                }
            else:
                self._capture_actor_anchor()
        return load_iteration
